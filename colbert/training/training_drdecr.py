import time
import torch
import random
import torch.nn as nn
import numpy as np
import copy

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints

from colbert.training.drdecr_util import calculate_distance

def align(maxlen, student_out, teacher_out, teacher_queries):
    '''re-order teacher output tokens so that it aligns with
    student output tokens with greedy search'''
    batch_distance_array = calculate_distance(student_out, teacher_out)
    batch_distance_array = batch_distance_array.cpu().detach().numpy()
    for idx, distance_array in enumerate(batch_distance_array):
        swaps = []
        for i in range(maxlen):
            minValue = np.amin(distance_array)
            indexs = np.where(distance_array == np.amin(minValue))
            #get the index of the first min value
            i, j = indexs[0][0], indexs[1][0]
            #swap arrary row i and row j
            distance_array[[i, j]] = distance_array[[j, i]]
            distance_array[
                j, :] = 10  #anything larger than 1 to avoid double count
            distance_array[:, j] = 10
            swaps.append((i, j))
        for swap in swaps:
            teacher_out[idx][[swap[0],
                              swap[1]]] = teacher_out[idx][[swap[1], swap[0]]]
            teacher_queries[0][idx][[swap[0],
                                     swap[1]]] = teacher_queries[0][idx][[
                                         swap[1], swap[0]
                                     ]]

def train_drdecr(config: ColBERTConfig, triples, queries=None, collection=None, teacher_config=None, teacher_triples=None, teacher_queries=None, teacher_collection=None):
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    if config.rank < 1:
        config.help()

    assert not ( config.use_ib_negatives and config.distill_query_passage_separately ) , f" Simultaneous use of --use_ib_negatives and --distill_query_passage_separately options is not supported (yet)"

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    if teacher_config is not None:
        teacher_reader = LazyBatcher(teacher_config, teacher_triples, teacher_queries, teacher_collection, (0 if config.rank == -1 else config.rank), config.nranks)
        teacher_colbert = ColBERT(name=teacher_config.checkpoint, colbert_config=teacher_config)
        teacher_colbert.to(DEVICE)
        if config.distill_query_passage_separately:
            #assert False, "distill_query_passage_separately functionality is not supported (yet)"
            print_message("distill_query_passage_separately functionality is not supported (yet)")
            if config.loss_function == 'MSE':
                student_teacher_loss_fct = torch.nn.MSELoss()
            else:
                student_teacher_loss_fct = torch.nn.L1Loss()
        else:
            student_teacher_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        
        teacher_colbert = torch.nn.parallel.DistributedDataParallel(teacher_colbert, device_ids=[config.rank],
                                                                    output_device=config.rank,
                                                                    find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    if teacher_config is not None:
        for batch_idx, BatchSteps, teacher_BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader, teacher_reader):
            if (warmup_bert is not None) and warmup_bert <= batch_idx:
                set_bert_grad(colbert, True)
                warmup_bert = None

            this_batch_loss = 0.0

            for queries_passages, teacher_queries_passages in zip(BatchSteps, teacher_BatchSteps):
                # assert(teacher_config is not None or torch.equal(queries_passages[1][0], teacher_queries_passages[1][0]))

                with amp.context():
                    if config.distill_query_passage_separately :
                        if config.query_only:
                            assert False, "Training with --query-only option is not supported (yet)."
                        else:
                            queries, passages, target_scores = queries_passages
                            encoding = [queries, passages]
                            scores, student_output_q, student_output_p = colbert(*encoding)

                            with torch.no_grad():
                                teacher_queries, teacher_passages, teacher_target_scores = teacher_queries_passages
                                teacher_encoding = [teacher_queries, teacher_passages]
                                teacher_scores, teacher_output_q, teacher_output_p  = teacher_colbert(*teacher_encoding)

                            teacher_queries_toks_masks = (teacher_queries_passages[0][0].repeat_interleave(config.nway, dim=0).contiguous(), teacher_queries_passages[0][1].repeat_interleave(config.nway, dim=0).contiguous())
                            teacher_queries = copy.deepcopy(teacher_queries_toks_masks)
                            maxlen = config.query_maxlen
                            align(maxlen, student_output_q, teacher_output_q, teacher_queries)
                            loss = config.query_weight * student_teacher_loss_fct(student_output_q, teacher_output_q) + (1 - config.query_weight)*student_teacher_loss_fct(student_output_p, teacher_output_p)
                    else:
                        try:
                            queries, passages, target_scores = queries_passages
                            encoding = [queries, passages]
                        except:
                            encoding, target_scores = queries_passages
                            encoding = [encoding.to(DEVICE)]

                        scores = colbert(*encoding)

                        if config.use_ib_negatives:
                            scores, ib_loss = scores

                        scores = scores.view(-1, config.nway)

                        with torch.no_grad():
                            try:
                                teacher_queries, teacher_passages, teacher_target_scores = teacher_queries_passages
                                teacher_encoding = [teacher_queries, teacher_passages]
                            except:
                                teacher_encoding, teacher_target_scores = teacher_queries_passages
                                teacher_encoding = [teacher_encoding.to(DEVICE)]

                            teacher_scores = teacher_colbert(*teacher_encoding)

                            if config.use_ib_negatives:
                                teacher_scores, teacher_ib_loss = teacher_scores

                            teacher_scores = teacher_scores.view(-1, config.nway)

                        loss = student_teacher_loss_fct(
                                    torch.nn.functional.log_softmax(scores / config.student_teacher_temperature, dim=-1),
                                    torch.nn.functional.softmax(teacher_scores / config.student_teacher_temperature, dim=-1),
                                ) * (config.student_teacher_temperature ** 2)

                    loss = loss / config.accumsteps

                if config.rank < 1:
                    print_progress(scores.view(-1,2) if config.distill_query_passage_separately else scores)

                amp.backward(loss)

                this_batch_loss += loss.item()

            train_loss = this_batch_loss if train_loss is None else train_loss
            train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

            amp.step(colbert, optimizer, scheduler)

            if config.rank < 1:
                print_message(batch_idx, train_loss)
                manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)
    else:
        for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
            if (warmup_bert is not None) and warmup_bert <= batch_idx:
                set_bert_grad(colbert, True)
                warmup_bert = None

            this_batch_loss = 0.0

            for batch in BatchSteps:
                with amp.context():
                    try:
                        queries, passages, target_scores = batch
                        encoding = [queries, passages]
                    except:
                        encoding, target_scores = batch
                        encoding = [encoding.to(DEVICE)]

                    scores = colbert(*encoding)

                    if config.use_ib_negatives:
                        scores, ib_loss = scores

                    scores = scores.view(-1, config.nway)

                    if len(target_scores) and not config.ignore_scores:
                        target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                        target_scores = target_scores * config.distillation_alpha
                        target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                        loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                    else:
                        loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                    if config.use_ib_negatives:
                        if config.rank < 1:
                            print('\t\t\t\t', loss.item(), ib_loss.item())

                        loss += ib_loss

                    loss = loss / config.accumsteps

                if config.rank < 1:
                    print_progress(scores)

                amp.backward(loss)

                this_batch_loss += loss.item()

            train_loss = this_batch_loss if train_loss is None else train_loss
            train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

            amp.step(colbert, optimizer, scheduler)

            if config.rank < 1:
                print_message(batch_idx, train_loss)
                manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.



def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
