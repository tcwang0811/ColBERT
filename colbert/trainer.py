from colbert.infra.run import Run
from colbert.infra.launcher import Launcher
from colbert.infra.config import ColBERTConfig, RunConfig

from colbert.training.training import train
from colbert.training.training_drdecr import train_drdecr


class Trainer:
    def __init__(self, triples, queries, collection, config=None):
        self.config = ColBERTConfig.from_existing(config, Run().config)

        self.triples = triples
        self.queries = queries
        self.collection = collection

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def train(self, checkpoint='bert-base-uncased'):
        """
            Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.
        """

        # Resources don't come from the config object. They come from the input parameters.
        # TODO: After the API stabilizes, make this "self.config.assign()" to emphasize this distinction.
        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)

        launcher = Launcher(train)

        self._best_checkpoint_path = launcher.launch(self.config, self.triples, self.queries, self.collection)


    def best_checkpoint_path(self):
        return self._best_checkpoint_path



class DrDECRTrainer:
    def __init__(self, triples, queries, collection, config=None,
                  teacher_triples=None, teacher_queries=None, teacher_collection=None, teacher_config=None):
        self.config = ColBERTConfig.from_existing(config, Run().config)

        self.triples = triples
        self.queries = queries
        self.collection = collection

        self.teacher_triples = teacher_triples
        self.teacher_queries = teacher_queries
        self.teacher_collection = teacher_collection
        if teacher_config is not None:
            self.teacher_config = ColBERTConfig.from_existing(teacher_config, Run().config)
        else:
            self.teacher_config = teacher_config

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def teacher_configure(self, **kw_args):
        self.teacher_config.configure(**kw_args)

    def train(self, checkpoint='bert-base-uncased', teacher_checkpoint=None):
        """
            Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.
        """

        # Resources don't come from the config object. They come from the input parameters.
        # TODO: After the API stabilizes, make this "self.config.assign()" to emphasize this distinction.
        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)

        if (self.teacher_config is not None) and (teacher_checkpoint is not None):
            self.teacher_configure(triples=self.teacher_triples, queries=self.teacher_queries, collection=self.teacher_collection)
            self.teacher_configure(checkpoint=teacher_checkpoint)

        launcher = Launcher(train_drdecr)

        self._best_checkpoint_path = launcher.launch(self.config, self.triples, self.queries, self.collection,
                                                     self.teacher_config, self.teacher_triples, self.teacher_queries, self.teacher_collection)


    def best_checkpoint_path(self):
        return self._best_checkpoint_path
