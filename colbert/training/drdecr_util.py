import torch


def calculate_distance(student_out, teacher_out):
    '''calculate the distance between student output tokens and teacher output tokens'''
    # start = time.time()
    prod = teacher_out.matmul(student_out.transpose(1, 2))
    student_out_norm = torch.norm(student_out, p=2, dim=-1)
    teacher_out_norm = torch.norm(teacher_out, p=2, dim=-1)
    m = teacher_out_norm.unsqueeze(2) * student_out_norm.unsqueeze(1)
    esp = torch.ones_like(m) * 10**-8
    distance = torch.ones_like(m) - prod / (m + esp)
    # end = time.time()
    # print("time to calculation distance matrix: ", end - start)
    return distance

