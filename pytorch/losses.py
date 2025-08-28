import torch
import torch.nn.functional as F
import torch.nn as nn

def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])

def loss_KD(student_dict, teacher_dict):
    T = 2
    ce_loss = nn.CrossEntropyLoss()
    soft_targets = nn.functional.softmax(teacher_dict['clipwise_output'] / T, dim=-1) # for panns equivalent models
    #soft_targets = nn.functional.softmax(teacher_dict/ T, dim=-1)  # for passt model  # log is applied while calculating loss...
    soft_prob = nn.functional.log_softmax(student_dict['clipwise_output'] / T, dim=-1)

    # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
    soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
    # Calculate the true label loss
    #label_loss = ce_loss(student_logits, labels)
    return soft_targets_loss


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    if loss_type == 'loss_KD':
        return loss_KD      
