import torch
import torch.nn.functional as F

def make_soft_porb(prob, policy_mask, eps=1e-7, t=4):
    soft_prob = (prob + eps) * policy_mask
    soft_prob = torch.pow(soft_prob, 1/t)
    soft_prob /= torch.sum(soft_prob, dim=1, keepdim=True)
    return soft_prob

def cross_entropy(pred, target, weight=1.):
    loss_sum = -torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1)
    return torch.mean(weight * loss_sum, dim=0)

def huber_loss(x, y, delta, weight=1.):
    absdiff = torch.abs(x - y)
    loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
    loss_sum = torch.sum(loss, dim=1)
    return torch.mean(weight * loss_sum, dim=0)

def mse_loss(pred, target, weight=1.):
    loss_sum = torch.mean(torch.square(pred - target), dim=1)
    return torch.mean(weight * loss_sum, dim=0)

def mse_loss_spat(pred, target, mask_sum_hw, weight=1.):
    loss_sum = torch.sum(torch.square(pred - target), dim=1) / mask_sum_hw
    return torch.mean(weight * loss_sum, dim=0)

def square_huber_loss(pred, x, y, delta, eps, weight=1.):
    sqerror = torch.square(x - y) + eps
    loss = huber_loss(pred, sqerror, delta=delta, weight=weight)
    return loss
