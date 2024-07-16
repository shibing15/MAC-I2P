from numpy import positive
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def log_sigmoid(x):

    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)

def log_minus_sigmoid(x):

    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)

def match_loss(img_features, pc_features, label, norm_factor):

    corr_map = torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=1) #(B, M, N)
    pos_mask = (label == 1)
    neg_mask = (label == 0)
    pos_num = pos_mask.sum().float()
    neg_num = neg_mask.sum().float()
    weight = label.new_zeros(label.size())
    weight[pos_mask] = 1 / pos_num
    weight[neg_mask] = 1 / neg_num * norm_factor
    weight /= weight.sum()
    return F.binary_cross_entropy_with_logits(corr_map, label, weight, reduction='sum')

#-----------------------------------------------------------------------------------#
def cw_loss(img_score_inline,img_score_outline,pc_score_inline,pc_score_outline):

    loss_inline=torch.mean(1-img_score_inline)+torch.mean(1-pc_score_inline)
    loss_outline=torch.mean(img_score_outline)+torch.mean(pc_score_outline)
    return loss_inline+loss_outline


def cal_acc(img_features,pc_features,mask):
    dist=torch.sum((img_features.unsqueeze(-1)-pc_features.unsqueeze(-2))**2,dim=1) #(B,N,N)
    furthest_positive,_=torch.max(dist*mask,dim=1)
    closest_negative,_=torch.min(dist+1e5*mask,dim=1)
    '''print(furthest_positive)
    print(closest_negative)
    print(torch.max(torch.sum(mask,dim=1)))
    assert False'''
    diff=furthest_positive-closest_negative
    accuracy=(diff<0).sum(dim=1)/dist.size(1)
    return accuracy