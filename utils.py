import torch

def CountEntropy(logits):
    '''计算信息熵'''
    # pred:[b,c]
    pred = torch.softmax(logits, dim=-1)
    entropy = - torch.mean(pred * torch.log(pred + 1e-8))
    return entropy