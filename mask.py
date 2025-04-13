# import library
import pandas as pd
import numpy as np
import torch
import random
import math


def get_pad_mask(seq, pad_idx):
    # 如果某个位置的值不等于填充值，它对应的布尔值为 True，否则为 False
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    len_s = seq.size(1)
    subsequent_mask = (1 - torch.triu(
        # 生成一个上三角矩阵，其中主对角线及其上方的元素为1，其他元素为0
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask
