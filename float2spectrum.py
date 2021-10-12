#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class float2spectrum(nn.Module):
    def __init__(self, channels):
        super(float2spectrum, self).__init__()
        self.thida = torch.linspace(0,2*math.pi,int(channels/2))
        
    def forward(self, tensor):
        k_thida = torch.einsum("nm,k->nmk", tensor, self.thida)
        emb_x = torch.cat((k_thida.cos(),k_thida.sin()), dim=-1)
        return emb_x
    
    
'''
tensor = torch.arange(0,6)
tensor_b = 0.1*tensor
tensor_c = -0.1*tensor
x = torch.stack([tensor,tensor_b,tensor_c])

a = float2spectrum(96)
y = a(x)

 
'''