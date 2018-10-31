# -*- coding: utf-8 -*-
# @Time    : 18-10-31 上午10:46
# @Author  : Pugu
# @FileName: decode.py
# @Software: PyCharm

import numpy as np


def de_poisson(s: np.ndarray, a: float = 1000, shape_to: tuple = None):
    """
    Decode Poisson spike trains to images for visualization.
    :param s: spike trains
    :param a: parameter for expanding the probability of Poisson distribution
    :param shape_to: return a image-like array with this shape
    :return:
    """
    t, n = s.shape
    
    if shape_to is None:
        w = h = int(np.sqrt(n))
        if w * h != n:
            raise AttributeError("shape_to is None")
    else:
        w, h = shape_to
    
    sum_of_neurons = s.sum(0)
    list_p_per_neurons = []
    for i in range(n):
        list_p_per_neurons.append(float(sum_of_neurons[i]) / t)
        pass
    
    out: np.ndarray = np.array(list_p_per_neurons)
    out = (out * a).clip(0, 255).astype(np.uint8)
    out = out.reshape((w, h))
    
    return out
