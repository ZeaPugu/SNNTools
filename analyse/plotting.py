# -*- coding: utf-8 -*-
# @Time    : 18-10-31 下午9:08
# @Author  : Pugu
# @FileName: plotting.py
# @Software: PyCharm

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Optional, Tuple, List

from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance


class Keys:
    ALL = "all"
    PROP = "proportion"
    NG = "ngram"


def plot_comparison(curves_list: List[Dict[str, List[float]]], names: List[str] = None,
                    axes: Optional[Axes] = None,
                    figsize: Tuple[int, int] = (7, 4), dx: int = 250):
    if len(curves_list) > 0:
        num_curves = len(curves_list)
        if isinstance(curves_list[0], dict):
            def check(keys, i):
                if i == 0:
                    return check(list(curves_list[0].keys()), i + 1)
                if i >= num_curves:
                    return True
                if isinstance(curves_list[i], dict):
                    ks = list(curves_list[i].keys())
                    if len(list(set(ks).difference(set(keys)))) == 0 and len(list(set(keys).difference(set(ks)))) == 0:
                        return check(keys, i + 1)
                    else:
                        return False
                else:
                    return False
            
            if not check([], 0):
                assert "The list of curves is not correct"
                return None
            pass
        else:
            assert "The list of curves is not correct"
            return None
        pass
    else:
        assert "The list of curves is empty"
        return None
    names = [] if names is None else names
    while len(names) < num_curves:
        names.append("curve" + str(len(names) + 1))
    keys = list(curves_list[0].keys())
    num_keys = len(keys)
    xticks = len(curves_list[0][keys[0]]) * dx
    xstep = int(xticks // 10)
    figsize = (figsize[0], figsize[1] * num_keys)
    if not axes:
        _, axes = plt.subplots(nrows=num_keys, figsize=figsize)
    else:
        axes.clear()
    for i, key in enumerate(keys):
        for j, curves in enumerate(curves_list):
            axes[i].plot(np.arange(0, xticks, step=dx), [p for p in curves[key]], label=names[j])
        axes[i].set_ylim(bottom=0, top=100)
        axes[i].set_title(key)
        axes[i].set_xlabel("No. of examples")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_xticks(range(0, xticks + xstep, xstep))
        axes[i].set_yticks(range(0, 110, 10))
        axes[i].legend()
        pass
    
    return axes
