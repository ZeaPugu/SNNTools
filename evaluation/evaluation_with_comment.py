# -*- coding: utf-8 -*-
# @Time    : 18-11-14 下午2:57
# @Author  : Pugu
# @FileName: evaluation_with_comment.py.py
# @Software: PyCharm

import torch

from itertools import product
from typing import Optional, Tuple, Dict
from sklearn.linear_model import LogisticRegression


# Comments of some evaluation methods

def assign_labels(spikes: torch.Tensor, labels: torch.Tensor, n_labels: int, rates: Optional[torch.Tensor] = None,
                  alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # language=rst
    """
    Assign labels to the neurons based on highest average spiking activity.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a single layer's spiking activity.
    :param labels: Vector of shape ``(n_samples,)`` with data labels corresponding to spiking activity.
    :param n_labels: The number of target labels in the data.
    :param rates: If passed, these represent spike rates from a previous ``assign_labels()`` call.
    :param alpha: Rate of decay of label assignments.
    :return: Tuple of class assignments, per-class spike proportions, and per-class firing rates.
    """
    # NOTICE spikes: 脉冲记录
    # NOTICE labels: 一次更新周期内所有样本对应的标签
    # NOTICE n_labels: 标签个数，10个，0~9
    # NOTICE rates: 脉冲发射频率，每个神经元每个标签的平均脉冲次数
    # NOTICE alpha: 频率记录的衰减（历史rates影响因素）
    n_neurons = spikes.size(2)  # NOTICE 获取神经元个数
    
    if rates is None:  # NOTICE 如果rates不存在（第一次assign可能不存在）
        rates = torch.zeros_like(torch.Tensor(n_neurons, n_labels))  # NOTICE 初始化rates，全零
    
    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)  # NOTICE 对脉冲记录进行时间维度求和，求和后spikes维度为(n_samples, n_neurons)
    
    for i in range(n_labels):  # NOTICE 每个标签
        # Count the number of samples with this label.
        n_labeled = torch.sum(labels == i).float()  # NOTICE 每个周期内标签为i的图片个数
        # NOTICE >>> dd = torch.Tensor([0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3])
        # NOTICE >>> (dd == 0) =
        # NOTICE                tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
        
        if n_labeled > 0:  # NOTICE 如果当前标签i有图片
            # Get indices of samples with this label.
            indices = torch.nonzero(labels == i).view(-1)  # NOTICE 获取标签为i的图片在当前样本中的索引
            # NOTICE >>> dd = torch.Tensor([0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3])
            # NOTICE >>> (dd == 0) =
            # NOTICE                tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
            # NOTICE >>> torch.nonzero(dd == 0) =
            # NOTICE                tensor([[ 0],
            # NOTICE                        [ 9],
            # NOTICE                        [10]])
            # NOTICE >>> torch.nonzero(dd == 0).view(-1)
            # NOTICE                tensor([0, 9, 10])
            
            # Compute average firing rates for this label.
            rates[:, i] = alpha * rates[:, i] + (torch.sum(spikes[indices], 0) / n_labeled)  # NOTICE 计算rates
            # NOTICE torch.sum(spikes[indices], 0) : 每个神经元对同一标签的的脉冲次数总和
            # NOTICE >>> aa = torch.Tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
            # NOTICE                        [ 0.0608,  0.0061,  0.9497,  0.3343],
            # NOTICE                        [ 0.6058,  0.9553,  1.0960,  2.3332]])
            # NOTICE >>> torch.sum(aa, 0) =
            # NOTICE                 tensor([ 0.1595,  0.7452,  2.7176,  2.0882])
            # NOTICE 返回行（第0维度）求和结果
    
    # Compute proportions of spike activity per class.
    proportions = rates / rates.sum(1, keepdim=True)  # NOTICE 从平均脉冲次数计算平均脉冲百分比
    proportions[proportions != proportions] = 0  # Set NaNs to 0 NOTICE 修正proportions中除以0后异常的项
    
    # Neuron assignments are the labels they fire most for.
    assignments = torch.max(proportions, 1)[1]  # NOTICE 获取最高平均脉冲百分比的标签作为绑定
    # NOTICE >>> aa = torch.Tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
    # NOTICE                        [ 0.0608,  0.0061,  0.9497,  0.3343],
    # NOTICE                        [ 0.6058,  0.9553,  1.0960,  2.3332]])
    # NOTICE >>> torch.max(aa, 1) =
    # NOTICE                 (tensor([0.6719, 0.9497, 2.3332]), tensor([2, 2, 3]))
    # NOTICE >>> torch.max(aa, 1)[1] =
    # NOTICE                 tensor([2, 2, 3])
    # NOTICE 返回列（第1维度）的最大值及其索引
    
    return assignments, proportions, rates


def all_activity(spikes: torch.Tensor, assignments: torch.Tensor, n_labels: int) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest average spiking activity over all neurons.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a layer's spiking activity.
    :param assignments: A vector of shape ``(n_neurons,)`` of neuron label assignments.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "all activity" classification scheme.
    """
    # NOTICE spikes: 脉冲记录
    # NOTICE assignments: 神经元的标签绑定，由assign_labels()得到
    # NOTICE n_labels: 标签个数，10个，0~9
    n_samples = spikes.size(0)  # NOTICE 获取单次更新周期的样本个数
    
    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)  # NOTICE 对脉冲记录进行时间维度求和，求和后spikes维度为(n_samples, n_neurons)
    
    rates = torch.zeros(n_samples, n_labels)  # NOTICE 初始化发射频率，有别于绑定函数中的rates，该处rates第一维度为样本个数，且不记录历史信息
    for i in range(n_labels):  # NOTICE 每一标签
        # Count the number of neurons with this label assignment.
        n_assigns = torch.sum(assignments == i).float()  # NOTICE 获取每一标签绑定的神经元个数
        
        if n_assigns > 0:  # NOTICE 如果绑定的神经元个数大于0
            # Get indices of samples with this label.
            indices = torch.nonzero(assignments == i).view(-1)  # NOTICE 获取绑定到标签i的神经元的索引
            
            # Compute layer-wise firing rate for this label.
            rates[:, i] = torch.sum(spikes[:, indices], 1) / n_assigns  # NOTICE 计算所有绑定为i的神经元对每一个样本的平均脉冲次数
            # NOTICE 假设样本 Xk 的标签为 j，则所有绑定到标签 i 的神经元脉冲频率理论上就会相对较低
            # NOTICE 若绑定为 j 的所有神经元对 Xk 反应比 i 神经元低，则对该样本的预测即发生错误
            # NOTICE 该种精确度计算方式并不考虑未绑定任何标签的神经元（但实际上在assign_labels()函数中，所有神经元均会赋予一个标签，不论其是否任何样本做出反应）
    
    # Predictions are arg-max of layer-wise firing rates.
    return torch.sort(rates, dim=1, descending=True)[1][:, 0]
    # NOTICE 对每个样本的10个标签的平均脉冲频率进行降序排序，获取最高脉冲频率对应的标签索引作为当前预测
    # NOTICE >>> ee = torch.rand(3, 10)
    # NOTICE              tensor([[0.7373, 0.7399, 0.4749, 0.5785, 0.5803, 0.0065, 0.5851, 0.9947, 0.4681, 0.8364],
    # NOTICE                      [0.9407, 0.9264, 0.1395, 0.5373, 0.8746, 0.7871, 0.4469, 0.6040, 0.5475, 0.9910],
    # NOTICE                      [0.5748, 0.2231, 0.4587, 0.0506, 0.3413, 0.4430, 0.1063, 0.7016, 0.1799, 0.7781]])
    # NOTICE
    # NOTICE >>> torch.sort(ee, dim=1, descending=True)[1]
    # NOTICE              tensor([[     7,      9,      1,      0,      6,      4,      3,      2,      8,      5],
    # NOTICE                      [     9,      0,      1,      4,      5,      7,      8,      3,      6,      2],
    # NOTICE                      [     9,      7,      0,      2,      5,      4,      1,      8,      6,      3]])
    # NOTICE
    # NOTICE >>> torch.sort(ee, dim=1, descending=True)[1][:, 0]
    # NOTICE              tensor([7, 9, 9])


def proportion_weighting(spikes: torch.Tensor, assignments: torch.Tensor, proportions: torch.Tensor,
                         n_labels: int) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest average spiking activity over all neurons, weighted by class-wise
    proportion.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a single layer's spiking activity.
    :param assignments: A vector of shape ``(n_neurons,)`` of neuron label assignments.
    :param proportions: A matrix of shape ``(n_neurons, n_labels)`` giving the per-class proportions of neuron spiking
                        activity.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "proportion weighting" classification
             scheme.
    """
    # NOTICE spikes: 脉冲记录
    # NOTICE assignments: 神经元的标签绑定，由assign_labels()得到
    # NOTICE proportions: 每个神经元对每种标签的响应百分比（i标签脉冲次数 / 所有标签脉冲次数），由assign_labels()得到
    # NOTICE n_labels: 标签个数，10个，0~9
    n_samples = spikes.size(0)  # NOTICE 获取单次更新周期的样本个数
    
    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)  # NOTICE 对脉冲记录进行时间维度求和，求和后spikes维度为(n_samples, n_neurons)
    
    rates = torch.zeros(n_samples, n_labels)  # NOTICE 初始化发射频率，有别于绑定函数中的rates，该处rates第一维度为样本个数，且不记录历史信息
    for i in range(n_labels):  # NOTICE 每一标签
        # Count the number of neurons with this label assignment.
        n_assigns = torch.sum(assignments == i).float()  # NOTICE 获取每一标签绑定的神经元个数
        
        if n_assigns > 0:  # NOTICE 如果绑定的神经元个数大于0
            # Get indices of samples with this label.
            indices = torch.nonzero(assignments == i).view(-1)  # NOTICE 获取绑定到标签i的神经元的索引
            
            # Compute layer-wise firing rate for this label.
            rates[:, i] += torch.sum((proportions[:, i] * spikes)[:, indices], 1) / n_assigns
            # NOTICE 计算所有绑定为i的神经元对每一个样本的平均脉冲次数，使用proportions作为脉冲次数的权重
            # NOTICE
            # NOTICE 思想是倘若某一神经元 Nk 上一更新期对标签 i 响应最高，则此轮计算时标签 i 权重最高，即此轮 Nk 对标签 i 的脉冲频率保留最多
            # NOTICE 同理，如果上轮中对标签 j 未有任何响应（对应proportions[Nk, i] = 0.），则此轮计算时直接忽略其脉冲响应频率，不作为预测的评定标准
    
    # Predictions are arg-max of layer-wise firing rates.
    predictions = torch.sort(rates, dim=1, descending=True)[1][:, 0]  # NOTICE 对rates进行标签维度排序，返回最高响应的标签作为预测
    
    return predictions
