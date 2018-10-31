# -*- coding: utf-8 -*-
# @Time    : 18-10-31 上午11:04
# @Author  : Pugu
# @FileName: diehl_cook.py
# @Software: PyCharm

from typing import Dict

import torch

from bindsnet.network import *
from bindsnet.network.nodes import *
from bindsnet.network.topology import *
from bindsnet.learning import PostPre


class RowColDiehlCook(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing the inhibitory layer and
    replacing it with a recurrent inhibitory connection in the output layer (what used to be the excitatory layer).
    """
    
    def __init__(self, n_inpt: int, n_neurons: int = 100, inh: float = 17.5, dt: float = 1.0, nu_pre: float = 1e-4,
                 nu_post: float = 1e-2, wmin: float = None, wmax: float = None, norm: float = 78.4,
                 theta_plus: float = 0.05, theta_decay: float = 1e-7) -> None:
        # language=rst
        """
        Constructor for class ``RowColDiehlCook``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu_pre: Pre-synaptic learning rate.
        :param nu_post: Post-synaptic learning rate.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane threshold potential.
        :param theta_decay: Time constant of ``DiehlAndCookNodes`` threshold potential decay.
        """
        super().__init__(dt=dt)
        
        # self.n_inpt = n_inpt
        self.ww = self.hh = int(np.sqrt(n_inpt))
        if self.ww * self.hh != n_inpt:
            raise AttributeError("the shape of input is not square")
        self.n_inpt = 2 * n_inpt
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt
        
        input_layer = Input(n=self.n_inpt, traces=True, trace_tc=5e-2)
        self.add_layer(input_layer, name="X")
        
        output_layer = DiehlAndCookNodes(
            n=self.n_neurons, traces=True, rest=-65.0, reset=-60.0, thresh=-52.0, refrac=5,
            decay=1e-2, trace_tc=5e-2, theta_plus=theta_plus, theta_decay=theta_decay
        )
        self.add_layer(output_layer, name="Y")
        
        w_conn_1 = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        
        input_connection = Connection(
            source=self.layers["X"], target=self.layers["Y"], w=w_conn_1, update_rule=PostPre,
            nu=(nu_pre, nu_post), wmin=wmin, wmax=wmax, norm=norm
        )
        
        self.add_connection(input_connection, source="X", target="Y")
        
        w_conn_2 = -self.inh * (torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons)))
        recurrent_connection = Connection(
            source=self.layers["Y"], target=self.layers["Y"], w=w_conn_2, wmin=-self.inh, wmax=0
        )
        self.add_connection(recurrent_connection, source="Y", target="Y")
        self.summary()
        pass
    
    def summary(self, ):
        print(self.layers)
        for k in self.layers:
            print()
            print("<><> layer name:", k)
            print("     number of neurons:", self.layers[k].n)
            pass
        pass
    
    def run(self, inpts: Dict[str, torch.Tensor], time: int, **kwargs):
        for key in inpts:
            inpt_row: torch.Tensor = inpts[key]
            tc, nc = inpt_row.shape
            inpt_col = inpt_row.reshape((tc, self.ww, self.hh))
            inpt_col = inpt_col.transpose(1, 2).reshape((tc, nc))
            true_inpt = torch.cat((inpt_row, inpt_col), 1)
            inpts[key] = true_inpt
        super().run(inpts, time)
        
        pass
