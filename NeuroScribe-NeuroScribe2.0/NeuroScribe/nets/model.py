

import numpy as np
import torch
import torch.nn as nn
from .encoder import MergedModel
from braindecode.models import EEGNetv4,ATCNet
import torch.nn.functional as F
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))


grandparent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))


sys.path.append(grandparent_dir)
from dmp.utils.utils import init
from dmp.utils import pytorch_util as ptu
from dmp.utils.dmp_layer import DMPIntegrator, DMPParameters



class NeuroScribe(nn.Module):
    """
        NeuroScribe
    """

    def __init__(self,
                 init_w=3e-3,
                 layer_sizes=[784, 500, 100],
                 hidden_activation=F.relu,
                 pt=None,
                 output_activation=torch.tanh,
                 hidden_init=ptu.fanin_init,
                 b_init_value=0.1,
                 state_index=np.arange(1),
                 N=5,  # N of basis functions
                 T=10,  # Rollout length
                 l=10,  #
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.N = N
        self.l = l
        # e.g. 30 * 2 + 2 * 2 = 64
        self.output_size = N * len(state_index) + 2 * len(state_index)
        output_size = self.output_size
        self.T = T
        self.output_activation = output_activation
        self.state_index = state_index
        self.output_dim = output_size

        tau = 1
        dt = 1.0 / (T * self.l)
        #dt=0.01

        self.output_activation = torch.tanh

        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None)
        self.func = DMPIntegrator()
        self.L = nn.Linear(404, 1)

        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)

        # middles layers
        layer_sizes = [784, 500, 100, 500, 2 * output_size, output_size]

        self.hidden_activation = hidden_activation

        self.TSFEEGNet=MergedModel(n_classes=404,n_channels=5,input_window_size=384)

        # load the pre-trained net's parameters
        #self.pt.load_state_dict(torch.load(pt, map_location=torch.device('cpu')))
        self.convSize = 4 * 4 * 50
        self.imageSizex = 4000
        self.imageSizey=6

        self.middle_layers = []  # 4 layers
        for i in range(1, len(layer_sizes) - 1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            hidden_init(layer.weight)
            layer.bias.data.fill_(b_init_value)
            self.middle_layers.append(layer)
            # add custom module
            self.add_module('middle_layer_' + str(i), layer)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.last_fc = init_(nn.Linear(layer_sizes[-1], output_size))

    def forward(self, input, y0, return_preactivations=False):

        output=self.TSFEEGNet(input)
        s=self.L(output)


        y0 = y0.reshape(-1, 1)[:, 0]
        dy0 = torch.zeros_like(y0) + 0.01  # [batch size * 2], [0.01000, 0.0100, ...]

        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0,s=s)

        # the goal y
        y = y.view(input.shape[0], len(self.state_index), -1)
        return y.transpose(2, 1)
