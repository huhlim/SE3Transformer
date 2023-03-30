#!/usr/bin/env python

import torch
import torch.nn as nn

import dgl

from typing import Optional

from se3_transformer import Fiber, SE3Transformer
from se3_transformer.layers import LinearSE3, NormSE3


class LinearModule(nn.Module):
    """
    Operates only within a node, so it basically applies nn.Linear to every node.
    """

    def __init__(
        self,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        n_layer: Optional[int] = 2,
        use_norm: Optional[bool] = True,
        nonlinearity: Optional[nn.Module] = nn.ReLU(),
        **kwargs,
    ):
        """
        arguments:
        - fiber_in: Fiber, numbers of input features
        - fiber_hidden: Fiber, numbers of intermediate features
        - fiber_out: Fiber, numbers of output features
        - n_layer: int, the number linear layers
        - use_norm: bool, if True, NormSE3 will be inserted before a LinearSE3 layer
        - nonlinearity: activation function for NormSE3
        """

        super().__init__()
        #
        linear_module = []
        #
        if n_layer >= 2:
            if use_norm:
                linear_module.append(NormSE3(Fiber(fiber_in), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(fiber_in), Fiber(fiber_hidden)))
            #
            for _ in range(n_layer - 2):
                if use_norm:
                    linear_module.append(NormSE3(Fiber(fiber_hidden), nonlinearity=nonlinearity))
                linear_module.append(LinearSE3(Fiber(fiber_hidden), Fiber(fiber_hidden)))
            #
            if use_norm:
                linear_module.append(NormSE3(Fiber(fiber_hidden), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(fiber_hidden), Fiber(fiber_out)))
        else:
            if use_norm:
                linear_module.append(NormSE3(Fiber(fiber_init), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(fiber_init), Fiber(fiber_out)))
        #
        self.linear_module = nn.Sequential(*linear_module)

    def forward(self, x):
        return self.linear_module(x)


class InteractionModule(nn.Module):
    """
    Utilization of SE3-Transformer block
    """

    def __init__(
        self,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        fiber_edge: Optional[Fiber] = Fiber({}),
        n_layer: Optional[int] = 2,
        n_head: Optional[int] = 2,
        use_norm: Optional[bool] = True,
        use_layer_norm: Optional[bool] = True,
        nonlinearity: Optional[nn.Module] = nn.ReLU(),
        low_memory: Optional[bool] = True,
        **kwargs,
    ):
        """
        arguments:
        - fiber_in: Fiber, numbers of input features
        - fiber_hidden: Fiber, numbers of intermediate features
        - fiber_out: Fiber, numbers of output features
        - fiber_edge: Fiber, numbers of edge features
        - n_layer: int, the number linear layers
        - n_head: int, the number of attention heads
        - use_norm: bool, if True, NormSE3 will be inserted before a LinearSE3 layer
        - use_layer_norm: bool, if True, LayerNorm will be used between MLP (radial)
        - nonlinearity: activation function for NormSE3
        - low_memory: bool, if True, gradient checkpoint will be activated for ConvSE3
        """

        super().__init__()

        self.graph_module = SE3Transformer(
            num_layers=n_layer,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_heads=n_head,
            channels_div=2,
            fiber_edge=fiber_edge,
            norm=use_norm,
            use_layer_norm=use_layer_norm,
            nonlinearity=nonlinearity,
            low_memory=low_memory,
        )

    def forward(self, batch: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):
        out = self.graph_module(batch, node_feats=node_feats, edge_feats=edge_feats)
        return out
