#!/usr/bin/env python

import torch
import torch.nn as nn

import dgl

import sys

from se3_transformer import Fiber, LinearModule, InteractionModule
from se3_transformer.utils import degree_to_dim


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        self.linear = LinearModule(**config["linear"])
        self.interact = InteractionModule(**config["interact"])

    def forward(self, batch: dgl.DGLGraph):
        edge_feats = {}
        node_feats = {str(degree): batch.ndata[f"node_feat_{degree}"] for degree in [0, 1]}
        #
        out = self.linear(node_feats)
        out = self.interact(batch, node_feats=out, edge_feats=edge_feats)
        return out


def create_random_example(n_point, fiber_in):
    # create a fully connected graph
    edges = [[], []]
    for i in range(n_point):
        for j in range(n_point):
            edges[0].append(i)
            edges[1].append(j)
    edges = tuple([torch.as_tensor(x) for x in edges])
    g = dgl.graph(edges)
    #
    pos = torch.randn((n_point, 3))
    g.ndata["pos"] = pos[:, None, :]
    for fiber in fiber_in:
        dim = degree_to_dim(fiber.degree)
        g.ndata[f"node_feat_{fiber.degree}"] = torch.randn((n_point, fiber.channels, dim))
    #
    src, dst = g.edges()
    g.edata["rel_pos"] = pos[dst] - pos[src]
    return g


def main():
    config = {}
    #
    config["linear"] = {}
    config["linear"]["fiber_in"] = Fiber([(0, 8), (1, 4)])
    config["linear"]["fiber_hidden"] = Fiber([(0, 16), (1, 8)])
    config["linear"]["fiber_out"] = Fiber([(0, 16), (1, 8)])
    config["linear"]["n_layer"] = 2
    config["linear"]["use_norm"] = True
    config["linear"]["nonlinearity"] = nn.ReLU()
    #
    config["interact"] = {}
    config["interact"]["fiber_in"] = Fiber([(0, 16), (1, 8)])
    config["interact"]["fiber_hidden"] = Fiber([(0, 16), (1, 8)])
    config["interact"]["fiber_out"] = Fiber([(0, 2), (1, 1)])
    config["interact"]["fiber_edge"] = Fiber({})
    config["interact"]["n_layer"] = 2
    config["interact"]["n_head"] = 2
    config["interact"]["use_norm"] = True
    config["interact"]["use_layer_norm"] = True
    config["interact"]["nonlinearity"] = nn.ReLU()
    config["interact"]["low_memory"] = True
    #
    model = Model(config)
    #
    batch = create_random_example(n_point=10, fiber_in=config["linear"]["fiber_in"])
    out = model(batch)
    print(out)
    print(out["0"].size())  # = (n_point, 2, 1)
    print(out["1"].size())  # = (n_point, 1, 3)


if __name__ == "__main__":
    main()
