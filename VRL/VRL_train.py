import torch
import torch.optim as optim

import numpy as np
import pkbar
import time

import tensorly as tl
from tensorly.decomposition import matrix_product_state, parafac
from tensorly.mps_tensor import mps_to_tensor
tl.set_backend('pytorch')

import tensornetwork as tn
from tensornetwork import contractors
backend = 'pytorch'

from .K_Spin import K_Spin


def VRL_train_combined(five_tuple, k, data, combined_cores, lr=0.02, epochs=5000):
    s, a, P_mat, R_vec, gamma = five_tuple
    energy_history = []
    Pbar = pkbar.Pbar(name='progress', target=epochs)
    op = optim.SGD([data], lr=lr, momentum=0.9, weight_decay=5e-4)

    for e in range(epochs):
        op.zero_grad()
        new_edges = []
        new_cores = tn.replicate_nodes(combined_cores)
        
        for j in range(len(combined_cores)):
            edges = new_cores[j].get_all_dangling()
            for edge in edges:
                new_edges.append(edge)
        spin = K_Spin(s, a, k, data=data)
        
        for j in range(k):
            new_edges[j] ^ spin.qubits[j][0]
        energy = -contractors.branch(tn.reachable(new_cores[0]), nbranch=2).get_tensor()
        energy.backward()
        energy_history.append(energy)
        op.step()
        Pbar.update(e)
        
    return spin, energy_history


def VRL_train(five_tuple, k, data, H_core, lr=0.00001, epochs=int(1e4)):
    s, a, P_mat, R_vec, gamma = five_tuple
    energy_history = []
    Pbar = pkbar.Pbar(name='progress', target=epochs)
    op = optim.SGD([data], lr=lr, momentum=0.9, weight_decay=5e-4)
    
    for e in range(epochs):
        spins = []
        core = []
        edge = []
        energy = 0
        op.zero_grad()

        for i in range(k):
            core.append(tn.replicate_nodes(H_core[i]))
            edge.append([])
            for j in range(len(core[i])):
                edge[i] += core[i][j].get_all_dangling()

        for i in range(k):
            spins.append(K_Spin(s, a, i + 1, data=data, mode='distributed'))
            for j in range(i + 1):
                edge[i][j] ^ spins[i].qubits[j][0]

        for i in range(k):
            energy -= contractors.branch(tn.reachable(core[i]), nbranch=1).get_tensor()
        energy.backward()
        energy_history.append(energy)

        op.step()
        Pbar.update(e)
        
    return spins, energy_history