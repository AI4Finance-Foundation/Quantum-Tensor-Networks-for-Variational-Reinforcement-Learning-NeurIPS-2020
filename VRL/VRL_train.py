import torch
import torch.optim as optim

import numpy as np
import pkbar
import time

import tensorly as tl
from tensorly.decomposition import tensor_train, parafac
from tensorly.tt_tensor import tt_to_tensor
tl.set_backend('pytorch')

import tensornetwork as tn
from tensornetwork import contractors
backend = 'pytorch'

from .K_Spin import K_Spin
from .util import *


def VRL_train(five_tuple, k, data, H, H_core, omega, lr=0.00001, epochs=int(1e4), lam=1):
    s, a, P_mat, R_vec, gamma = five_tuple
    energy_history = []
    Pbar = pkbar.Pbar(name='progress', target=epochs)
    op = optim.SGD([data], lr=lr, momentum=0.9, weight_decay=5e-4)
        
    for e in range(epochs):
        spins = []
        core = []
        edge = []
        energy = 0
        regularity = 0
        
        op.zero_grad()
             
        for i in range(k):
            H_core[i] = tensor_completion(H_core[i], H[i], omega[i])
            core.append(tn.replicate_nodes(H_core[i]))
            edge.append([])
            for c in core[i]:
                edge[i] += c.get_all_dangling()
                
        for i in range(k):
            spins.append(K_Spin(s, a, i + 1, data=data, softmax=True))
            for j in range(i + 1):
                edge[i][j] ^ spins[i].qubits[j][0]

        for i in range(k):
            energy -= contractors.branch(tn.reachable(core[i]), nbranch=1).get_tensor()
        energy_history.append(energy)
        
        for j in range(s):
            regularity += (1 - torch.sum(data[j * a : (j+1) * a], 0)) ** 2
        target = energy + lam * regularity

        target.backward()
        op.step()
        Pbar.update(e)
        
    return spins, energy_history