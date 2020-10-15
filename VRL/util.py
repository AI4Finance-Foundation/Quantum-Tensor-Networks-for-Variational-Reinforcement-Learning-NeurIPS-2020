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


def initialize_H(five_tuple, k):
    s, a, P_mat, R_vec, gamma = five_tuple
    h = torch.zeros([s * a] * k, dtype=torch.float32)
    pbar = pkbar.Pbar(name='initialize H, k='+str(k), target=(s * a))
    
    for i in range(s * a):
        pbar.update(i)
        in_edge = torch.sum(P_mat[..., i // a])
        
        if k == 1:
            h[i] = 1
        if k == 2:
            for j in range(s * a):
                h[i, j] = P_mat[i, j]
        if k == 3:
            for j in range(s * a):
                for l in range(s * a):
                    h[i, j, l] = P_mat[i, j] * P_mat[j, l]   
        if k == 4:
            for j in range(s * a):
                for l in range(s * a):
                    for m in range(s * a):
                        h[i, j, l, m] = P_mat[i, j] * P_mat[j, l] * P_mat[l, m]
                        
        h[i, ...] *= in_edge
    for n in range(s * a):
        h[..., n] *= R_vec[n]
        
    h.requires_grad = False
    return tn.Node(h * gamma ** (k - 1), backend=backend)


def put_mps(tensors):
    '''
    returns
        a set of tensor cores connected in MPS
        a set of connected edges
    '''
    mps = []
    for i in range(len(tensors)):
        mps.append(tn.Node(tensors[i].detach().clone(), backend=backend))
    
    if len(tensors) == 1:
        return mps, []
    
    connected_edges = []
    conn = mps[0][1] ^ mps[1][0]
    for k in range(1, len(tensors)-1):
        conn = mps[k][2] ^ mps[k+1][0]
        connected_edges.append(conn)

    return mps, connected_edges


def put_cp(factors):
    '''
    returns
        a set of tensor cores connected in cp
        a set of connected edges
    '''
    cp = []
    for f in factors:
        cp.append(tn.Node(f.detach().clone(), backend=backend))
    
    chi = factors[0].shape[1]
    core = torch.zeros([chi] * len(factors))
    for i in range(chi):
        index = tuple([i] * len(factors))
        core[index] = 1
    core = tn.Node(core, backend=backend)
    cp.append(core)
    
    connected_edges = []
    for k in range(len(factors)):
        conn = cp[k][1] ^ core[k]
        connected_edges.append(conn)

    return cp, connected_edges


def fill_dims(tensor, dim):
    N = tensor.shape[0]
    order = len(tensor.shape)
    if order == dim:
        return tensor
    pad = torch.zeros([N] * dim, dtype=torch.float32)
    for i in range(N ** order):
        index = np.unravel_index(i, tensor.shape)
        pad[index] = tensor[index]      
    return pad


def build_combined_network(five_tuple, k, chi, omega, mode='full'):
    s, a, P, R, gamma = five_tuple
    data = torch.zeros((s * a, 1), requires_grad=True)
    H = torch.randn([s * a] * k, dtype=torch.float32)
    O = torch.zeros(omega[-1].shape)
    for i in range(k):
        Hi = initialize_H(five_tuple, i + 1)
        H += fill_dims(Hi.get_tensor(), k)
        O += fill_dims(omega[i], k)
    H *= O
    
    if mode == 'cp':
        [core, factors], error = parafac(H, chi, n_iter_max=20, init='svd', 
                                         return_errors=True, mask=O)
        combined_cores, _ = put_cp(factors)
    if mode == 'mps':
        tensors = matrix_product_state(H, chi)
        tensors[0] = tensors[0].squeeze(0)
        tensors[-1] = tensors[-1].squeeze(-1)
        combined_cores, _ = put_mps(tensors)
        error = None
    if mode == 'full':
        combined_cores = [tn.Node(H, backend=backend)]
        error = None
        
    return H, combined_cores, data, error


def build_network(five_tuple, k, chi, omega):
    s, a, P, R, gamma = five_tuple
    H = []
    H_core = []
    data = torch.zeros((s * a, 1), requires_grad=True)
    for i in range(k):
        H.append(initialize_H(five_tuple, i + 1))
        if i > 1:
            [core, factors], error = parafac(H[i].get_tensor(), 
                                             chi, n_iter_max=20, init='svd', 
                                             return_errors=True, mask=omega[i])
            combined_cores, _ = put_cp(factors)
            H_core.append(combined_cores)
        else:
            masked_H = H[i].get_tensor() * omega[i]
            H_core.append([tn.Node(masked_H, backend=backend)])
    return H, H_core, data, error


def explore_env(trajectories, k, R, P, s, a):
    omega = []
    for kk in range(k):
        omega.append(torch.zeros([s * a] * (kk + 1)))
    
    indices = torch.empty(trajectories)
    s_curr = np.random.randint(s)
    for t in range(trajectories):
        a_curr = np.random.randint(a)
        s_next = torch.argmax(P[s_curr, a_curr])
        indices[t] = s_curr * a + a_curr
        s_curr = s_next
    for r in range(trajectories - k):
        for kk in range(k):
            index = tuple(indices[r : r + kk].to(int))
            omega[kk][index] = 1
    return omega