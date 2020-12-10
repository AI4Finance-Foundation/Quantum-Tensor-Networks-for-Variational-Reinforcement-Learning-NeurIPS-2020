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
        mps.append(tn.Node(tensors[i].detach().clone().squeeze(), backend=backend))
    
    if len(tensors) == 1:
        return mps, []
    
    connected_edges = []
    conn = mps[0][-1] ^ mps[1][0]
    if len(mps) > 2:
        for k in range(1, len(tensors)-1):
            conn = mps[k][-1] ^ mps[k+1][0]
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


def build_network(five_tuple, k, chi, omega):
    s, a, P, R, gamma = five_tuple
    H = []
    H_core = []
    data = softmax_by_state(torch.randn((s * a, 1)), s, a)
    data.requires_grad=True
    
    for i in range(k):
        H.append(initialize_H(five_tuple, i + 1))
        if i >= 1:
            factors = tensor_train(H[i].get_tensor(), chi).factors
            combined_cores, _ = put_mps(factors)
            H_core.append(combined_cores)
        else:
            masked_H = H[i].get_tensor() * omega[i]
            H_core.append([tn.Node(masked_H, backend=backend)])
    return H, H_core, data


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


def softmax_by_state(data, s_size, a_size):
    states = []
    softmax = torch.nn.Softmax(dim=0)
    for s in range(s_size):
        state = data[s * a_size : (s+1) * a_size, :]
        states.append(softmax(state))
    cat = torch.cat(states, dim=0)
    return cat


def tensor_permute(tensor, k):
    dims = torch.tensor(range(len(tensor.shape)))
    perm = torch.roll(dims, -k, 0)
    return tensor.permute(tuple(perm))


def tensor_shift(tensors, k):
    shifted_tensors = []
    order = torch.tensor(range(len(tensors)))
    perm = torch.roll(order, -k, 0)
    for p in perm:
        shifted_tensors.append(tensors[p])
    return shifted_tensors


def tensor_connect(tensors, k):
    shifted = tensor_shift(tensors, k)
    conn = shifted[1]
    for i in range(2, len(shifted)):
        conn = tl.tenalg.contract(conn, len(conn.shape)-1, shifted[i], 0)
    mid_size = 1
    for j in conn.shape[1:-1]:
        mid_size *= j
    conn = conn.reshape(conn.shape[0], mid_size, conn.shape[-1])
    return conn


def tt_als_step(X, Y, R, omega, _lambda, ranks):
    for i in range(Y.shape[1]):
        seen = []
        for j in range(len(omega[i, :])):
            if omega[i, j]:
                seen.append(j)
        if len(seen) == 0:
            Y[:, i, :] = torch.zeros(ranks)
            continue
        temp_X = torch.cat([X[j, :].unsqueeze(0) for j in seen], axis=0)
        temp_R = torch.cat([R[:, j].unsqueeze(-1) for j in seen], axis=1)
        XTX = temp_X.T @ temp_X
        lambdaI = torch.eye(XTX.shape[0]) * _lambda
        y = torch.solve(torch.matmul(temp_R[i, :], temp_X).unsqueeze(-1), XTX + lambdaI).solution.squeeze()
        Y[:, i, :] = y.reshape(ranks)
    return Y


def tt_als(T, cores, omega):
    if len(cores) <= 1:
        return cores
    new_cores = []
    for core in cores:
        new_cores.append(core.get_tensor())
    new_cores[0] = new_cores[0].unsqueeze(0)
    new_cores[-1] = new_cores[-1].unsqueeze(-1)
    for s in range(len(cores)):
        B = tensor_connect(new_cores, s)
        B_mat = tl.unfold(B, 1)
        T_mat = tl.unfold(T.get_tensor(), s)
        omega_mat = tl.unfold(omega, s)
        ranks = (B.shape[-1], B.shape[0])
        new_cores[s] = tt_als_step(B_mat, new_cores[s], T_mat, omega_mat, 0.01, ranks)
    return put_mps(new_cores)[0] 


def tensor_completion(cores, target, omega, lr=0.01, epochs=1):
    if len(cores) <= 1:
        return cores 
    mps = []
    for core in cores:
        mps.append(core.get_tensor().clone().detach())

    mps[0] = mps[0].unsqueeze(0)
    mps[-1] = mps[-1].unsqueeze(-1)
    
    for core in mps:
        core.requires_grad = True
        
    criterion = torch.nn.MSELoss()
    op = optim.SGD(mps, lr=lr, momentum=0.9, weight_decay=5e-4)
    for e in range(epochs):
        op.zero_grad()
        loss = criterion(omega * tt_to_tensor(mps), omega * target.get_tensor())
        loss.backward()
        op.step()
    return put_mps(mps)[0]