import torch


def initialize_R(s, a):
    R = torch.zeros((s, a)) 
    for i in range(s):
        if i % 5 == 0:
            R[i, 0] = -1
        if (i - 4) % 5 == 0:
            R[i, 1] = -1
        if i in range(5):
            R[i, 2] = -1
        if i in range(s - 5, s):
            R[i, 3] = -1
    R[1, :] = 10
    R[3, :] = 5
    R_vec = R.reshape(s * a)
    return R, R_vec
    
    
def initialize_P(s, a):
    P = torch.zeros((s, a, s)) # s_t, a, s_(t+1)
    for i in range(s):
        if i % 5 == 0:
            P[i, 0, i] = 1
        if (i - 4) % 5 == 0:
            P[i, 1, i] = 1
        if i in range(5):
            P[i, 2, i] = 1
        if i in range(s - 5, s):
            P[i, 3, i] = 1    

        for j in range(s):
            if j == i - 1:
                P[i, 0, j] = 1
            if j == i + 1:
                P[i, 1, j] = 1
            if j == i - 5:
                P[i, 2, j] = 1
            if j == i + 5:
                P[i, 3, j] = 1
        if i == 1:        
            P[i, :, :] = 0
            P[i, :, 21] = 1
        if i == 3:
            P[i, :, :] = 0
            P[i, :, 13] = 1

    P_mat = torch.empty((s, a, s, a))
    for ss in range(s):
        for i in range(a):
            P_mat[:, :, ss, i] = P[:, :, ss]
    P_mat = P_mat.reshape(s * a, s * a)
    return P, P_mat