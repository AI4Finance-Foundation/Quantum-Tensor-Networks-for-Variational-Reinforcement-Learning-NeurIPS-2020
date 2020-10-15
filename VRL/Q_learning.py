import numpy as np


def explore(Q, a, eps, ss):
    choice = np.random.uniform()
    if choice < eps:
        return np.random.randint(a)
    else:
        return np.argmax(Q[ss])
    
    
def Q_learning(five_tuple, lr=0.1, epochs=1000, gamma=0.5):
    s, a, P, R, gamma = five_tuple
    Q_history = []
    Q = np.zeros((s, a))
    ss = np.random.randint(s)
    aa = np.random.randint(a)
    
    for _ in range(epochs):    
        Q_history.append(np.sum(Q))
        new_s = np.argmax(P[ss, aa])
        Q[ss, aa] += lr * (R[ss, aa] + gamma * np.max(Q[new_s, :]) - Q[ss, aa])
        ss = new_s.item()
        aa = explore(Q, a, 0.2, ss)
   
    return Q, Q_history