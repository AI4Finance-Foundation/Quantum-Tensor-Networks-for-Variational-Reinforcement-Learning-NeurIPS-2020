# Varitional Reinfocement Learning (VRL) using Quantum Tensor Networks

This repository provides the implementation for VRL algorithm and applies it to the classic GridWorld puzzle.

## Usage

First, import the VRL module by
```
from VRL import *
```
Then, initialize the parameters.  An example initialization would be
```
s, a = 25, 4
k, gamma, chi = 3, 0.5, 50
R, R_vec = initialize_R(s, a)
P, P_mat = initialize_P(s, a)
five_tuple = s, a, P_mat, R_vec, gamma
```

Now, explore the environment and train the model by running
```
omega = explore_env(1000, k, R, P, s, a)
H, cores, data, completion_error = build_network(five_tuple, k, chi, omega)
spin, energy_history = VRL_train(five_tuple, k, data, cores, lr=0.1, epochs=1000)
```
One can retrieve the trained policy from the variable `spin`; the optimization processes for tensor completion and energy minimization are tracked by `completion_error` and `energy_history` respectively.
