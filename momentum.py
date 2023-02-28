import numpy as np
from numpy.linalg import matrix_power
from scipy.sparse import dia_array

# Representation of states:
# Momentum indices range n = -np to np.
# Spin indices are m =  0, 1 (ground, excited)
# idx = 2 * (np + n) + m

class MomentumLattice:

    def __init__(self, k_field, size):
        self.size = size # number of momentum indices
        self.p0 = 0
        self.kvec = k_field

    def momentum_operator(self):
        op_size = 2 * self.size + 1
        momentum = np.zeros((op_size, op_size), dtype="complex")
        for n in range(-self.size, self.size+1):
            idx = n + self.size
            momentum[idx, idx] = self.p0 + float(n) * self.kvec
        return dia_array(momentum)
    
    def kinetic_energy(self, mass):
        return self.momentum_operator().power(2) / (2 * mass)

class ElectricField:

    def __init__(self, kvec, omega):
        self.kvec = kvec
        self.omega = omega