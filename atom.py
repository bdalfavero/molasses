import numpy as np
#from scipy.sparse import dia_array
#from scipy.sparse.linalg import expm
from scipy.linalg import expm
from sampler import MomentumSampler
from scipy.sparse import dia_array
from math import sqrt

class Atom:

    def __init__(self, omega0, mass, gamma, momentum_lattice, table_size):
        # Set properties
        self.omega = omega0
        self.mass = mass
        self.gamma = gamma
        self.lattice = momentum_lattice
        self.dimension = int(2 * self.lattice.size + 1) # Operator dimension
        self.sampler = MomentumSampler(table_size, self.lattice.kvec)
        # Set initial wave function
        self.psi = np.zeros((self.dimension), dtype="complex")
        self.psi[self.lattice.size] = 1.0
    
    def reset_psi(self):
        self.psi = np.zeros((self.dimension), dtype="complex")
        self.psi[self.lattice.size] = 1.0
    
    def rabi_term(self, e_field):
        # The momentum lattice is constructed in such a way that
        # the Rabi Hamiltonian only has ones on the off-diagonal.
        hr = np.zeros((self.dimension, self.dimension), dtype="complex")
        for j in range(self.dimension):
            if (j != 0):
                hr[j - 1, j] = e_field.omega / 2.
            if (j != self.dimension - 1):
                hr[j + 1, j] = e_field.omega / 2.
        return dia_array(hr)

    def bare_hamiltonian(self):
        # The excited state energies are -delta,
        # where delta is the detuning.
        delta = self.omega
        h_bare = np.zeros((self.dimension, self.dimension), dtype="complex")
        for i in range(self.dimension):
            if (self.lattice.size % 2 == 0):
                if (i % 2 == 0):
                    h_bare[i, i] = 0.0
                else:
                    h_bare[i, i] = -delta
            else:
                if (i % 2 == 0):
                    h_bare[i, i] = -delta
                else:
                    h_bare[i, i] = 0.0
        return dia_array(h_bare)

    def hamiltonian(self, e_field):
        # Get kinetic term.
        kinetic = self.lattice.kinetic_energy(self.mass)
        # Get the Rabi term.
        rabi = self.rabi_term(e_field)
        # Get the bare Hamiltonian.
        bare = self.bare_hamiltonian()
        return kinetic + rabi + bare
    
    def excited_projector(self):
        proj_exc = np.zeros((self.dimension, self.dimension), dtype="complex")
        for i in range(self.dimension):
            if (self.lattice.size % 2 == 0):
                if (i % 2 == 0):
                    proj_exc[i, i] = 0.0
                else:
                    proj_exc[i, i] = 1.0
            else:
                if (i % 2 == 0):
                    proj_exc[i, i] = 1.0
                else:
                    proj_exc[i, i] = 0.0
        return dia_array(proj_exc)
    
    def effective_hamiltonian(self, e_field):
        return self.hamiltonian(e_field) - complex(0., 1.) * self.gamma / 2 * self.excited_projector()
    
    def jump_probability(self, dt):
        # Probability that the atom will spontaneously emit
        # from its current state.
        proj_exc = self.excited_projector()
        return self.gamma * dt * np.vdot(self.psi, proj_exc @ self.psi).real
    
    def step_forward(self, e_field, dt):
        # First, evolve under the effective Hamiltonian.
        self.psi = self.psi - 1j * dt * self.effective_hamiltonian(e_field) @ self.psi
        # Check if a jump will occur.
        p_jump = self.jump_probability(dt)
        r = np.random.rand()
        if (r <= p_jump):
            # Shift the amplitudes from excited states to ground states.
            ell = jump_operator(self.lattice)
            self.psi = ell @ self.psi
            # Draw k_prime following Moelmer pg. 533.
            r = np.random.rand()
            if (r <= 1.0 / 5):
                kp = -self.lattice.kvec
            elif (r <= 4.0 / 5):
                kp = 0.0
            else:
                kp = self.lattice.kvec
            # Shift p0 of the lattice by the required amount.
            self.lattice.p0 = self.lattice.p0 + self.lattice.kvec - kp
        # Renormalize the state.
        self.psi = self.psi / sqrt(np.vdot(self.psi, self.psi).real)


def jump_operator(lattice):
    ell = np.zeros((2 * lattice.size + 1, 2 * lattice.size + 1), dtype="complex")
    if (lattice.size % 2 == 0):
        p = 1
    else:
        p = 0
    for i in range(2 * lattice.size):
        ell[i, i+1] = float(p)
        if (p == 0):
            p = 1
        else:
            p = 0
    return dia_array(ell)   