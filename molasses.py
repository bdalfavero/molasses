#!/usr/bin/env python3

# molasses.py
# Simulation of optical molasses based on Moelmer

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from numpy.linalg import matrix_power
from scipy.sparse import dia_array
from scipy.constants import pi
from math import ceil
from atom import *
from momentum import *
from sampler import *


def solve(e_field, atom, steps, runs, time):
    dt = time / float(steps)
    # First two moments of momentum.
    p = np.zeros(steps + 1)
    p[0] = np.vdot(atom.psi, atom.lattice.momentum_operator() @ atom.psi).real
    p2 = np.zeros(steps + 1)
    p2[0] += np.vdot(atom.psi, atom.lattice.momentum_operator().power(2) @ atom.psi).real
    p0 = np.zeros(steps + 1)
    p0[0] = atom.lattice.p0
    for j in range(runs):
        # Reset the atom to default initial state.
        atom.reset_psi()
        atom.lattice.p0 = 0.0
        for i in range(1, steps+1):
            # Step the wave function forward, and store the state to the array.
            atom.step_forward(e_field, dt)
            p[i] += np.vdot(atom.psi, atom.lattice.momentum_operator() @ atom.psi).real
            p2[i] += np.vdot(atom.psi, atom.lattice.momentum_operator().power(2) @ atom.psi).real
            p0[i] += atom.lattice.p0
    p = p / float(runs)
    p2 = p2 / float(runs)
    p0 = p0 / float(runs)
    # Plot momentum.
    t = np.linspace(0.0, time, steps + 1)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, p / atom.lattice.kvec, '-')
    ax[0].set_ylabel("p")
    ax[1].plot(t, p2 / (atom.lattice.kvec ** 2), '-')
    ax[1].set_ylabel("p2")
    fig2, ax2 = plt.subplots()
    ax2.plot(t, p0)
    ax2.set_ylabel("p0")
    plt.show()

def main():
    k = 2.0 * pi * 1.28e4 * 100.0 / sqrt(3.8e7)
    lattice = MomentumLattice(k, 20)
    gamma = 1.0
    omega = -0.5
    mass = 200.0 * k ** 2 / gamma
    table_size = 50
    atom = Atom(omega, mass, gamma, lattice, table_size)
    e_field = ElectricField(k, 0.5)
    solve(e_field, atom, int(1e4), 100, 100.0)

if __name__ == "__main__":
    main()
