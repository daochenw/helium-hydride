# scf calculates solutions to the (spatial) Fock eigen-equation.

import numpy as np
import pre_scf


def scf(zeta1=2.0925, zeta2=1.24, r=1.4632, za=2, zb=1, eps=1e-4, max_iter=25):

    # retrieve the Fock operator matrix elements 'h' (core), 'tt' (two-electron) in the STO basis of 2 spatial orbitals,
    # and also the basis' overlap matrix 's' and the canonical diagonalisation matrix 'x' of 's'
    ref = pre_scf.integral(zeta1, zeta2, r, za, zb)
    mat = pre_scf.matrices(ref)

    def formg(p_):
        # this calculates G, the two electron part of the  is implementing eqn (3.154) S.O.
        g_ = np.zeros([2, 2])
        for i in range(2):
            for j in range(2):
                g_[i, j] = \
                    sum([p_[k, l]*(mat['tt'][i, j, k, l] - 0.5*mat['tt'][i, l, k, j]) for k in range(2) for l in range(2)])
        return g_

    # initial guess of density matrix p
    p = np.zeros([2, 2])
    # h is the core Hamiltonian matrix (electron kinetic and nuclear attraction terms in the Fock matrix) which
    # doesn't change - since the corresponding operator itself doesn't depend on the current eigen-solution orbitals
    # (<-> current density matrix p)
    h = mat['h']

    # start the SCF iterations
    n, delta = 0, 1
    while all([delta >= eps, n <= max_iter]):
        print('p is: ', p)
        n = n+1
        # g is the G matrix (electron-electron Coulomb repulsion terms)
        g = formg(p)
        print('g is: ', g)
        # F is Fock matrix - cf. eqn (3.154) S.O.
        f = h + g
        print('f is: ', f)
        # calculate electronic energy - cf. eqn (3.184) S.O. - this formula is only valid for the H.F. ground state;
        # note the Fock matrix f appearing is obtained via the same 'p' in the formula
        en = sum([0.5*p[i, j]*(h[i, j]+f[i, j]) for i in range(2) for j in range(2)])
        print('electronic energy is: ', en)
        # f_trans is x†.f.x - the transformed Fock matrix: f_trans*x_inv*c = x_inv*c*eps,
        # write x_inv*c = c_ -> usual eigen-equation
        f_trans = mat['x'].transpose() @ f @ mat['x']
        print('f_trans is: ', f_trans)
        ev, c_trans = np.linalg.eig(f_trans)
        print('c_trans is: ', c_trans)
        print('ev is: ', ev)
        # c is the coefficient matrix of the Fock eigenstates in terms of the original minimal STO
        c = mat['x'] @ c_trans
        print('c is: ', c)
        # store the old density matrix
        p_old = p
        # form the new density matrix - cf. eqn (3.145) S.O., note the N is the # of electrons and so only one summand
        # (as there is only one spatial orbital involved in the G.S. - giving two spin orbitals, up & down). Possibly
        # generalise to p = 2 * np.outer(c[:, 1], c[:, 1].conjugate(), in case of complex coefficients)
        p = 2 * np.outer(c[:, 0], c[:, 0])
        # calculate distance between p_old and p_new
        delta = np.sqrt(np.sum((p-p_old)**2)/4)

        print('In iteration: ', n, ', the density matrix delta is: ', delta, '\n')

        if delta < eps:
            print('The SCF has converged.')

        if n > max_iter:
            print('The SCF has not converged.')

    # add in the nuclear energy:
    # en = en + za * zb / r

    print('The numerical HF G.S. electronic energy is: ', en)

    return c, en, mat
