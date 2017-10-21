import numpy as np
import cmath
import post_scf
from scipy.optimize import minimize

pauli = post_scf.pauli
mat_new = post_scf.matrices()
tsrp = post_scf.tsrp
h_tot = mat_new['h_tot']

i = 1


def etsrp(t, s):
    return np.cos(t)*np.eye(16)+1j*np.sin(t)*tsrp(s)


def energy(l, epsilon=10e-6):

    state = prepare(l)
    true = np.dot(np.conj(state), np.matmul(h_tot, state))
    # observables have real expectations (Hermitian matrices have real e-vals)
    assert abs(true.imag/true.real) < 10e-8
    true = true.real
    return true


def prepare(t):
    state = np.zeros(16, dtype=complex)
    if t.size == 3:
        # UCC evolution (T=T1+T2, one Trotter step, HF reference state - so ops annihilating HF are dropped, consider
        # only z spin sz = 0 states as this is a property of the true G.S. - so ops involving e.g. a3†a2 are dropped)
        # exp(it(a3†a1 - h.c.))
        u1 = etsrp(t[0], 'xzyi')@etsrp(-t[0], 'yzxi')
        # exp(it(a4†a2 - h.c.))
        u2 = etsrp(t[1], 'ixzy')@etsrp(-t[1], 'iyzx')
        # exp(it(a4†a3†a2a1 - h.c.))
        u3 = etsrp(t[2], 'xxxy')@etsrp(-t[2], 'yyyx')@ \
            etsrp(t[2], 'xxyx')@etsrp(-t[2], 'yyxy')@ \
            etsrp(t[2], 'xyyy')@etsrp(-t[2], 'yxxx') @ \
            etsrp(t[2], 'yxyy')@etsrp(-t[2], 'xyxx')

        # HF reference state
        state[12] = 1
        # UCC ansatz
        state = u3 @ u2 @ u1 @ state

    else:
        assert t.size == 4
        state[[3, 6, 9, 12]] = t
        state = state/np.linalg.norm(state)

    assert abs(np.linalg.norm(state) - 1) < 10e-8
    return state


def var_en(state, operation, n):

    # quantum computer simulator
    r, a = np.absolute(true), cmath.phase(true)
    p_cos = 0.5 * (1 - r*np.cos(a))
    f1 = np.random.binomial(n, p_cos)/n
    rcosa = 1-2*f1

    p_sin = 0.5 * (1 - r*np.sin(a))
    f2 = np.random.binomial(n, p_sin)/n
    rsina = 1-2*f2

    r = np.sqrt(rcosa**2+rsina**2)
    a = np.arctan2(rsina, rcosa)
    out = r*np.exp(1j*a)

    return np.array([out.real, out.imag])


def optimise():

    print('\nNow we optimise the energy over {FCI - (sz=0)} coefficients \n')
    # start from the HF ground state
    # t0 = np.zeros(3)
    t0 = np.ones(4)
    res = minimize(energy, t0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True, 'maxiter': 1000})

    return res

e = optimise()
