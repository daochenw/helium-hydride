# ve implements the variational eigen-solver method by preparing UCC ansatz states and evaluating its energy through
# simulated measurements (which introduces noise).

from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import importlib
import numpy as np
import matplotlib.pyplot as plt
import post_scf
import classical
importlib.reload(classical)
importlib.reload(post_scf)

pauli = post_scf.pauli
tsrp = post_scf.tsrp
mat_new, h_tot, h_c, tt, operations, coef = None, None, None, None, None, None


def mat(r):
    global mat_new, h_tot, h_c, tt, operations, coef
    # 1.4632
    mat_new = post_scf.matrices(r=r)
    h_tot = mat_new['h_tot']
    h_c = mat_new['h_c']
    tt = mat_new['tt']

    operations_ = ['ziii', 'izii', 'iizi', 'iiiz',
                   'zzii', 'iizz', 'izzi', 'ziiz',
                   'xzxi', 'yzyi', 'ixzx', 'iyzy',
                   'zxzx', 'zyzy', 'xzxz', 'yzyz',
                   'xixi', 'yiyi', 'ixix', 'iyiy',
                   'xxyy', 'yyxx', 'yxxy', 'xyyx']

    a1 = -0.5*h_c[0, 0]-0.25*(tt[0, 0, 0, 0]+tt[0, 0, 1, 1])
    a2 = -0.5*h_c[1, 1]-0.25*(tt[1, 1, 1, 1]+tt[0, 0, 1, 1])
    c1 = 0.5*h_c[0, 1]+0.25*(tt[0, 0, 0, 1]+tt[1, 1, 1, 0])
    d1 = -0.25*tt[0, 0, 0, 1]
    d2 = -0.25*tt[1, 1, 1, 0]
    f1 = -0.25*tt[0, 1, 0, 1]

    coef = [a1, a1, a2, a2,
            0.25*tt[0, 0, 0, 0], 0.25*tt[1, 1, 1, 1], 0.25*tt[0, 0, 1, 1], 0.25*tt[0, 0, 1, 1],
            c1, c1, c1, c1,
            d1, d1, d2, d2,
            d1, d1, d2, d2,
            f1, f1, -f1, -f1]

    operations = [tsrp(operations_[i]) for i in range(24)]


def plot_disa(r=100):
    # plots the g.s. energy for different bond length values
    r_range = np.linspace(0.5, 5, r)
    store = np.zeros([r_range.size, 3])
    n = 0
    for i in r_range:
        mat(i)
        print('The bond length is: ', i)
        store[n, 0] = i
        t0 = np.zeros(3)
        za, zb = 2, 1
        # store[n, 1] = var_e(t0, mode=0, n_repeat=10).fun + za * zb / i
        # store[n, 2] = var_e(t0, mode=1, n_repeat=100).fun + za * zb / i
        store[n, 1] = energy(t0, mode=0) + za * zb / i         # the HF energy is energy(t0)
        # store[n, 2] = energy(t0, mode=1, n_repeat=1000) + za * zb / i
        store[n, 2] = var_e(t0, mode=0).fun + za * zb / i

        n = n + 1
    plt.plot(store[:, 0], store[:, 1], label='Hartree-Fock')
    plt.plot(store[:, 0], store[:, 2], label='UCC variational')
    plt.xlabel('R (a.u.)')
    plt.ylabel('E (a.u.)')
    plt.title('Helium hydride ground state energy (E) as a function of bond length (R)')
    plt.grid(ls='--')
    plt.legend()
    return store


def etsrp(t, s):
    return np.cos(t)*np.eye(16)+1j*np.sin(t)*tsrp(s)


def prepare(t):
    state = np.zeros(16, dtype=complex)
    if t.size == 3:
        # UCC evolution (T=T1+T2, one Trotter step, HF reference state - so ops annihilating HF are dropped,
        # consider only z spin sz = 0 states as this is a property of the true G.S. - so ops involving e.g. a3†a2
        # are dropped)
        print('t is', t)

        # exp(it(a3†a1 - h.c.))
        u1 = etsrp(t[0], 'xzyi') @ etsrp(-t[0], 'yzxi')
        # exp(it(a4†a2 - h.c.))
        u2 = etsrp(t[1], 'ixzy') @ etsrp(-t[1], 'iyzx')
        # exp(it(a4†a3†a2a1 - h.c.))
        u3 = etsrp(t[2], 'xxxy') @ etsrp(-t[2], 'yyyx') @ \
            etsrp(t[2], 'xxyx') @ etsrp(-t[2], 'yyxy') @ \
            etsrp(t[2], 'xyyy') @ etsrp(-t[2], 'yxxx') @ \
            etsrp(t[2], 'yxyy') @ etsrp(-t[2], 'xyxx')

        # HF reference state
        state[12] = 1
        # UCC ansatz
        state = u3 @ u2 @ u1 @ state

    else:
        assert t.size == 4
        state[[3, 6, 9, 12]] = t
        state = state / np.linalg.norm(state)

    assert abs(np.linalg.norm(state) - 1) < 10e-8
    return state


def energy(t, mode=1, n_repeat=10):

    state = prepare(t)
    true = np.dot(np.conj(state), np.matmul(h_tot, state))

    # observables have real expectations (Hermitian matrices have real e-vals)

    if mode == 1:
        true = q_energy(state, n_repeat)
        # print('true energy is: ', true0)
        # print('mle energy is: ', true)
        # print('error is: ', np.abs((true-true0)))

    assert abs(true.imag / true.real) < 10e-8
    true = true.real

    # print('Current energy is: ', true)
    return true


def q_energy(state, n_repeat=10):

    mle_op = np.zeros(24)

    def var_op(st, op, n_):

        # quantum computer simulator
        true = np.dot(np.conj(st), np.matmul(op, st))
        assert abs(true.imag) < 10e-8
        assert -1-1e-8 <= true.real <= 1+1e-8

        p1 = min(max((1 + true.real)/2, 0), 1)
        smpl_ = np.random.binomial(n_, p1)

        return smpl_

    for i in range(24):
        smpl = var_op(state, operations[i], n_repeat)
        mle_op[i] = 2*smpl/n_repeat-1

    bias = h_c[0, 0] + h_c[1, 1] + 0.5*tt[0, 0, 1, 1] + 0.25*(tt[0, 0, 0, 0]+tt[1, 1, 1, 1])
    # bias = 0
    mle_energy = np.dot(coef, mle_op) + bias

    return mle_energy


def var_e(t0=np.zeros(3), mode=0, n_repeat=10):
    # t0=np.zeros(3) = start from the HF ground state
    # could also use  t0 = np.ones(4) for another start state
    print('\nNow we optimise the energy over {FCI - (sz=0)} coefficients \n')

    def energy_(t): return energy(t, mode=mode, n_repeat=n_repeat)

    res = minimize(energy_, t0, method='nelder-mead', options={'xtol': 1e-5, 'disp': True, 'maxiter': 1000})

    return res


def plot_2d():
    cls = classical.mps()
    m = cls.shape[0]
    print('Im here')
    x = np.arange(-2, 2, 0.05)
    z = np.zeros(x.size)
    x_list = []
    z_list = []

    def energy_1(x_):
        state = np.array([prepare(np.array([x_, 0, 0]))])
        state_rep = np.repeat(state, m, axis=0)
        e = energy(np.array([x_, 0, 0]))
        # _, _, d = stabiliser.min_energy(state)
        d = np.linalg.norm(cls[:, :, 0] - state_rep, axis=1)
        if min(d) < 0.1:
            x_list.append(x_)
            z_list.append(e)
        return e

    for p in range(x.size):
            z[p] = energy_1(x[p])
    plt.scatter(x_list, z_list, c='r')
    plt.plot(x, z)
    plt.show()


def plot_3d(mode=0):

    if mode == 1:
        cls = classical.mps()
        m = cls.shape[0]
    x = np.arange(-2, 2, 0.1)
    y = np.arange(-2, 2, 0.1)
    z = np.zeros([x.size, y.size])
    cls_list = []

    def energy_2(x_, y_):
        e = energy(np.array([x_, y_, 0]))
        if mode == 1:
            state = np.array([prepare(np.array([x_, y_, 0]))])
            state_rep = np.repeat(state, m, axis=0)
            d = np.linalg.norm(cls[:, :, 0] - state_rep, axis=1)
            if min(d) < 0.1:
                cls_list.append((x_, y_, e))
        return e

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for p in range(x.size):
        for q in range(y.size):
            z[p, q] = energy_2(x[p], y[q])

    if mode == 1:
        for s in cls_list:
            ax.scatter(s[0], s[1], s[2], zdir='z', c='r')

    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, z,
                           linewidth=0, antialiased=False)

    plt.show()