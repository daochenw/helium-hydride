# 'classical' generates various types of classically preparable ansatz states, currently the stabiliser states and
# matrix product states (mps) of 4 qubits are implemented

import numpy as np
import post_scf

mat_new = post_scf.matrices()
h_tot = mat_new['h_tot']

st_12, st_14, st_23, st_34 = np.zeros(16),  np.zeros(16),  np.zeros(16),  np.zeros(16)
st_12[12] = 1
st_14[9] = 1
st_23[6] = 1
st_34[3] = 1


def stabiliser():
    coef = np.array([0, 1, -1, 1j, -1j])
    store = np.zeros([125, 16, 2])
    counter = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                state = st_12 + coef[i]*st_14 + coef[j]*st_23 + coef[k]*st_34
                state = state / np.linalg.norm(state)
                e = np.dot(np.conj(state), np.matmul(h_tot, state))
                assert e.imag < 1e-8

                store[counter, :, 0] = state
                store[counter, :, 1] = e
                counter = counter + 1
    return store


def mps():
    # mps = matrix product states
    k1, l1, k2, l2 = 50, 50, 50, 50
    store = np.zeros([k1*l1*k2*l2, 16, 2], dtype=complex)
    counter = 0
    if True:
        for t1 in np.linspace(0, 4*np.pi, k1):
            for p1 in np.linspace(0, 2*np.pi, l1):
                for t2 in np.linspace(0, 4*np.pi, k2):
                    for p2 in np.linspace(0, 4*np.pi, l2):
                        state = np.cos(t1/2)*np.cos(t2/2)*st_12 + \
                            np.exp(1j*p2)*np.cos(t1 / 2) * np.sin(t2 / 2) * st_14 + \
                            np.exp(1j*p1)*np.sin(t1 / 2) * np.cos(t2 / 2) * st_23 + \
                            np.exp(1j*(p1+p2))*np.sin(t1 / 2) * np.sin(t2 / 2) * st_34

                        state = state / np.linalg.norm(state)
                        e = np.dot(np.conj(state), np.matmul(h_tot, state))
                        assert e.imag < 1e-8

                        store[counter, :, 0] = state
                        store[counter, :, 1] = e.real
                        counter = counter + 1

    return store
    # return energies, states, d


