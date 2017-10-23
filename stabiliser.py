import numpy as np
import post_scf

mat_new = post_scf.matrices()
h_tot = mat_new['h_tot']

st_12, st_14, st_23, st_34 = np.zeros(16),  np.zeros(16),  np.zeros(16),  np.zeros(16)
st_12[12] = 1
st_14[9] = 1
st_23[6] = 1
st_34[3] = 1

coef = np.array([0, 1, -1, 1j, -1j])
energies = np.zeros([5, 5, 5])


def min_energy(compare):
    states = []
    d = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                state = st_12 + coef[i]*st_14 + coef[j]*st_23 + coef[k]*st_34
                state = state / np.linalg.norm(state)
                states.append(state)
                d.append(np.linalg.norm(state-compare))
                e = np.dot(np.conj(state), np.matmul(h_tot, state))
                assert e.imag < 1e-8
                energies[i, j, k] = e.real
                # print(energies[i, j, k])

    return energies, states, d
