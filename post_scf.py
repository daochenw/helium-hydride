# post_scf calculates Hamiltonian matrix elements between the two (spatial) solutions to the Fock eigen-equation as
# found in the scf procedure (scf.py).

import numpy as np
import scf

pauli = dict(i=np.eye(2),
             x=np.array([[0, 1], [1, 0]]),
             y=np.array([[0, -1j], [1j, 0]]),
             z=np.array([[1, 0], [0, -1]]),
             l=np.array([[0, 1], [0, 0]]),
             r=np.array([[0, 0], [1, 0]]))


def tsrp(s):
    # tensor product
    return np.kron(np.kron(pauli[s[0]], np.kron(pauli[s[1]], pauli[s[2]])), pauli[s[3]])


def elecop(s):
    s = np.array(s)
    s = s-1
    a = [tsrp('liii'), tsrp('zlii'), tsrp('zzli'), tsrp('zzzl')]
    if s.size == 2:
        out = a[s[0]].transpose()@a[s[1]]
    else:
        assert s.size == 4
        out = a[s[0]].transpose()@a[s[1]].transpose()@a[s[2]]@a[s[3]]
    return out


def matrices():
    c, en, mat = scf.scf()

    h_c = np.zeros([2, 2])

    for i in range(2):
        for j in range(2):
            h_c[i, j] = np.dot(np.kron(c[:, i], c[:, j]), mat['h'].flatten())

    tt = np.zeros([2, 2, 2, 2])

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    tt[i, j, k, l] = np.dot(
                        np.kron(c[:, i], np.kron(c[:, j], np.kron(c[:, k], c[:, l]))), mat['tt'].flatten())

    # define Hamiltonian
    # [i, j] refers to ai†.aj & [i, j, k, l] refers to ai†.aj†.ak.al
    h_tot = h_c[0, 0]*(elecop([1, 1])+elecop([2, 2])) + \
        h_c[0, 1]*(elecop([1, 3])+elecop([2, 4])+elecop([3, 1])+elecop([4, 2])) + \
        h_c[1, 1]*(elecop([3, 3])+elecop([4, 4])) + \
        tt[0, 0, 0, 0]*elecop([1, 2, 2, 1]) + \
        tt[0, 0, 0, 1]*(elecop([1, 2, 4, 1])+elecop([1, 4, 2, 1])+elecop([2, 1, 3, 2])+elecop([2, 3, 1, 2])) + \
        tt[0, 1, 0, 1]*(elecop([1, 2, 4, 3])+elecop([3, 4, 2, 1])+elecop([1, 4, 2, 3])+elecop([3, 2, 4, 1])) + \
        tt[0, 0, 1, 1]*(elecop([2, 3, 3, 2])+elecop([1, 4, 4, 1])) + \
        tt[0, 1, 1, 1]*(elecop([4, 1, 3, 4])+elecop([4, 3, 1, 4])+elecop([3, 4, 2, 3])+elecop([3, 2, 4, 3])) + \
        tt[1, 1, 1, 1]*elecop([3, 4, 4, 3])

    mat_new = dict(h_c=h_c, tt=tt, h_tot=h_tot)

    return mat_new
