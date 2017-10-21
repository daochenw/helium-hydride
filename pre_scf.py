import numpy as np
from scipy.special import erf

# STO-3G (approx. to STO - zeta = 1 - with 3 contracted Gaussian functions (CGF)); to be clear, the CGFs are introduced
# for computational convenience when it comes to evaluating integrals  rather than any deeper purpose. The STO itself
# consists of two spatial orbitals (which is why it is "minimal" in the two electron case). STOs has e^(-kr) about the
# scaling as r -> inf which agrees with experiment; STOs are parametrised by the 'k' and the centre from which r is
# taken. Further details are provided in Chapter 3.5 of Szabo & Ostlund (S.O. in below).

# defining a k =1 STO in terms of CGFs (others can be reached by appropriate scaling via its decay rate k),
# see S.O. p157.
coef = np.array([0.444635, 0.535328, 0.154329])
expon = np.array([0.109818, 0.405771, 2.22766])


def f0(t):
    if abs(t) < 10**(-6):
        # check this
        return 1 - t/3
    f = 0.5*np.sqrt(np.pi/t)*erf(np.sqrt(t))
    return f


def ke(a, b, rab2):
    return a*b/(a+b)*(3-2*a*b/(a+b)*rab2)*(np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*rab2)


def nuc(a, b, rab2, rcp2, zc):
    return -2*np.pi/(a+b)*zc*np.exp(-a*b/(a+b)*rab2)*f0((a+b)*rcp2)


def twoe(a, b, c, d, rab2, rcd2, rpq2):
    return 2*np.pi**2.5/((a+b)*(c+d)*np.sqrt(a+b+c+d)) * \
          np.exp(-a*b/(a+b)*rab2-c*d/(c+d)*rcd2) * \
          f0((a+b)*(c+d)/(a+b+c+d)*rpq2)


def ovlp(a, b, rab2):
    return (np.pi/(a+b))**1.5*np.exp(-a*b*rab2/(a+b))


def integral(zeta1, zeta2, r, za, zb):
    a1 = expon*zeta1**2
    d1 = coef*((2*a1/np.pi)**0.75)
    a2 = expon*zeta2**2
    d2 = coef*((2*a2/np.pi)**0.75)
    r2 = r*r

    s12 = 0
    ke11, ke12, ke22 = 0, 0, 0
    nuca11, nuca12, nuca22 = 0, 0, 0
    nucb11, nucb12, nucb22 = 0, 0, 0
    v1111, v2111, v2121, v2211, v2221, v2222 = 0, 0, 0, 0, 0, 0

    for i in range(3):
        for j in range(3):
            rap = a2[j]*r/(a1[i]+a2[j])
            rap2 = rap**2
            rbp2 = (r-rap)**2
            s12 = s12 + ovlp(a1[i], a2[j], r2)*d1[i]*d2[j]
            ke11 = ke11 + ke(a1[i], a1[j], 0)*d1[i]*d1[j]
            ke12 = ke12 + ke(a1[i], a2[j], r2)*d1[i]*d2[j]
            ke22 = ke22 + ke(a2[i], a2[j], 0)*d2[i]*d2[j]
            nuca11 = nuca11 + nuc(a1[i], a1[j], 0, 0, za)*d1[i]*d1[j]
            nuca12 = nuca12 + nuc(a1[i], a2[j], r2, rap2, za)*d1[i]*d2[j]
            nuca22 = nuca22 + nuc(a2[i], a2[j], 0, r2, za)*d2[i]*d2[j]
            nucb11 = nucb11 + nuc(a1[i], a1[j], 0, r2, zb)*d1[i]*d1[j]
            nucb12 = nucb12 + nuc(a1[i], a2[j], r2, rbp2, zb)*d1[i]*d2[j]
            nucb22 = nucb22 + nuc(a2[i], a2[j], 0, 0, zb)*d2[i]*d2[j]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):

                    rap = a2[i]*r/(a2[i]+a1[j])
                    rbp = r-rap
                    raq = a2[k]*r/(a2[k]+a1[l])
                    rbq = r-raq
                    rpq = rap-raq
                    rap2 = rap*rap
                    rbp2 = rbp*rbp
                    raq2 = raq*raq
                    rbq2 = rbq*rbq
                    rpq2 = rpq*rpq

                    v1111 = v1111+twoe(a1[i], a1[j], a1[k], a1[l], 0, 0, 0) * \
                        d1[i]*d1[j]*d1[k]*d1[l]

                    v2111 = v2111 + twoe(a2[i], a1[j], a1[k], a1[l], r2, 0, rap2) * \
                        d2[i]*d1[j]*d1[k]*d1[l]

                    v2121 = v2121 + twoe(a2[i], a1[j], a2[k], a1[l], r2, r2, rpq2) * \
                        d2[i]*d1[j]*d2[k]*d1[l]

                    v2211 = v2211 + twoe(a2[i], a2[j], a1[k], a1[l], 0, 0, r2) * \
                        d2[i]*d2[j]*d1[k]*d1[l]

                    v2221 = v2221 + twoe(a2[i], a2[j], a2[k], a1[l], 0, r2, rbq2) * \
                        d2[i]*d2[j]*d2[k]*d1[l]

                    v2222 = v2222 + twoe(a2[i], a2[j], a2[k], a2[l], 0, 0, 0) * \
                        d2[i]*d2[j]*d2[k]*d2[l]

    out = dict(s_12=s12, ke_11=ke11, ke_12=ke12, ke_22=ke22,
               nuca_11=nuca11, nuca_12=nuca12, nuca_22=nuca22,
               nucb_11=nucb11, nucb_12=nucb12, nucb_22= nucb22,
               v_1111=v1111, v_2111=v2111, v_2121=v2121, v_2211=v2211, v_2221=v2221, v_2222=v2222)

    return out


def sym(a):
    return a + a.T - np.diag(a.diagonal())


def matrices(ref):

    # h is core hamiltonian
    h = np.array([[ref['ke_11']+ref['nuca_11']+ref['nucb_11'],
                   ref['ke_12']+ref['nuca_12']+ref['nucb_12']],
                  [0, ref['ke_22']+ref['nuca_22']+ref['nucb_22']]])
    h = sym(h)

    # s is overlap matrix
    s = np.array([[1, ref['s_12']], [0, 1]])
    s = sym(s)

    # x is s.t. x†sx = 1, obtained here via canonical diagonalisation († is Hermitian conjugate or transpose here as
    # everything is real)
    x = np.array([[1/np.sqrt(2+2*s[0, 1]), 1/np.sqrt(2-2*s[0, 1])], [0, 0]])
    x[1, 0] = x[0, 0]
    x[1, 1] = -x[0, 1]

    # tt is matrix of two-electron integrals
    tt = np.zeros([2, 2, 2, 2])
    tt[0, 0, 0, 0] = ref['v_1111']
    tt[1, 0, 0, 0], tt[0, 1, 0, 0], tt[0, 0, 1, 0], tt[0, 0, 0, 1] = np.ones(4)*ref['v_2111']
    tt[1, 0, 1, 0], tt[0, 1, 1, 0], tt[1, 0, 0, 1], tt[0, 1, 0, 1] = np.ones(4)*ref['v_2121']
    tt[1, 1, 0, 0], tt[0, 0, 1, 1] = np.ones(2)*ref['v_2211']
    tt[1, 1, 1, 0], tt[1, 1, 0, 1], tt[1, 0, 1, 1], tt[0, 1, 1, 1] = np.ones(4)*ref['v_2221']
    tt[1, 1, 1, 1] = ref['v_2222']

    mat = dict(h=h, tt=tt, s=s, x=x)

    return mat
