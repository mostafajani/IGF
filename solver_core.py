import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def reference_solution(alpha, beta, n_eval=2001, tol=1e-12):
    """Shooting on the apex value s = u(0) with u'(0)=0, targeting u(1)=0."""

    def rhs(x, y):
        u, v = y
        dudx = v
        dvdx = alpha * u - beta / np.sqrt(1.0 + v ** 2)
        return [dudx, dvdx]
    x_eval = np.linspace(0.0, 1.0, n_eval)

    def shoot(s):
        sol = solve_ivp(rhs, (0, 1), [s, 0.0], method='RK45', t_eval=x_eval, rtol=tol, atol=tol)
        return sol.y[0, -1]
    lo, hi = (1e-08, 1.0)
    while shoot(hi) < 0.0:
        hi *= 2.0
        if hi > 10000.0:
            raise RuntimeError('shooting bracket failed')
    s_star = brentq(shoot, lo, hi, xtol=1e-14, rtol=1e-14)
    sol = solve_ivp(rhs, (0, 1), [s_star, 0.0], method='RK45', t_eval=x_eval, rtol=tol, atol=tol)
    return (x_eval, sol.y[0], sol.y[1], s_star)

def kernel_factory(name, **kw):
    """Return (K, dK, ddK):  K(x,y), d/dx K, d^2/dx^2 K  with x,y scalars/arrays."""
    if name == 'rbf':
        g = kw['gamma']

        def K(x, y):
            d = x - y
            return np.exp(-g * d * d)

        def dK(x, y):
            d = x - y
            return -2.0 * g * d * np.exp(-g * d * d)

        def ddK(x, y):
            d = x - y
            return (-2.0 * g + 4.0 * g * g * d * d) * np.exp(-g * d * d)
    elif name == 'poly':
        deg, c0 = (kw['degree'], kw.get('coef0', 1.0))

        def K(x, y):
            return (x * y + c0) ** deg

        def dK(x, y):
            return deg * y * (x * y + c0) ** (deg - 1)

        def ddK(x, y):
            return deg * (deg - 1) * y * y * (x * y + c0) ** (deg - 2)
    elif name == 'matern':
        ell = kw['ell']
        s5 = np.sqrt(5.0)

        def K(x, y):
            r = np.abs(x - y) / ell
            return (1 + s5 * r + 5 * r ** 2 / 3) * np.exp(-s5 * r)

        def dK(x, y):
            r = np.abs(x - y) / ell
            sgn = np.sign(x - y)
            return -(5.0 / (3.0 * ell ** 2)) * (x - y) * (1 + s5 * r) * np.exp(-s5 * r)

        def ddK(x, y):
            r = np.abs(x - y) / ell
            return -(5.0 / (3.0 * ell ** 2)) * np.exp(-s5 * r) * (1.0 + s5 * r - 5.0 * r ** 2)
    else:
        raise ValueError(name)
    return (K, dK, ddK)

def lssvr_collocation(alpha, beta, N=24, kernel='rbf', w_bc=1000.0, max_newton=50, tol=1e-11, verbose=False, **kkw):
    j = np.arange(N)
    xc = 0.5 * (1.0 - np.cos(np.pi * j / (N - 1)))
    K, dK, ddK = kernel_factory(kernel, **kkw)

    def field(c, b, x):
        """u(x), u'(x), u''(x) for coefficient vector c and bias b."""
        u = np.array([np.sum(c * K(xi, xc)) + b for xi in x])
        du = np.array([np.sum(c * dK(xi, xc)) for xi in x])
        ddu = np.array([np.sum(c * ddK(xi, xc)) for xi in x])
        return (u, du, ddu)

    def residual_and_jacobian(c, b):
        u, du, ddu = field(c, b, xc)
        g = (1.0 + du ** 2) ** (-0.5)
        gp = -du * (1.0 + du ** 2) ** (-1.5)
        F = ddu - alpha * u + beta * g
        KK = K(xc[:, None], xc[None, :])
        dKK = dK(xc[:, None], xc[None, :])
        ddKK = ddK(xc[:, None], xc[None, :])
        Jc = ddKK - alpha * KK + beta * (gp[:, None] * dKK)
        Jb = -alpha * np.ones(N)
        bc1 = w_bc * dK(np.array([0.0]), xc)
        bc2 = w_bc * np.append(K(np.array([1.0]), xc), 1.0)
        J = np.vstack([np.hstack([Jc, Jb[:, None]]), np.append(bc1, 0.0)[None, :], bc2[None, :]])
        F = np.concatenate([F, [w_bc * np.sum(c * dK(0.0, xc))], [w_bc * (np.sum(c * K(1.0, xc)) + b)]])
        return (F, J)
    c, b = (np.zeros(N), 0.0)
    hist = []
    for it in range(max_newton):
        F, J = residual_and_jacobian(c, b)
        nrm = np.linalg.norm(F)
        hist.append(nrm)
        if verbose:
            print(f'  Newton it {it:2d}  ||F|| = {nrm:.3e}')
        if nrm < tol:
            break
        step, *_ = np.linalg.lstsq(J, -F, rcond=None)
        lam = 1.0
        accepted = False
        theta = np.concatenate([c, [b]])
        while lam > 0.0001:
            tn = theta + lam * step
            Fn, _ = residual_and_jacobian(tn[:-1], tn[-1])
            if np.linalg.norm(Fn) < (1 - 0.0001 * lam) * nrm:
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
        c, b = (tn[:-1], tn[-1])
    return (xc, c, b, hist, K, dK, ddK)

def lssvr_eval(x, xc, c, b, K, dK, ddK):
    u = np.array([np.sum(c * K(xi, xc)) + b for xi in x])
    du = np.array([np.sum(c * dK(xi, xc)) for xi in x])
    return (u, du)

def norms(x, u, du, uref, duref):
    h = 1.0 / (len(x) - 1)
    e = u - uref
    de = du - duref
    L2 = np.sqrt(h * np.sum(e ** 2))
    Linf = np.max(np.abs(e))
    H1 = np.sqrt(h * np.sum(e ** 2 + de ** 2))
    return (L2, Linf, H1)
import csv

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import csv

def rbf_family(gamma):

    def K(x, y):
        d = x - y
        return np.exp(-gamma * d * d)

    def dK(x, y):
        d = x - y
        return -2.0 * gamma * d * np.exp(-gamma * d * d)

    def ddK(x, y):
        d = x - y
        return (-2.0 * gamma + 4.0 * gamma ** 2 * d * d) * np.exp(-gamma * d * d)
    return (K, dK, ddK)

def poly_family(degree, coef0):

    def K(x, y):
        return (x * y + coef0) ** degree

    def dK(x, y):
        return degree * y * (x * y + coef0) ** (degree - 1)

    def ddK(x, y):
        return degree * (degree - 1) * y * y * (x * y + coef0) ** (degree - 2)
    return (K, dK, ddK)

def combine(weights, x, y, which):
    """which: 0 -> K, 1 -> dK/dx, 2 -> d2K/dx2, for the weighted kernel sum."""
    out = 0.0
    for w, (_, fam) in zip(weights, KERNEL_BANK.items()):
        out = out + w * fam[which](x, y)
    return out

def lssvr_inverse(x_obs, u_obs, N=24, weights=None, w_data=None, w_bc=1000.0, noise_std=0.0, reg=0.001, max_gn=60, tol=1e-10, alpha0=None, beta0=None, verbose=False):
    """
    Identify (alpha, beta) AND the state (c, b) simultaneously.
    weights   : MKL weights over KERNEL_BANK (None -> pure rbf_g8).
    noise_std : relative noise fraction (API name retained for compatibility); if w_data is None the data
                weight is set by inverse-variance weighting, i.e. noisy data
                are trusted LESS and the PDE residual acts as a regulariser.
    reg       : Tikhonov penalty on the state coefficients c.
    """
    if w_data is None:
        w_data = min(1000.0, 1.0 / max(noise_std, 1e-12))
    reg_eff = max(reg * (noise_std / 0.01) ** 2, 0.0) if noise_std > 0 else 0.0
    j = np.arange(N)
    xc = 0.5 * (1.0 - np.cos(np.pi * j / (N - 1)))
    if weights is None:
        weights = np.array([0.0, 1.0, 0.0, 0.0])
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    Kblocks, dKblocks, ddKblocks = ([], [], [])
    for _, fam in KERNEL_BANK.items():
        Kf, dKf, ddKf = fam
        Kblocks.append(Kf(xc[:, None], xc[None, :]))
        dKblocks.append(dKf(xc[:, None], xc[None, :]))
        ddKblocks.append(ddKf(xc[:, None], xc[None, :]))

    def blocks(w):
        Km = sum((wi * Kb for wi, Kb in zip(w, Kblocks)))
        dKm = sum((wi * dKb for wi, dKb in zip(w, dKblocks)))
        ddKm = sum((wi * ddKb for wi, ddKb in zip(w, ddKblocks)))
        return (Km, dKm, ddKm)
    Km, dKm, ddKm = blocks(weights)

    def obs_rows(w):
        Ko = sum((wi * fam[0](x_obs[:, None], xc[None, :]) for wi, (_, fam) in zip(w, KERNEL_BANK.items())))
        return Ko
    Ko = obs_rows(weights)
    b_row_du = sum((wi * fam[1](np.array([0.0]), xc) for wi, (_, fam) in zip(weights, KERNEL_BANK.items())))
    b_row_u = sum((wi * fam[0](np.array([1.0]), xc) for wi, (_, fam) in zip(weights, KERNEL_BANK.items())))
    if alpha0 is None:
        alpha0 = 1.0
    if beta0 is None:
        beta0 = 1.0
    theta = np.concatenate([np.zeros(N), [0.0, alpha0, beta0]])

    def residual_jacobian(theta):
        c, b, alpha, beta = (theta[:N], theta[N], theta[N + 1], theta[N + 2])
        u = Km @ c + b
        du = dKm @ c
        ddu = ddKm @ c
        g = (1.0 + du ** 2) ** (-0.5)
        gp = -du * (1.0 + du ** 2) ** (-1.5)
        Fpde = ddu - alpha * u + beta * g
        Jc = ddKm - alpha * Km + beta * (gp[:, None] * dKm)
        Jb = -alpha * np.ones(N)
        Ja = -u
        Jbe = g
        J = np.hstack([Jc, Jb[:, None], Ja[:, None], Jbe[:, None]])
        F = Fpde
        u_pred = Ko @ c + b
        Fd = w_data * (u_pred - u_obs)
        Jd = np.hstack([w_data * Ko, w_data * np.ones((len(x_obs), 1)), np.zeros((len(x_obs), 2))])
        J = np.vstack([J, Jd])
        F = np.concatenate([F, Fd])
        Jb1 = np.concatenate([w_bc * b_row_du, [0.0, 0.0, 0.0]])
        Fb1 = w_bc * (b_row_du @ c)
        Jb2 = np.concatenate([w_bc * b_row_u, [w_bc, 0.0, 0.0]])
        Fb2 = w_bc * (b_row_u @ c + b)
        J = np.vstack([J, Jb1[None, :], Jb2[None, :]])
        F = np.concatenate([F, [Fb1, Fb2]])
        if reg_eff > 0:
            Rg = np.hstack([np.sqrt(reg_eff) * np.eye(N), np.zeros((N, 3))])
            J = np.vstack([J, Rg])
            F = np.concatenate([F, np.sqrt(reg_eff) * c])
        return (F, J)
    hist = []
    for it in range(max_gn):
        F, J = residual_jacobian(theta)
        nrm = np.linalg.norm(F)
        hist.append(nrm)
        if verbose:
            print(f'  GN it {it:2d}  ||F||={nrm:.3e}  alpha={theta[N + 1]:.6f} beta={theta[N + 2]:.6f}')
        if nrm < tol:
            break
        step, *_ = np.linalg.lstsq(J, -F, rcond=None)
        lam = 1.0
        accepted = False
        while lam > 1e-05:
            tn = theta + lam * step
            if tn[N + 1] > 0 and tn[N + 2] > 0:
                Fn, _ = residual_jacobian(tn)
                if np.linalg.norm(Fn) < (1 - 0.0001 * lam) * nrm:
                    accepted = True
                    break
            lam *= 0.5
        if not accepted:
            break
        theta = tn
        if lam * np.linalg.norm(step) < 1e-12 * (1 + np.linalg.norm(theta)):
            break
    return (theta[:N], theta[N], theta[N + 1], theta[N + 2], hist, xc, weights)

def profile(x, xc, c, b, weights):
    u = np.zeros_like(x)
    for xi_i, xi in enumerate(x):
        s = 0.0
        for wi, (_, fam) in zip(weights, KERNEL_BANK.items()):
            s = s + wi * np.sum(c * fam[0](xi, xc))
        u[xi_i] = s + b
    return u

def learn_mkl_weights(x_obs, u_obs, N=24, iters=25, lr=0.15, noise_std=0.0, verbose=False):
    w = np.full(len(KERNEL_BANK), 1.0 / len(KERNEL_BANK))
    for it in range(iters):
        c, b, a_id, b_id, hist, xc, w_used = lssvr_inverse(x_obs, u_obs, N=N, weights=w, noise_std=noise_std, max_gn=40)

        def misfit(wv):
            wv = np.clip(wv, 0, None)
            wv = wv / wv.sum()
            Ko = sum((wi * fam[0](x_obs[:, None], xc[None, :]) for wi, (_, fam) in zip(wv, KERNEL_BANK.items())))
            return 0.5 * np.sum((Ko @ c + b - u_obs) ** 2)
        f0 = misfit(w)
        grad = np.zeros_like(w)
        eps = 0.0001
        for k in range(len(w)):
            wp = w.copy()
            wp[k] += eps
            wm = w.copy()
            wm[k] -= eps
            grad[k] = (misfit(wp) - misfit(wm)) / (2 * eps)
        w_new = w - lr * grad
        w_new = np.clip(w_new, 0.001, None)
        w = w_new / w_new.sum()
        if verbose:
            print(f'  MKL it {it:2d}  misfit={f0:.3e}  w={np.round(w, 3)}')
    return w

KERNEL_BANK = {"rbf_g4": rbf_family(4.), "rbf_g8": rbf_family(8.), "rbf_g16": rbf_family(16.), "poly3": poly_family(3, 1.)}
