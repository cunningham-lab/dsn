import numpy as np


def sample_LRRNN(N, params):
    nettype = params["nettype"]
    if nettype == "rank1_spont":
        g = params["g"]
        Mm = params["Mm"]
        Mn = params["Mn"]
        Sm = params["Sm"]
        Sn = params["Sn"]
        x1 = np.random.normal(0.0, 1.0, (N, 1))
        x2 = np.random.normal(0.0, 1.0, (N, 1))
        m = Mm + Sm * x1
        n = Mn + Sn * x2
        xi = np.random.normal(0.0, 1.0 / np.sqrt(N), (N, N))
        P = np.dot(m, n.T) / N
        W = g * xi + P
        LRRNN = {"W": W, "m": m, "n": n, "xi": xi}
    elif nettype == "rank1_input":
        g = params["g"]
        Mm = params["Mm"]
        Mn = params["Mn"]
        MI = params["MI"]
        Sm = params["Sm"]
        Sn = params["Sn"]
        SmI = params["SmI"]
        SnI = params["SnI"]
        Sperp = params["Sperp"]
        x1 = np.random.normal(0.0, 1.0, (N, 1))
        x2 = np.random.normal(0.0, 1.0, (N, 1))
        h = np.random.normal(0.0, 1.0, (N, 1))
        m = Mm + Sm * x1
        n = Mn + Sn * x2
        I = MI + (SmI / Sm) * x1 + (SnI / Sn) * x2 + Sperp * h

        xi = np.random.normal(0.0, 1.0 / np.sqrt(N), (N, N))
        P = np.dot(m, n.T) / N
        W = g * xi + P
        LRRNN = {"W": W, "m": m, "n": n, "xi": xi, "I": I}
    else:
        raise NotImplemented()

    return LRRNN


def sim_RNN(x0, W, I, dt, tau, T):
    N = x0.shape[0]
    x = np.zeros((N, T + 1))
    x[:, 0] = x0
    fac = dt / tau
    x_i = x0
    for i in range(T):
        dx = fac * (-x_i + np.dot(W, np.tanh(x_i)) + I)
        x_i = x_i + dx
        x[:, i + 1] = x_i

    return x


def measure_mu(kappa, m, I, t_start):
    mu = np.mean(kappa * m + I)
    return mu


def measure_kappa(x, n, t_start):
    N = len(n)
    x_valid = x[:, t_start:]
    phi_means = np.tanh(np.mean(x_valid, axis=1))
    r = np.dot(phi_means, n) / N
    return r


def measure_vars(x, t_start):
    x_valid = x[:, t_start:]
    x_means = np.mean(x_valid, axis=1)
    x_vars = np.var(x_valid, axis=1)
    delta_inf = np.var(x_means)
    delta_T = np.mean(x_vars)
    return delta_inf, delta_T
