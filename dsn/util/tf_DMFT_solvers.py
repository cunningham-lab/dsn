import numpy as np
import tensorflow as tf
from tf_util.tf_util import get_array_str
import dsn.util.tf_integrals as tfi
from dsn.util.tf_langevin import bounded_langevin_dyn, bounded_langevin_dyn_np
import dsn.util.np_integrals as npi
import os

DTYPE = tf.float64


def rank1_spont_static_solve(
    mu_init, delta_0_init, g, Mm, Mn, Sm, num_its, eps, gauss_quad_pts=50
):

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        mu = x[:, 0]
        delta_0 = x[:, 1]

        Phi = tfi.Phi(mu, delta_0, num_pts=gauss_quad_pts)
        PhiSq = tfi.PhiSq(mu, delta_0, num_pts=gauss_quad_pts)

        F = Mm * Mn * Phi
        H = (g ** 2) * PhiSq + (Sm ** 2) * (Mn ** 2) * Phi ** 2
        return tf.stack([F, H], axis=1)

    x_init = tf.stack([mu_init, delta_0_init], axis=1)
    non_neg = [False, True]
    xs_end = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg)
    mu = xs_end[:, 0]
    delta_0 = xs_end[:, 1]
    return mu, delta_0


def rank1_spont_chaotic_solve(
    mu_init,
    delta_0_init,
    delta_inf_init,
    g,
    Mm,
    Mn,
    Sm,
    num_its,
    eps,
    gauss_quad_pts=50,
    db=False,
):

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        mu = x[:, 0]
        delta_0 = x[:, 1]
        delta_inf = x[:, 2]

        Phi = tfi.Phi(mu, delta_0, num_pts=gauss_quad_pts)
        PrimSq = tfi.PrimSq(mu, delta_0, num_pts=gauss_quad_pts)
        IntPrimPrim = tfi.IntPrimPrim(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)
        IntPhiPhi = tfi.IntPhiPhi(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)

        F = Mm * Mn * Phi
        G_squared = delta_inf ** 2 + 2 * (
            (g ** 2) * (PrimSq - IntPrimPrim)
            + (Mn ** 2) * (Sm ** 2) * (Phi ** 2) * (delta_0 - delta_inf)
        )
        G = tf.sqrt(tf.nn.relu(G_squared))
        H = (g ** 2) * IntPhiPhi + (Mn ** 2) * (Sm ** 2) * (Phi ** 2)
        return tf.stack([F, G, H], axis=1)

    x_init = tf.stack([mu_init, delta_0_init, delta_inf_init], axis=1)
    non_neg = [False, True, True]
    if db:
        xs_end, xs = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg, db=db)
    else:
        xs_end = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg, db=db)

    mu = xs_end[:, 0]
    delta_0 = xs_end[:, 1]
    delta_inf = xs_end[:, 2]

    if db:
        return mu, delta_0, delta_inf, xs
    else:
        return mu, delta_0, delta_inf


def rank1_input_chaotic_solve(
    mu_init,
    kappa_init,
    delta_0_init,
    delta_inf_init,
    g,
    Mm,
    Mn,
    MI,
    Sm,
    Sn,
    SmI,
    SnI,
    Sperp,
    num_its,
    eps,
    gauss_quad_pts=50,
    db=False,
):

    square_diff_init = (tf.square(delta_0_init) - tf.square(delta_inf_init)) / 2.0
    SI_squared = (SmI ** 2 / Sm ** 2) + (SnI ** 2) / (Sn ** 2) + Sperp ** 2

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        mu = x[:, 0]
        kappa = x[:, 1]
        square_diff = x[:, 2]
        delta_inf = x[:, 3]

        delta_0 = tf.sqrt(2 * square_diff + tf.square(delta_inf))

        Phi = tfi.Phi(mu, delta_0, num_pts=gauss_quad_pts)
        Prime = tfi.Prime(mu, delta_0, num_pts=gauss_quad_pts)
        PrimSq = tfi.PrimSq(mu, delta_0, num_pts=gauss_quad_pts)
        IntPrimPrim = tfi.IntPrimPrim(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)
        IntPhiPhi = tfi.IntPhiPhi(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)

        F = Mm * kappa + MI  # mu
        G = Mn * Phi + SnI * Prime
        H = tf.square(g) * (PrimSq - IntPrimPrim) + (
            tf.square(Sm) * tf.square(kappa) + 2 * SmI * kappa + SI_squared
        ) * (delta_0 - delta_inf)
        I = (
            tf.square(g) * IntPhiPhi
            + tf.square(Sm) * tf.square(kappa)
            + 2 * SmI * kappa
            + SI_squared
        )

        return tf.stack([F, G, H, I], axis=1)

    x_init = tf.stack([mu_init, kappa_init, square_diff_init, delta_inf_init], axis=1)
    non_neg = [False, False, True, True]

    if db:
        xs_end, xs = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg, db=db)
    else:
        xs_end = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg, db=db)

    mu = xs_end[:, 0]
    kappa = xs_end[:, 1]
    square_diff = xs_end[:, 2]
    delta_inf = xs_end[:, 3]

    delta_0 = tf.sqrt(2 * square_diff + tf.square(delta_inf))

    if db:
        return mu, kappa, delta_0, delta_inf, xs
    else:
        return mu, kappa, delta_0, delta_inf


def rank2_CDD_static_solve(
    kappa1_init,
    kappa2_init,
    delta_0_init,
    cA,
    cB,
    g,
    rhom,
    rhon,
    betam,
    betan,
    gammaA,
    gammaB,
    num_its,
    eps,
    gauss_quad_pts=50,
    db=False,
):
    # Use equations 159 and 160 from M&O 2018

    SI = 1.2
    Sy = 1.2

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        kappa1 = x[:, 0]
        kappa2 = x[:, 1]
        delta_0 = x[:, 2]

        mu = tf.zeros((1,), dtype=DTYPE)

        Prime = tfi.Prime(mu, delta_0, num_pts=gauss_quad_pts)
        PhiSq = tfi.PhiSq(mu, delta_0, num_pts=gauss_quad_pts)

        F = (
            rhom * rhon * kappa1
            + betam * betan * (kappa1 + kappa2)
            + cA * (SI ** 2)
            + rhon * gammaA
        ) * Prime
        G = (
            rhom * rhon * kappa2
            + betam * betan * (kappa1 + kappa2)
            + cB * (SI ** 2)
            + rhon * gammaB
        ) * Prime
        H = (g ** 2) * PhiSq
        H += ((Sy ** 2) + tf.square(betam)) * (tf.square(kappa1) + tf.square(kappa2))
        H += (
            (SI ** 2) * (cA ** 2 + cB ** 2)
            + tf.square(rhom * kappa1 + gammaA)
            + tf.square(rhom * kappa2 + gammaB)
        )

        return tf.stack([F, G, H], axis=1)

    x_init = tf.stack([kappa1_init, kappa2_init, delta_0_init], axis=1)
    non_neg = [False, False, True]

    if db:
        xs_end, xs = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg, db=db)
    else:
        xs_end = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg, db=db)

    kappa1 = xs_end[:, 0]
    kappa2 = xs_end[:, 1]
    delta_0 = xs_end[:, 2]

    mu = tf.zeros((1,), dtype=DTYPE)

    Prime = tfi.Prime(mu, delta_0, num_pts=gauss_quad_pts)

    z = betam * (kappa1 + kappa2) * Prime

    if db:
        return kappa1, kappa2, delta_0, z, xs
    else:
        return kappa1, kappa2, delta_0, z


def rank2_CDD_chaotic_solve(
    kappa1_init,
    kappa2_init,
    delta_0_init,
    delta_inf_init,
    cA,
    cB,
    g,
    rhom,
    rhon,
    betam,
    betan,
    gammaA,
    gammaB,
    num_its,
    eps,
    gauss_quad_pts=50,
    db=False,
):

    SI = 1.2
    Sy = 1.2

    SyA = Sy
    SyB = Sy
    SIA = SI
    SIB = SI
    SIctxA = 1.0
    SIctxB = 1.0
    Sw = 1.0

    Sm1 = SyA + rhom * SIctxA + betam * Sw
    Sm2 = SyB + rhom * SIctxB + betam * Sw

    square_diff_init = (tf.square(delta_0_init) - tf.square(delta_inf_init)) / 2.0

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        kappa1 = x[:, 0]
        kappa2 = x[:, 1]
        square_diff = x[:, 2]
        delta_inf = x[:, 3]

        mu = tf.zeros((1,), dtype=DTYPE)
        delta_0 = tf.sqrt(2 * square_diff + tf.square(delta_inf))

        Prime = tfi.Prime(mu, delta_0, num_pts=gauss_quad_pts)
        PrimSq = tfi.PrimSq(mu, delta_0, num_pts=gauss_quad_pts)
        IntPrimPrim = tfi.IntPrimPrim(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)
        IntPhiPhi = tfi.IntPhiPhi(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)

        noise_corr = (
            (Sw ** 2 + tf.square(betam)) * (tf.square(kappa1) + tf.square(kappa2))
            + (SI ** 2) * (cA ** 2 + cB ** 2)
            + tf.square(rhom * kappa1 + gammaA)
            + tf.square(rhom * kappa2 + gammaB)
        )

        F = (
            rhom * rhon * kappa1
            + betam * betan * (kappa1 + kappa2)
            + cA * SI
            + rhon * gammaA
        ) * Prime
        G = (
            rhom * rhon * kappa2
            + betam * betan * (kappa1 + kappa2)
            + cB * SI
            + rhon * gammaB
        ) * Prime
        H = tf.square(g) * (PrimSq - IntPrimPrim) + noise_corr * (delta_0 - delta_inf)
        I = tf.square(g) * IntPhiPhi + noise_corr

        return tf.stack([F, G, H, I], axis=1)

    x_init = tf.stack(
        [kappa1_init, kappa2_init, square_diff_init, delta_inf_init], axis=1
    )
    non_neg = [False, False, True, True]
    if db:
        xs_end, xs = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg, db=db)
    else:
        xs_end = bounded_langevin_dyn(f, x_init, eps, num_its, non_neg, db=db)

    kappa1 = xs_end[:, 0]
    kappa2 = xs_end[:, 1]
    square_diff = xs_end[:, 2]
    delta_inf = xs_end[:, 3]

    mu = tf.zeros((1,), dtype=DTYPE)
    delta_0 = tf.sqrt(2 * square_diff + tf.square(delta_inf))

    Prime = tfi.Prime(mu, delta_0, num_pts=gauss_quad_pts)

    z = betam * (kappa1 + kappa2) * Prime

    if db:
        return kappa1, kappa2, delta_0, delta_inf, z, xs
    else:
        return kappa1, kappa2, delta_0, delta_inf, z


def rank1_spont_static_solve_np(mu_init, delta_0_init, g, Mm, Mn, Sm, num_its, eps):
    def f(x):
        mu = x[:, 0]
        delta_0 = x[:, 1]

        Phi = npi.Phi(mu, delta_0)
        PhiSq = npi.PhiSq(mu, delta_0)

        F = Mm * Mn * Phi
        H = (g ** 2) * PhiSq + (Sm ** 2) * (Mn ** 2) * Phi ** 2
        return np.stack([F, H], axis=1)

    x_init = np.stack([mu_init, delta_0_init], axis=1)
    non_neg = [False, True]
    xs_end = bounded_langevin_dyn_np(f, x_init, eps, num_its, non_neg)
    mu = xs_end[:, 0]
    delta_0 = xs_end[:, 1]
    return mu, delta_0


def rank1_input_chaotic_solve_np(
    mu_init,
    kappa_init,
    delta_0_init,
    delta_inf_init,
    g,
    Mm,
    Mn,
    MI,
    Sm,
    Sn,
    SmI,
    SnI,
    Sperp,
    num_its,
    eps,
    gauss_quad_pts=50,
    db=False,
):

    square_diff_init = (np.square(delta_0_init) - np.square(delta_inf_init)) / 2.0
    SI_squared = (SmI ** 2 / Sm ** 2) + (SnI ** 2) / (Sn ** 2) + Sperp ** 2

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        mu = x[:, 0]
        kappa = x[:, 1]
        square_diff = x[:, 2]
        delta_inf = x[:, 3]

        delta_0 = np.sqrt(2 * square_diff + np.square(delta_inf))

        Phi = npi.Phi(mu, delta_0, num_pts=gauss_quad_pts)
        Prime = npi.Prime(mu, delta_0, num_pts=gauss_quad_pts)
        PrimSq = npi.PrimSq(mu, delta_0, num_pts=gauss_quad_pts)
        IntPrimPrim = npi.IntPrimPrim(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)
        IntPhiPhi = npi.IntPhiPhi(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)

        F = Mm * kappa + MI  # mu
        G = Mn * Phi + SnI * Prime
        H = np.square(g) * (PrimSq - IntPrimPrim) + (
            np.square(Sm) * np.square(kappa) + 2 * SmI * kappa + SI_squared
        ) * (delta_0 - delta_inf)
        I = (
            np.square(g) * IntPhiPhi
            + np.square(Sm) * np.square(kappa)
            + 2 * SmI * kappa
            + SI_squared
        )

        return np.stack([F, G, H, I], axis=1)

    x_init = np.stack([mu_init, kappa_init, square_diff_init, delta_inf_init], axis=1)
    non_neg = [False, False, True, True]

    if db:
        xs_end, xs = bounded_langevin_dyn_np(f, x_init, eps, num_its, non_neg, db=db)
    else:
        xs_end = bounded_langevin_dyn_np(f, x_init, eps, num_its, non_neg, db=db)

    mu = xs_end[:, 0]
    kappa = xs_end[:, 1]
    square_diff = xs_end[:, 2]
    delta_inf = xs_end[:, 3]

    delta_0 = np.sqrt(2 * square_diff + np.square(delta_inf))

    if db:
        return mu, kappa, delta_0, delta_inf, xs
    else:
        return mu, kappa, delta_0, delta_inf


def rank2_CDD_static_solve_np(
    kappa1_init,
    kappa2_init,
    delta_0_init,
    cA,
    cB,
    g,
    rhom,
    rhon,
    betam,
    betan,
    gammaA,
    gammaB,
    num_its,
    eps,
    num_pts=200,
    db=False,
):
    # Use equations 159 and 160 from M&O 2018

    SI = 1.2
    Sy = 1.2

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        kappa1 = x[:, 0]
        kappa2 = x[:, 1]
        delta_0 = x[:, 2]

        mu = np.zeros((1,))

        Prime = npi.Prime(mu, delta_0, num_pts=num_pts)
        PhiSq = npi.PhiSq(mu, delta_0, num_pts=num_pts)

        F = (
            rhom * rhon * kappa1
            + betam * betan * (kappa1 + kappa2)
            + cA * (SI ** 2)
            + rhon * gammaA
        ) * Prime
        G = (
            rhom * rhon * kappa2
            + betam * betan * (kappa1 + kappa2)
            + cB * (SI ** 2)
            + rhon * gammaB
        ) * Prime
        H = (g ** 2) * PhiSq
        H += ((Sy ** 2) + np.square(betam)) * (np.square(kappa1) + np.square(kappa2))
        H += (
            (SI ** 2) * (cA ** 2 + cB ** 2)
            + np.square(rhom * kappa1 + gammaA)
            + np.square(rhom * kappa2 + gammaB)
        )

        return np.stack([F, G, H], axis=1)

    x_init = np.stack([kappa1_init, kappa2_init, delta_0_init], axis=1)
    non_neg = [False, False, True]

    if db:
        xs_end, xs = bounded_langevin_dyn_np(f, x_init, eps, num_its, non_neg, db=db)
    else:
        xs_end = bounded_langevin_dyn_np(f, x_init, eps, num_its, non_neg, db=db)

    kappa1 = xs_end[:, 0]
    kappa2 = xs_end[:, 1]
    delta_0 = xs_end[:, 2]

    mu = np.zeros((1,))

    Prime = npi.Prime(mu, delta_0)

    z = betam * (kappa1 + kappa2) * Prime

    if db:
        return kappa1, kappa2, delta_0, z, xs
    else:
        return kappa1, kappa2, delta_0, z


def warm_start(system):
    assert system.name == "LowRankRNN"
    ws_filename = get_warm_start_dir(system)
    print(ws_filename)
    ws_its = 1500
    if not os.path.isfile(ws_filename):
        rank = system.model_opts["rank"]
        behavior_type = system.behavior["type"]
        if rank == 2 and behavior_type == "CDD":
            cAs = [0, 1]
            cBs = [0, 1]
            a = system.a
            b = system.b
            step = system.warm_start_grid_step
            grid_vals_list = []
            nvals = []
            j = 0
            for i in range(len(system.all_params)):
                param = system.all_params[i]
                if param in system.free_params:
                    vals = np.arange(a[j], b[j] + step, step)
                    j += 1
                else:
                    vals = np.array([system.fixed_params[param]])
                grid_vals_list.append(vals)
                nvals.append(vals.shape[0])
            m = np.prod(np.array(nvals))

            grid = np.array(np.meshgrid(*grid_vals_list))
            grid = np.reshape(grid, (len(system.all_params), m))

            solution_grids = np.zeros((2, 2, m, 3))
            for cA in cAs:
                for cB in cBs:
                    _cA = cA * np.ones((m,))
                    _cB = cB * np.ones((m,))
                    kappa1_init = -5.0 * np.ones((m,))
                    kappa2_init = -4.0 * np.ones((m,))
                    delta0_init = 2.0 * np.ones((m,))
                    kappa1, kappa2, delta_0, z, xs = rank2_CDD_static_solve_np(
                        kappa1_init,
                        kappa2_init,
                        delta0_init,
                        _cA,
                        _cB,
                        grid[0],
                        grid[1],
                        grid[2],
                        grid[3],
                        grid[4],
                        grid[5],
                        grid[6],
                        ws_its,
                        system.solve_eps,
                        num_pts=50,
                        db=True,
                    )

                    solution_grids[cA, cB] = np.stack((kappa1, kappa2, delta_0), axis=1)
        elif rank == 1 and behavior_type == "BI":
            step = system.warm_start_grid_step
            a = system.a
            b = system.b
            free_param_inds = []
            grid_vals_list = []
            nvals = []
            j = 0
            for i in range(len(system.all_params)):
                param = system.all_params[i]
                if param in system.free_params:
                    vals = np.arange(a[j], b[j] + step, step)
                    free_param_inds.append(i)
                    j += 1
                else:
                    vals = np.array([system.fixed_params[param]])
                grid_vals_list.append(vals)
                nvals.append(vals.shape[0])
            print("nvals", nvals)
            m = np.prod(np.array(nvals))
            print("m", m)

            grid = np.array(np.meshgrid(*grid_vals_list))
            grid = np.reshape(grid, (len(system.all_params), m))

            mu_init = 5.0 * np.ones((m,))
            kappa_init = 5.0 * np.ones((m,))
            delta_0_init = 5.0 * np.ones((m,))
            delta_inf_init = 4.0 * np.ones((m,))

            mu, kappa, delta_0, delta_inf, xs = rank1_input_chaotic_solve_np(
                mu_init,
                kappa_init,
                delta_0_init,
                delta_inf_init,
                grid[0],
                grid[1],
                grid[2],
                grid[3],
                grid[4],
                grid[5],
                grid[6],
                grid[7],
                grid[8],
                ws_its,
                system.solve_eps,
                gauss_quad_pts=50,
                db=True,
            )
            solution_grid = np.stack((mu, kappa, delta_0, delta_inf), axis=1)

        np.savez(
            ws_filename,
            param_grid=grid[free_param_inds, :],
            solution_grid=solution_grid,
            xs=xs,
        )
    else:
        print("Already warm_started.")
        print(ws_filename)
        npzfile = np.load(ws_filename)
        xs = npzfile["xs"]
    return ws_filename, xs


def get_warm_start_dir(system):
    rank = system.model_opts["rank"]
    type = system.behavior["type"]
    a_str = get_array_str(system.a)
    b_str = get_array_str(system.b)
    step = system.warm_start_grid_step
    ws_dir = "data/warm_starts/"
    if not os.path.isdir(ws_dir):
        os.makedirs(ws_dir)
    ws_filename = ws_dir + "rank%d_%s_a=%s_b=%s_step=%.2E.npz" % (
        rank,
        type,
        a_str,
        b_str,
        step,
    )
    return ws_filename
