# Copyright 2019 Sean Bittner, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import numpy as np
import tensorflow as tf
import dsn.util.tf_integrals as tfi
from dsn.util.tf_langevin import langevin_dyn_rank1_spont_static, \
                                 langevin_dyn_rank1_spont_chaos, \
                                 langevin_dyn_rank1_input_chaos, \
                                 langevin_dyn_rank2_CDD_chaos

DTYPE = tf.float64

def rank1_spont_static_solve(mu_init, delta_0_init, g, Mm, Mn, Sm, num_its, eps):

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        mu = x[:, 0]
        delta_0 = x[:, 1]

        Phi = tfi.Phi(mu, delta_0)
        PhiSq = tfi.PhiSq(mu, delta_0)

        F = Mm * Mn * Phi
        H = (g ** 2) * PhiSq + (Sm ** 2) * (Mn ** 2) * Phi ** 2
        return tf.stack([F, H], axis=1)

    x_init = tf.stack([mu_init, delta_0_init], axis=1)
    xs_end = langevin_dyn_rank1_spont_static(f, x_init, eps, num_its)
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
    if db:
        xs_end, xs = langevin_dyn_rank1_spont_chaos(f, x_init, eps, num_its, db=db)
    else:
        xs_end = langevin_dyn_rank1_spont_chaos(f, x_init, eps, num_its, db=db)

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
    SI_squared = (SmI**2/Sm**2) + (SnI**2)/(Sn**2) + Sperp**2

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        mu = x[:, 0]
        kappa = x[:, 1]
        square_diff = x[:, 2]
        delta_inf = x[:, 3]

        delta_0 = tf.sqrt(2*square_diff + tf.square(delta_inf))

        Phi = tfi.Phi(mu, delta_0, num_pts=gauss_quad_pts)
        Prime = tfi.Prime(mu, delta_0, num_pts=gauss_quad_pts)
        PrimSq = tfi.PrimSq(mu, delta_0, num_pts=gauss_quad_pts)
        IntPrimPrim = tfi.IntPrimPrim(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)
        IntPhiPhi = tfi.IntPhiPhi(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)

        F = Mm*kappa + MI # mu
        G = Mn*Phi + SnI*Prime
        H = tf.square(g)*(PrimSq - IntPrimPrim) + \
            (tf.square(Sm)*tf.square(kappa) + 2*SmI*kappa + SI_squared)*(delta_0 - delta_inf)
        I = tf.square(g)*IntPhiPhi + tf.square(Sm)*tf.square(kappa) + 2*SmI*kappa + SI_squared

        return tf.stack([F, G, H, I], axis=1)

    x_init = tf.stack([mu_init, kappa_init, square_diff_init, delta_inf_init], axis=1)

    if db:
        xs_end, xs = langevin_dyn_rank1_input_chaos(f, x_init, eps, num_its, db=db)
    else:
        xs_end = langevin_dyn_rank1_input_chaos(f, x_init, eps, num_its, db=db)

    mu = xs_end[:, 0]
    kappa = xs_end[:,1]
    square_diff = xs_end[:, 2]
    delta_inf = xs_end[:, 3]

    delta_0 = tf.sqrt(2*square_diff + tf.square(delta_inf))

    if db:
        return mu, kappa, delta_0, delta_inf, xs
    else:
        return mu, kappa, delta_0, delta_inf

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

    Sm1 = SyA + rhom*SIctxA + betam*Sw
    Sm2 = SyB + rhom*SIctxB + betam*Sw

    square_diff_init = (tf.square(delta_0_init) - tf.square(delta_inf_init)) / 2.0

    # convergence equations used for langevin-like dynamimcs solver
    def f(x):
        kappa1 = x[:, 0]
        kappa2 = x[:, 1]
        square_diff = x[:, 2]
        delta_inf = x[:, 3]

        mu = tf.zeros((1,), dtype=DTYPE)
        delta_0 = tf.sqrt(2*square_diff + tf.square(delta_inf))

        Prime = tfi.Prime(mu, delta_0, num_pts=gauss_quad_pts)
        PrimSq = tfi.PrimSq(mu, delta_0, num_pts=gauss_quad_pts)
        IntPrimPrim = tfi.IntPrimPrim(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)
        IntPhiPhi = tfi.IntPhiPhi(mu, delta_0, delta_inf, num_pts=gauss_quad_pts)

        noise_corr = (Sw**2 + tf.square(betam))*(tf.square(kappa1) + tf.square(kappa2)) \
                   + (SI**2)*(cA**2 + cB**2) + tf.square(rhom*kappa1 + gammaA) \
                   + tf.square(rhom*kappa2 + gammaB)

        F = (rhom*rhon*kappa1 + betam*betan*(kappa1+kappa2) + cA*SI + rhon*gammaA)*Prime
        G = (rhom*rhon*kappa2 + betam*betan*(kappa1+kappa2) + cB*SI + rhon*gammaB)*Prime
        H = tf.square(g)*(PrimSq - IntPrimPrim) + noise_corr*(delta_0 - delta_inf)
        I = tf.square(g)*IntPhiPhi + noise_corr
        
        return tf.stack([F, G, H, I], axis=1)

    x_init = tf.stack([kappa1_init, kappa2_init, square_diff_init, delta_inf_init], axis=1)

    if db:
        xs_end, xs = langevin_dyn_rank2_CDD_chaos(f, x_init, eps, num_its, db=db)
    else:
        xs_end = langevin_dyn_rank2_CDD_chaos(f, x_init, eps, num_its, db=db)

    kappa1 = xs_end[:, 0]
    kappa2 = xs_end[:,1]
    square_diff = xs_end[:, 2]
    delta_inf = xs_end[:, 3]

    mu = tf.zeros((1,), dtype=DTYPE)
    delta_0 = tf.sqrt(2*square_diff + tf.square(delta_inf))

    Prime = tfi.Prime(mu, delta_0, num_pts=gauss_quad_pts)
    
    z = betam*(kappa1+kappa2)*Prime


    if db:
        return kappa1, kappa2, delta_0, delta_inf, z, xs
    else:
        return kappa1, kappa2, delta_0, delta_inf, z



