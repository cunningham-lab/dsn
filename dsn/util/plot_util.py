import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.manifold import TSNE
import tensorflow as tf
from scipy import stats
import os
import time
from matplotlib import animation

def assess_constraints(fnames, alpha, frac_samps, n_suff_stats):

    n_fnames = len(fnames)
    AL_final_its = []
    pvals_list = []
    for i in range(n_fnames):
        fname = fnames[i]
        try:
            npzfile = np.load(fname)
        except:
            continue
        print('T_xs', npzfile["T_xs"].shape)
        mu = npzfile["mu"]
        total_samps = npzfile["T_xs"].shape[1]
        k_max = npzfile["T_xs"].shape[0] - 1
        p_values = np.zeros((k_max + 1, n_suff_stats))

        boot_samps = 200
        samps_per_boot = int(frac_samps * total_samps)

        for k in range(k_max):
            T_xs = npzfile["T_xs"][k]
            for j in range(n_suff_stats):
                boot_means = np.zeros((boot_samps,))
                for ii in range(boot_samps):
                    inds_ii = np.random.choice(
                        np.arange(total_samps), samps_per_boot, replace=False
                    )
                    boot_means[ii] = np.mean(T_xs[inds_ii, j])
                gt = float(np.sum(boot_means > mu[j]))
                lt = float(np.sum(boot_means < mu[j]))
                p_val = 2 * min(gt / boot_samps, lt / boot_samps)
                p_values[k, j] = p_val

            con_sat = np.prod(p_values[k, :] > (alpha / n_suff_stats))
            if con_sat == 1:
                print('Converged!')
                AL_final_its.append(k)
                break
            if k == (k_max-1):
                AL_final_its.append(None)
        pvals_list.append(p_values)
    return pvals_list, AL_final_its


def assess_constraints2(fnames, k_max, n_suff_stats, tol=0.1):
    n_fnames = len(fnames)
    AL_final_its = []
    for i in range(n_fnames):
        fname = fnames[i]
        try:
            npzfile = np.load(fname)
        except:
            n_fnames = n_fnames - 1
            continue
        mu = npzfile["mu"]
        if (not (type(tol) == np.ndarray)):
            tol = tol*np.ones((mu.shape[0],))

        for k in range(k_max + 1):
            T_xs = npzfile["T_xs"][k]
            total_samps = T_xs.shape[0]
            T_x_mean = np.mean(T_xs, 0)
            failed = False
            for j in range(n_suff_stats // 2):
                if T_x_mean[j] < (mu[j] - tol[j]) or (mu[j] + tol[j]) < T_x_mean[j]:
                    failed = True
                    break
            for j in range(n_suff_stats // 2, n_suff_stats):
                if mu[j] + tol[j] < T_x_mean[j]:
                    failed = True
                    break
            if not failed:
                AL_final_its.append(k)
                break

        if failed:
            AL_final_its.append(None)

    return AL_final_its


def plot_opt(
    fnames,
    legendstrs=[],
    con_method="1",
    frac_samps=0.2,
    maxconlim=3.0,
    alpha=0.05,
    plotR2=False,
    fontsize=14,
    tol=0.1,
):
    max_legendstrs = 10
    n_fnames = len(fnames)
    # read optimization diagnostics from files
    costs_list = []
    Hs_list = []
    R2s_list = []
    mean_T_xs_list = []
    T_xs_list = []
    epoch_inds_list = []
    last_inds = []
    flag = False
    for i in range(n_fnames):
        fname = fnames[i]
        if (os.path.isfile(fname)):
            try:
                npzfile = np.load(fname)
            except:
                n_fnames = n_fnames - 1
                continue
        else:
            n_fnames = n_fnames - 1
            continue
        costs = npzfile["costs"]
        Hs = npzfile["Hs"]
        R2s = npzfile["R2s"]
        mean_T_xs = npzfile["mean_T_xs"]
        T_xs = npzfile["T_xs"]
        epoch_inds = npzfile["epoch_inds"]

        check_rate = npzfile["check_rate"]
        last_inds.append(npzfile['it'] // check_rate)

        costs_list.append(costs)
        Hs_list.append(Hs)
        R2s_list.append(R2s)
        mean_T_xs_list.append(mean_T_xs)
        epoch_inds_list.append(epoch_inds)

        if (not flag):
            mu = npzfile["mu"]
            check_rate = npzfile["check_rate"]
            last_ind = npzfile["it"] // check_rate
            nits = costs.shape[0]
            k_max = T_xs.shape[0] - 1
            iterations = np.arange(0, check_rate * nits, check_rate)
            n_suff_stats = mean_T_xs_list[0].shape[1]
            p_values, AL_final_its = assess_constraints(
                fnames, alpha, frac_samps, n_suff_stats
            )
            print('al final')
            print(AL_final_its)
            if con_method == "1":
                pass
            elif con_method == "2":
                AL_final_its = assess_constraints2(fnames, k_max, n_suff_stats, tol=tol)
            else:
                raise NotImplementedError()
            flag = True

    figs = []

    # plot cost, entropy and r^2
    num_panels = 3 if plotR2 else 2
    figsize = (num_panels * 4, 4)
    fig, axs = plt.subplots(1, num_panels, figsize=figsize)
    figs.append(fig)
    ax = axs[0]
    for i in range(n_fnames):
        costs = costs_list[i]
        ax.plot(iterations[:last_ind], costs[:last_ind], label=legendstrs[i])
    ax.set_xlabel("iterations", fontsize=fontsize)
    ax.set_ylabel("cost", fontsize=fontsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax = axs[1]
    for i in range(n_fnames):
        Hs = Hs_list[i]
        epoch_inds = epoch_inds_list[i]
        last_ind = last_inds[i]
        if (np.sum(np.isnan(Hs[:last_ind])) > 0):
            print('has nan')
        if i < 5:
            ax.plot(iterations[:last_ind], Hs[:last_ind], label=legendstrs[i])
        else:
            ax.plot(iterations[:last_ind], Hs[:last_ind])
        if n_fnames == 1 and AL_final_its[i] is not None:
            if (epoch_inds.shape[0] < T_xs.shape[0]):
                conv_it = iterations[AL_final_its[i]]
            else:
                conv_it = epoch_inds[AL_final_its[i]]
            ax.plot(
                [conv_it, conv_it],
                [np.min(Hs[:last_ind]), np.max(Hs[:last_ind])],
                "k--",
            )
    ax.set_xlabel("iterations", fontsize=fontsize)
    ax.set_ylabel("H", fontsize=fontsize)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if plotR2:
        ax = axs[2]
        for i in range(n_fnames):
            last_ind = last_inds[i]
            R2s = R2s_list[i]
            epoch_inds = epoch_inds_list[i]
            if i < max_legendstrs:
                ax.plot(iterations[:last_ind], R2s[:last_ind], label=legendstrs[i])
            else:
                ax.plot(iterations[:last_ind], R2s[:last_ind])
        if n_fnames == 1 and AL_final_its[i] is not None:
            if (epoch_inds.shape[0] < T_xs.shape[0]):
                conv_it = iterations[AL_final_its[i]]
            else:
                conv_it = epoch_inds[AL_final_its[i]]
            ax.plot(
                [conv_it, conv_it],
                [np.min(R2s[:last_ind]), np.max(R2s[:last_ind])],
                "k--",
            )
        ax.set_xlabel("iterations", fontsize=fontsize)
        ax.set_ylabel(r"$r^2$", fontsize=fontsize)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    ax.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    # plot constraints throughout optimization
    yscale_fac = 5
    n_cols = min(n_suff_stats, 4)
    n_rows = int(np.ceil(n_suff_stats / n_cols))
    figsize = (n_cols * 4, n_rows * 4)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axs = [axs]
    figs.append(fig)
    for i in range(n_suff_stats):
        ax = axs[i // n_cols][i % n_cols]
        # make ylim 2* mean abs error of last 50% of optimization
        median_abs_errors = np.zeros((n_fnames,))
        for j in range(n_fnames):
            mean_T_xs = mean_T_xs_list[j]
            epoch_inds = epoch_inds_list[j]
            num_epoch_inds = len(epoch_inds)
            last_ind = last_inds[j]
            if j < max_legendstrs:
                ax.plot(
                    iterations[:last_ind], mean_T_xs[:last_ind, i], label=legendstrs[j]
                )
            else:
                ax.plot(iterations[:last_ind], mean_T_xs[:last_ind, i])

            median_abs_errors[j] = np.median(
                np.abs(mean_T_xs[(last_ind // 2) : last_ind, i] - mu[i])
            )
            if n_fnames == 1:
                T_x_means = np.mean(T_xs[:, :, i], axis=1)
                T_x_stds = np.std(T_xs[:, :, i], axis=1)
                num_epoch_inds = len(epoch_inds)
                # ax.errorbar(epoch_inds, T_x_means[:num_epoch_inds], T_x_stds[:num_epoch_inds], c='r', elinewidth=3)
                if AL_final_its[j] is not None:
                    if (epoch_inds.shape[0] < T_xs.shape[0]):
                        conv_it = iterations[AL_final_its[j]]
                    else:
                        conv_it = epoch_inds[AL_final_its[j]]
                    line_min = min(
                        [
                            np.min(mean_T_xs[:last_ind, i]),
                            mu[i] - yscale_fac * median_abs_errors[j],
                            np.min(T_x_means - 2 * T_x_stds),
                        ]
                    )
                    line_max = max(
                        [
                            np.max(mean_T_xs[:last_ind, i]),
                            mu[i] + yscale_fac * median_abs_errors[j],
                            np.max(T_x_means + 2 * T_x_stds),
                        ]
                    )
                    ax.plot([conv_it, conv_it], [line_min, line_max], "k--")

        ax.plot([iterations[0], iterations[last_ind]], [mu[i], mu[i]], "k-")
        # make ylim 2* mean abs error of last 50% of optimization
        if n_fnames == 1:
            ymin = min(
                mu[i] - yscale_fac * np.max(median_abs_errors),
                np.min(
                    T_x_means[(num_epoch_inds // 2) :]
                    - 2 * T_x_stds[(num_epoch_inds // 2) :]
                ),
            )
            ymax = max(
                mu[i] + yscale_fac * np.max(median_abs_errors),
                np.max(
                    T_x_means[(num_epoch_inds // 2) :]
                    + 2 * T_x_stds[(num_epoch_inds // 2) :]
                ),
            )
        else:
            ymin = mu[i] - yscale_fac * np.max(median_abs_errors)
            ymax = mu[i] + yscale_fac * np.max(median_abs_errors)
        if (np.isnan(ymin) or np.isnan(ymax)):
            ax.set_ylim(mu[i] - maxconlim, mu[i] + maxconlim)
        else:
            ax.set_ylim(max(ymin, mu[i] - maxconlim), min(ymax, mu[i] + maxconlim))
        ax.set_ylabel(r"$E[T_%d(z)]$" % (i + 1), fontsize=fontsize)
        if i == (n_cols - 1):
            ax.legend(fontsize=fontsize)
        if i > n_suff_stats - n_cols - 1:
            ax.set_xlabel("iterations", fontsize=fontsize)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.show()

    # plot the p-value based constraint satisfaction
    print("p values")
    n_cols = 4
    n_rows = int(np.ceil(n_fnames / n_cols))
    figsize = (n_cols * 4, n_rows * 4)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    figs.append(fig)
    for i in range(n_fnames):
        print(i)
        ax = plt.subplot(n_rows, n_cols, i + 1)
        for j in range(n_suff_stats):
            ax.plot(
                np.arange(k_max + 1), p_values[i, :, j], label=r"$T_%d(z)$" % (j + 1)
            )
        if AL_final_its[i] is not None:
            ax.plot(
                [AL_final_its[i], AL_final_its[i]], [0, 1], "k--", label="convergence"
            )
            ax.set_title(legendstrs[i], fontsize=fontsize)
        else:
            ax.set_title(legendstrs[i] + " no converge", fontsize=fontsize)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("aug Lag it", fontsize=fontsize)
        ax.set_ylabel("p value", fontsize=fontsize)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    return figs, AL_final_its, p_values


def coloring_from_str(c_str, system, npzfile, AL_final_it):
    cm = plt.cm.get_cmap("viridis")
    vmin = None
    vmax = None
    if c_str == "log_q_z":
        c = npzfile["log_q_zs"][AL_final_it]
        c_label_str = r"$log(q(z))$"
    elif c_str == "real part":
        c = npzfile["T_xs"][AL_final_it, :, 0]
        cm = plt.cm.get_cmap("Reds")
        c_label_str = r"real($\lambda_1$)"
    elif c_str == "dE":
        c = npzfile["T_xs"][AL_final_it, :, 0]
        cm = plt.cm.get_cmap("Greys")
        c_label_str = r"$d_{E,ss}$"
    elif c_str == "dP":
        c = npzfile["T_xs"][AL_final_it, :, 1]
        cm = plt.cm.get_cmap("Blues")
        c_label_str = r"$d_{P,ss}$"
    elif c_str == "dS":
        c = npzfile["T_xs"][AL_final_it, :, 2]
        cm = plt.cm.get_cmap("Reds")
        c_label_str = r"$d_{S,ss}$"
    elif c_str == "dV":
        c = npzfile["T_xs"][AL_final_it, :, 3]
        cm = plt.cm.get_cmap("Greens")
        c_label_str = r"$d_{V,ss}$"

    elif c_str == "ISN":
        _Z = npzfile["Zs"][AL_final_it, :, :]
        n = _Z.shape[0]
        print("running simulations to figure out what steady states are.")
        Z = tf.placeholder(dtype=tf.float64, shape=(1, n, system.D))
        r_t = system.simulate(Z)
        with tf.Session() as sess:
            _r_t = sess.run(r_t, {Z: np.expand_dims(_Z, 0)})

        assert system.behavior["type"] == "difference"
        r_E_ss_1 = _r_t[-1, 0, :, 0, 0]
        r_E_ss_2 = _r_t[-1, 1, :, 0, 0]

        W_EE = system.fixed_params["W_EE"]

        ISN_stat = r_E_ss_1 > (1.0 / np.square(2 * W_EE))
        ISN_running = r_E_ss_2 > (1.0 / np.square(2 * W_EE))

        c = np.zeros((n,))
        c[np.logical_and(ISN_stat, ISN_running)] = 1.0
        c[np.logical_and(np.logical_not(ISN_stat), ISN_running)] = 0.5
        c[np.logical_and(ISN_stat, np.logical_not(ISN_running))] = -0.5
        c[np.logical_and(np.logical_not(ISN_stat), np.logical_not(ISN_running))] = -1.0
        cm = plt.cm.get_cmap("rainbow")
        c_label_str = "ISN"

    elif c_str == "mu":
        c = npzfile["T_xs"][AL_final_it, :, 0]
        cm = plt.cm.get_cmap("Greys")
        c_label_str = r"$\mu$"
    elif c_str == "deltainf":
        c = npzfile["T_xs"][AL_final_it, :, 1]
        cm = plt.cm.get_cmap("Blues")
        c_label_str = r"$\Delta_\infty$"
    elif c_str == "deltaT":
        c = npzfile["T_xs"][AL_final_it, :, 2]
        cm = plt.cm.get_cmap("Reds")
        c_label_str = r"$\Delta_T$"

    elif c_str == "hubfreq":
        c = npzfile["T_xs"][AL_final_it, :, 0]
        cm = plt.cm.get_cmap("jet")
        c_label_str = r"$f_h$"
        vmin = 0.3
        vmax = 0.8
    else:
        # no coloring
        c = np.ones((npzfile["T_xs"].shape[1],))
        c_label_str = ""

    return c, c_label_str, cm, vmin, vmax


def dist_from_str(dist_str, f_str, system, npzfile, AL_final_it):
    dist_label_strs = []
    if dist_str in ["Zs", "T_xs"]:
        dist = npzfile[dist_str][AL_final_it, :, :]
        if f_str == "identity":
            if dist_str == "Zs":
                dist_label_strs = system.z_labels
            elif dist_str == "T_xs":
                dist_label_strs = system.T_x_labels
        elif f_str == "PCA":
            dist, evecs, evals = PCA(dist, dist.shape[1])
            dist_label_strs = ["PC%d" % i for i in range(1, system.D + 1)]
        elif f_str == "tSNE":
            np.random.seed(0)
            dist = TSNE(n_components=2).fit_transform(dist)
            dist_label_strs = ["tSNE 1", "tSNE 2"]
    else:
        raise NotImplementedError()
    return dist, dist_label_strs


def filter_outliers(c, num_stds=4):
    c_mean = np.mean(c)
    c_std = np.std(c)
    all_inds = np.arange(c.shape[0])
    below_inds = all_inds[c < c_mean - num_stds * c_std]
    over_inds = all_inds[c > c_mean + num_stds * c_std]
    plot_inds = all_inds[
        np.logical_and(c_mean - num_stds * c_std <= c, c <= c_mean + num_stds * c_std)
    ]
    return plot_inds, below_inds, over_inds


def plot_var_ellipse(ax, x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    h = plot_ellipse(ax, mean_x, mean_y, std_x, std_y, "k")
    return h


def plot_target_ellipse(ax, i, j, system, mu):
    mean_only = False
    if system.name == "Linear2D":
        if system.behavior["type"] == "oscillation":
            mean_x = mu[j]
            mean_y = mu[i]
            std_x = np.sqrt(mu[j + system.num_suff_stats // 2] - mu[j] ** 2)
            std_y = np.sqrt(mu[i + system.num_suff_stats // 2] - mu[i] ** 2)
    elif system.name in ["V1Circuit", "SCCircuit", "LowRankRNN"]:
        if system.behavior["type"] in ["difference", "standard", "struct_chaos"]:
            mean_x = mu[j]
            mean_y = mu[i]
            std_x = np.sqrt(mu[j + system.num_suff_stats // 2] - mu[j] ** 2)
            std_y = np.sqrt(mu[i + system.num_suff_stats // 2] - mu[i] ** 2)
        elif (system.behavior["type"]):
            mean_x = mu[j]
            mean_y = mu[i]
            mean_only = True
            std_x = None
            std_y = None
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    plot_ellipse(ax, mean_x, mean_y, std_x, std_y, "r", mean_only)


def plot_ellipse(ax, mean_x, mean_y, std_x, std_y, c, mean_only=False):
    t = np.arange(0, 1, 0.01)
    h = ax.plot(mean_x, mean_y, c=c, marker="+", ms=20)
    if (not mean_only):
        rx_t = std_x * np.cos(2 * np.pi * t) + mean_x
        ry_t = std_y * np.sin(2 * np.pi * t) + mean_y
        h = ax.plot(rx_t, ry_t, c)
    return h


def lin_reg_plot(x, y, xlabel="", ylabel="", pfname="images/temp.png", fontsize=30):
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.figure()
    plt.scatter(x, y)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    x_ax = np.arange(xmin, xmax, (xmax - xmin) / 95.0)
    y_lin = intercept + gradient * x_ax
    plt.plot(x_ax, y_lin, "-r")
    plt.text(
        xmin + 0.15 * (xmax - xmin),
        ymin + 0.95 * (ymax - ymin),
        "r = %.2f, p = %.2E" % (r_value, p_value),
        fontsize=(fontsize - 10),
    )
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(pfname)
    plt.show()


def dsn_pairplots(
    fnames,
    dist_str,
    system,
    D,
    f_str="identity",
    c_str=None,
    legendstrs=[],
    AL_final_its=[],
    xlims=None,
    ylims=None,
    ticks=None,
    fontsize=14,
    tri=True,
    ellipses=False,
    outlier_stds=2,
    pfnames=None,
    figsize=(10,10),
):
    n_fnames = len(fnames)

    # make sure D is greater than 1
    #if D < 2:
    #    print("Warning: D must be at least 2. Setting D = 2.")
    #    D = 2
        # If plotting ellipses, make sure D <= |T(x)|
    if (system.behavior["type"] in ["means", "pvar"]):
        if (ellipses and D > system.num_suff_stats):
            D = system.num_suff_stats
    else:
        if ellipses and D > system.num_suff_stats // 2:
            print("Warning: When plotting elipses, can only pairplot first moments.")
            print("Assuming T(x) = [first moments, second moments].")
            print("Setting D = |T(x)|/2.")
            D = system.num_suff_stats // 2

        # make all the legendstrs empty if no input
    if len(legendstrs) == 0:
        legendstrs = n_fnames * [""]
        # take the last aug lag iteration if haven't checked for convergence
    if len(AL_final_its) == 0:
        AL_final_its = n_fnames * [-1]

    figs = []
    dists = []
    for k in range(n_fnames):
        fname = fnames[k]
        AL_final_it = AL_final_its[k]
        if AL_final_it is None:
            print("%s has not converged so not plotting." % legendstrs[k])
            continue
        try:
            npzfile = np.load(fname)
        except:
            continue
        dist, dist_label_strs = dist_from_str(
            dist_str, f_str, system, npzfile, AL_final_it
        )
        dists.append(dist)
        if (D == 1):
            continue

        c, c_label_str, cm, vmin, vmax = coloring_from_str(c_str, system, npzfile, AL_final_it)
        plot_inds, below_inds, over_inds = filter_outliers(c, outlier_stds)
        if tri:
            fig, axs = plt.subplots(D - 1, D - 1, figsize=figsize)
            for i in range(D - 1):
                for j in range(1, D):
                    if (D == 2):
                        ax = plt.gca()
                    else:
                        ax = axs[i, j - 1]
                    if j > i:
                        ax.scatter(
                            dist[below_inds, j],
                            dist[below_inds, i],
                            c="w",
                            edgecolors="k",
                            linewidths=0.25,
                        )
                        ax.scatter(
                            dist[over_inds, j],
                            dist[over_inds, i],
                            c="k",
                            edgecolors="k",
                            linewidths=0.25,
                        )
                        h = ax.scatter(
                            dist[plot_inds, j],
                            dist[plot_inds, i],
                            c=c[plot_inds],
                            cmap=cm,
                            edgecolors="k",
                            linewidths=0.25,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        if ellipses:
                            plot_target_ellipse(ax, i, j, system, system.mu)
                            plot_var_ellipse(ax, dist[:, j], dist[:, i])
                        if i == j - 1:
                            ax.set_xlabel(dist_label_strs[j], fontsize=fontsize)
                            ax.set_ylabel(dist_label_strs[i], fontsize=fontsize)

                        if xlims is not None:
                            if dist_str == "T_xs":
                                xmin = system.mu[j] + xlims[0]
                                xmax = system.mu[j] + xlims[1]
                            else:
                                xmin = xlims[0]
                                xmax = xlims[1]
                            ax.set_xlim([xmin, xmax])
                            ax.plot([xmin, xmax], [0, 0], "--", c=[0.5, 0.5, 0.5])
                        if ylims is not None:
                            if dist_str == "T_xs":
                                ymin = system.mu[i] + ylims[0]
                                ymax = system.mu[i] + ylims[1]
                            else:
                                ymin = ylims[0]
                                ymax = ylims[1]
                            ax.set_ylim([ymin, ymax])
                            ax.plot([0, 0], [ymin, ymax], "--", c=[0.5, 0.5, 0.5])
                    else:
                        ax.axis("off")
        else:
            fig, axs = plt.subplots(D, D, figsize=figsize)
            for i in range(D):
                for j in range(D):
                    ax = axs[i, j]
                    ax.scatter(
                        dist[below_inds, j],
                        dist[below_inds, i],
                        c="w",
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    ax.scatter(
                        dist[over_inds, j],
                        dist[over_inds, i],
                        c="k",
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    h = ax.scatter(
                        dist[plot_inds, j],
                        dist[plot_inds, i],
                        c=c[plot_inds],
                        cmap=cm,
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    if ellipses:
                        plot_target_ellipse(ax, i, j, system, system.mu)
                        plot_var_ellipse(ax, dist[:, j], dist[:, i])
                    if i == (D - 1):
                        ax.set_xlabel(dist_label_strs[j], fontsize=fontsize)
                    if j == 0:
                        ax.set_ylabel(dist_label_strs[i], fontsize=fontsize)
                    if xlims is not None:
                        ax.set_xlim(xlims)
                    if ylims is not None:
                        ax.set_ylim(ylims)

                        # add the colorbar
        if c is not None:
            fig.subplots_adjust(right=0.90)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
            clb = fig.colorbar(h, cax=cbar_ax)
            a = (0.8 / (D - 1)) / (0.95 / (D - 1))
            b = (D - 1) * 1.15
            cbar_ax.text(
                a, b-.1, c_label_str, {"fontsize": fontsize + 2}, transform=ax.transAxes
            )
            # clb.ax.set_ylabel(c_label_str, rotation=270, fontsize=fontsize);

        plt.suptitle(legendstrs[k], fontsize=(fontsize + 4))
        if (pfnames is not None and len(pfnames) > (k)):
            plt.savefig(pfnames[k])
        figs.append(fig)
    plt.show()
    return dists


def pairplot(
    Z,
    dims,
    labels,
    origin=False,
    xlims=None,
    ylims=None,
    ticks=None,
    c=None,
    c_label=None,
    cmap=None,
    fontsize=12,
    figsize=(12, 12),
    pfname="images/temp.png",
):
    num_dims = len(dims)
    rand_order = np.random.permutation(Z.shape[0])
    Z = Z[rand_order, :]
    if c is not None:
        c = c[rand_order]
        plot_inds, below_inds, over_inds = filter_outliers(c, 2)

    fig, axs = plt.subplots(num_dims - 1, num_dims - 1, figsize=figsize)
    for i in range(num_dims - 1):
        dim_i = dims[i]
        for j in range(1, num_dims):
            if (num_dims == 2):
                ax = plt.gca()
            else:
                ax = axs[i, j - 1]
            if j > i:
                dim_j = dims[j]
                if (xlims is not None) and (ylims is not None) and origin:
                    ax.plot(xlims, [0, 0], c=0.5 * np.ones(3), linestyle="--")
                    ax.plot([0, 0], ylims, c=0.5 * np.ones(3), linestyle="--")
                if c is not None:
                    ax.scatter(
                        Z[below_inds, dim_j],
                        Z[below_inds, dim_i],
                        c="w",
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    ax.scatter(
                        Z[over_inds, dim_j],
                        Z[over_inds, dim_i],
                        c="k",
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    h = ax.scatter(
                        Z[plot_inds, dim_j],
                        Z[plot_inds, dim_i],
                        c=c[plot_inds],
                        cmap=cmap,
                        edgecolors="k",
                        linewidths=0.25,
                    )
                else:
                    h = ax.scatter(
                        Z[:, dim_j], Z[:, dim_i], edgecolors="k", linewidths=0.25, s=2
                    )
                if i + 1 == j:
                    ax.set_xlabel(labels[j], fontsize=fontsize)
                    ax.set_ylabel(labels[i], fontsize=fontsize)
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                if ticks is not None:
                    ax.set_xticks(ticks)
                    ax.set_yticks(ticks)

                if xlims is not None:
                    ax.set_xlim(xlims)
                if ylims is not None:
                    ax.set_ylim(ylims)
            else:
                ax.axis("off")

    if c is not None:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
        clb = fig.colorbar(h, cax=cbar_ax)
        a = (1.01 / (num_dims - 1)) / (0.9 / (num_dims - 1))
        b = (num_dims - 1) * 1.15
        plt.text(a, b, c_label, {"fontsize": fontsize}, transform=ax.transAxes)
    #plt.savefig(pfname)
    plt.show()
    return fig


def dsn_tSNE(
    fnames,
    dist_str,
    c_str,
    system,
    legendstrs=[],
    AL_final_its=[],
    fontsize=14,
    pfname="images/temp.png",
):
    n_fnames = len(fnames)

    # take the last aug lag iteration if haven't checked for convergence
    if len(AL_final_its) == 0:
        AL_final_its = n_fnames * [-1]

    figsize = (8, 8)
    figs = []
    for k in range(n_fnames):
        fname = fnames[k]
        AL_final_it = AL_final_its[k]
        npzfile = np.load(fname)
        dist, dist_label_strs = dist_from_str(
            dist_str, "tSNE", None, npzfile, AL_final_it
        )
        c, c_label_str, cm, _, _ = coloring_from_str(c_str, system, npzfile, AL_final_it)
        if AL_final_it is None:
            print("%s has not converged so not plotting." % legendstrs[k])
            continue
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        h = plt.scatter(
            dist[:, 0], dist[:, 1], c=c, cmap=cm, edgecolors="k", linewidths=0.25
        )

        plt.xlabel(dist_label_strs[0], fontsize=fontsize)
        plt.ylabel(dist_label_strs[1], fontsize=fontsize)

        # add the colorbar
        if c is not None:
            fig.subplots_adjust(right=0.90)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
            clb = fig.colorbar(h, cax=cbar_ax)
            plt.text(-0.2, 1.02 * np.max(c), c_label_str, {"fontsize": fontsize})
            # clb.ax.set_ylabel(c_label_str, rotation=270, fontsize=fontsize);
        plt.suptitle(legendstrs[k], fontsize=fontsize)
        plt.savefig(pfname)
        plt.show()
        figs.append(fig)
    return figs


def dsn_corrhists(fnames, dist_str, system, D, AL_final_its):
    rs, r2s, dist_label_strs = dsn_correlations(
        fnames, dist_str, system, D, AL_final_its
    )
    figs = []
    figs.append(pairhists(rs, dist_label_strs, "correlation hists"))
    figs.append(pairhists(r2s, dist_label_strs, r"$r^2$ hists"))
    return figs


def pairhists(x, dist_label_strs, title_str="", fontsize=16):
    D = x.shape[1]
    hist_ns = []
    fig, axs = plt.subplots(D, D, figsize=(12, 12))
    for i in range(D):
        for j in range(D):
            n, _, _ = axs[i][j].hist(x[:, j, i])
            if not (i == j):
                hist_ns.append(n)

    max_n = np.max(np.array(hist_ns))
    for i in range(D):
        for j in range(D):
            ax = axs[i][j]
            ax.set_xlim([-1, 1])
            ax.set_ylim([0, max_n])
            if i == (D - 1):
                ax.set_xlabel(dist_label_strs[j], fontsize=fontsize)
            if j == 0:
                ax.set_ylabel(dist_label_strs[i], fontsize=fontsize)
    plt.suptitle(title_str, fontsize=fontsize + 2)
    plt.show()
    return fig


def dsn_correlations(fnames, dist_str, system, D, AL_final_its):
    n_fnames = len(fnames)
    rs = np.zeros((n_fnames, D, D))
    r2s = np.zeros((n_fnames, D, D))
    for k in range(n_fnames):
        fname = fnames[k]
        AL_final_it = AL_final_its[k]
        if AL_final_it is None:
            rs[k, :, :] = np.nan
            r2s[k, :, :] = np.nan
            continue
        npzfile = np.load(fname)
        dist, dist_label_strs = dist_from_str(
            dist_str, "identity", system, npzfile, AL_final_it
        )
        for i in range(D):
            for j in range(D):
                ind = D * i + j + 1
                slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(
                    dist[:, j], dist[:, i]
                )
                rs[k, i, j] = r_value
                r2s[k, i, j] = r_value ** 2
    return rs, r2s, dist_label_strs


def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA

    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs


def make_training_movie(fname, system, step, save_fname='temp'):
    npzfile = np.load(fname)
    Hs = npzfile['Hs']
    Zs = npzfile['Zs']
    log_q_zs = npzfile['log_q_zs']
    Cs = npzfile['Cs']
    alphas = npzfile['alphas']
    check_rate = npzfile['check_rate']
    epoch_inds = npzfile['epoch_inds']

    cm = plt.get_cmap('tab20')
    scale = 100
    Cs = np.argmax(Cs, 2)
    def size_renorm(x, scale=30):
        y = (x - np.min(x))
        y = y / np.max(y)
        return scale*y
    
    color = [0.0, 0.3, 0.6]
    M = 100
    fontsize = 20

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    K = alphas.shape[1]
    N, _, D = Zs.shape
    Zs = np.transpose(Zs, [1, 0, 2])
    fig, axs = plt.subplots(D, D-1, figsize=(10,12))
    scats = []
    Cs = Cs.astype(float) / float(K)
    for i in range(D-1):
        for j in range(1, D):
            if (D==2):
                ax = plt.gca()
            else:
                ax = axs[i+1,j-1]
            if (j > i):
                s = size_renorm(log_q_zs[i,:M], scale)
                scats.append(ax.scatter(Zs[:M,0,j], Zs[:M,0,i], 
                                        s=s, c=cm(Cs[0,:M]), 
                                        edgecolors="k",
                                        linewidths=0.25,))
                scats[-1].set_cmap(cm)
                ax.set_xlim([-10, 10])
                ax.set_ylim([-10, 10])
            elif ((i==(D-2)) and j==1):
                pass
            else:
                ax.axis('off')
            if i == j - 1:
                ax.set_xlabel(system.z_labels[j], fontsize=fontsize)
                ax.set_ylabel(system.z_labels[i], fontsize=fontsize)

    if (K > 1):
        bar_ax = axs[-1,0]
        rect_colors = np.arange(K)/float(K)
        rects = bar_ax.bar(np.arange(1, K+1), alphas[0], color=cm(rect_colors))
        
        bar_ax.set_ylim([0, 3.0/K])
        bar_ax.set_xlabel('k')
        bar_ax.set_ylabel(r'$\alpha_k$')
        bar_ax.spines['right'].set_visible(False)
        bar_ax.spines['top'].set_visible(False)

    # plot entropy
    alpha = 0.05
    frac_samps = 0.8
    n_suff_stats = system.num_suff_stats
    pvals, AL_final_its = assess_constraints([fname], alpha, frac_samps, n_suff_stats)
    iterations = np.arange(0, check_rate * N, check_rate)
    H_ax = plt.subplot(D+1, 1, 1)
    lines = H_ax.plot(iterations, Hs, lw=1, c=color)
    H_ax.spines["right"].set_visible(False)
    H_ax.spines["top"].set_visible(False)
    H_ax.set_xlabel('iterations', fontsize=fontsize)
    H_ax.set_ylabel('entropy (H)', fontsize=fontsize)
    if AL_final_its[0] is not None:
        conv_it = iterations[AL_final_its[0]]
        H_ax.plot(
            [conv_it, conv_it],
            [np.min(Hs), np.max(Hs)],
            "k--",
        )
    pt = H_ax.plot(iterations[0], Hs[0], 'o', c=color, markersize=15)
    
    

    def animate(i):
        # we'll step two time-steps per frame.  This leads to nice results.
        i = (step * i) % N
        ind = 0
        for ii in range(D-1):
            for j in range(1, D):
                if (j > ii):
                    s = size_renorm(log_q_zs[i,:M], scale)
                    scat = scats[ind]
                    scat.set_offsets(np.stack((Zs[:M,i,j],Zs[:M,i,ii]), 1))
                    scat.set_color(cm(Cs[i,:M]))
                    scat.set_sizes(s)
                    ind += 1
                    
        AL_it = np.sum(epoch_inds < i*check_rate)
        H_ax.set_title('AL=%d' % AL_it)
        if (K > 1):
            j = 0
            for rect in rects:
                rect.set_height(alphas[i,j])
                j += 1

        pt[0].set_data(iterations[i], Hs[i])

        fig.canvas.draw()
        return lines + scats

    # instantiate the animator.
    frames = ((N-1)//step)
    anim = animation.FuncAnimation(fig, animate,
                                   frames=frames, interval=30, blit=True)

    print('Making video.')
    start_time = time.time()
    anim.save('%s.mp4' % save_fname, writer=writer)
    end_time = time.time()
    print('Video complete after %.3f seconds.' % (end_time - start_time))
    return None

