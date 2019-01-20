import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.manifold import TSNE

def assess_constraints(fnames, alpha, k_max, n_suff_stats):
	NUM_SAMPS = 200
	n_fnames = len(fnames);
	p_values = np.zeros((n_fnames, k_max+1, n_suff_stats));
	for i in range(n_fnames):
		fname = fnames[i];
		npzfile = np.load(fname);
		mu = npzfile['mu']

		for k in range(k_max+1):
			T_xs = npzfile['T_xs'][k];
			for j in range(n_suff_stats):
				t, p = scipy.stats.ttest_1samp(T_xs[:NUM_SAMPS,j], mu[j]);
				p_values[i,k,j] = p;

	AL_final_its = [];
	for i in range(n_fnames):
		for j in range(k_max+1):
			con_sat = np.prod(p_values[i,j,:] > (alpha / n_suff_stats));
			if (con_sat==1):
				AL_final_its.append(j);
				break;
			if (j==(k_max)):
				AL_final_its.append(None)
	return p_values, AL_final_its;

def plot_opt(fnames, legendstrs=[], alpha=0.05, plotR2=False, fontsize=14):
	max_legendstrs = 5;
	n_fnames = len(fnames);
	# read optimization diagnostics from files
	costs_list = [];
	Hs_list = [];
	R2s_list = [];
	mean_T_xs_list = [];
	T_xs_list = [];
	epoch_inds_list = []
	for i in range(n_fnames):
		fname = fnames[i];
		npzfile = np.load(fname);
		costs = npzfile['costs'];
		Hs = npzfile['Hs'];
		R2s = npzfile['R2s'];
		mean_T_xs = npzfile['mean_T_xs'];
		T_xs = npzfile['T_xs'];
		epoch_inds = npzfile['epoch_inds']

		costs_list.append(costs);
		Hs_list.append(Hs);
		R2s_list.append(R2s);
		mean_T_xs_list.append(mean_T_xs);
		epoch_inds_list.append(epoch_inds)

		if (i==0):
			mu = npzfile['mu'];
			check_rate = npzfile['check_rate'];
			last_ind = npzfile['it']//check_rate;
			nits = costs.shape[0];
			k_max = npzfile['T_xs'].shape[0]-1;
			iterations = np.arange(0, check_rate*nits, check_rate);
			n_suff_stats = mean_T_xs_list[0].shape[1];
			p_values, AL_final_its = assess_constraints(fnames, alpha, k_max, n_suff_stats);

	figs = [];

	# plot cost, entropy and r^2
	num_panels = 3 if plotR2 else 2;
	figsize = (num_panels*4, 4);
	fig, axs = plt.subplots(1, num_panels, figsize=figsize);
	figs.append(fig);
	ax = axs[0];
	for i in range(n_fnames):
		costs = costs_list[i];
		ax.plot(iterations[:last_ind], costs[:last_ind], label=legendstrs[i]);
	ax.set_xlabel('iterations', fontsize=fontsize);
	ax.set_ylabel('cost', fontsize=fontsize);
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	ax = axs[1];
	for i in range(n_fnames):
		Hs = Hs_list[i];
		epoch_inds = epoch_inds_list[i]
		if (i < 5):
			ax.plot(iterations[:last_ind], Hs[:last_ind], label=legendstrs[i]);
		else:
			ax.plot(iterations[:last_ind], Hs[:last_ind])
		if (n_fnames == 1 and AL_final_its[i] is not None):
			conv_it = epoch_inds[AL_final_its[i]]
			ax.plot([conv_it, conv_it], [np.min(Hs[:last_ind]), np.max(Hs[:last_ind])], 'k--')
	ax.set_xlabel('iterations', fontsize=fontsize);
	ax.set_ylabel('H', fontsize=fontsize);
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	if (plotR2):
		ax = axs[2]
		for i in range(n_fnames):
			R2s = R2s_list[i];
			epoch_inds = epoch_inds_list[i]
			if (i < max_legendstrs):
				ax.plot(iterations[:last_ind], R2s[:last_ind], label=legendstrs[i]);
			else:
				ax.plot(iterations[:last_ind], R2s[:last_ind])
		if (n_fnames == 1 and AL_final_its[i] is not None):
			conv_it = epoch_inds[AL_final_its[i]]
			ax.plot([conv_it, conv_it], [np.min(R2s[:last_ind]), np.max(R2s[:last_ind])], 'k--')
		ax.set_xlabel('iterations', fontsize=fontsize);
		ax.set_ylabel(r'$r^2$', fontsize=fontsize);
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

	ax.legend(fontsize=fontsize);
	plt.tight_layout();
	plt.show();


	# plot constraints throughout optimization
	yscale_fac = 5
	n_cols = 4
	n_rows = int(np.ceil(n_suff_stats/n_cols))
	figsize = (n_cols*3, n_rows*3)
	fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
	if (n_rows == 1):
		axs = [axs]
	figs.append(fig)
	for i in range(n_suff_stats):
		ax = axs[i//n_cols][i % n_cols]
		# make ylim 2* mean abs error of last 50% of optimization
		mean_abs_errors = np.zeros((n_fnames,))
		for j in range(n_fnames):
			mean_T_xs = mean_T_xs_list[j]
			epoch_inds = epoch_inds_list[j]
			if (j < max_legendstrs):
				ax.plot(iterations[:last_ind], mean_T_xs[:last_ind,i], label=legendstrs[j])
			else:
				ax.plot(iterations[:last_ind], mean_T_xs[:last_ind,i])
			mean_abs_errors[j] = np.mean(np.abs(mean_T_xs[(last_ind//2):last_ind, i] - mu[i]))
			if (n_fnames == 1):
				T_x_means = np.mean(T_xs[:,:,i], axis=1)
				T_x_stds = np.std(T_xs[:,:,i], axis=1)
				ax.errorbar(epoch_inds, T_x_means, T_x_stds, c='r', elinewidth=3)
				if (AL_final_its[j] is not None):
					conv_it = epoch_inds[AL_final_its[j]]
					line_min = min([np.min(mean_T_xs[:last_ind,i]), mu[i]-yscale_fac*mean_abs_errors[j], np.min(T_x_means - 4*T_x_stds)])
					line_max = max([np.max(mean_T_xs[:last_ind,i]), mu[i]+yscale_fac*mean_abs_errors[j], np.max(T_x_means + 4*T_x_stds)])
					ax.plot([conv_it, conv_it], [line_min, line_max], 'k--')
			
		ax.plot([iterations[0], iterations[last_ind]], [mu[i], mu[i]], 'k-')
		# make ylim 2* mean abs error of last 50% of optimization
		if (n_fnames == 1):
			ymin = min(mu[i]-yscale_fac*np.max(mean_abs_errors), np.min(T_x_means[(k_max//2):] - 2*T_x_stds[(k_max//2):]))
			ymax = max(mu[i]-yscale_fac*np.max(mean_abs_errors), np.max(T_x_means[(k_max//2):] + 2*T_x_stds[(k_max//2):]))
		else:
			ymin = mu[i]-yscale_fac*np.max(mean_abs_errors)
			ymax = mu[i]+yscale_fac*np.max(mean_abs_errors)
		ax.set_ylim(ymin, ymax)
		ax.set_ylabel(r"$E[T_%d(z)]$" % (i+1), fontsize=fontsize)
		if (i==(n_cols-1)):
			ax.legend(fontsize=fontsize)
		if (i > n_suff_stats - n_cols - 1):
			ax.set_xlabel('iterations', fontsize=fontsize)

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
	plt.tight_layout();
	plt.show();

	# plot the p-value based constraint satisfaction 
	n_cols = 4;
	n_rows = int(np.ceil(n_fnames/n_cols));
	figsize = (n_cols*3, n_rows*3);
	fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize);
	figs.append(fig);
	for i in range(n_fnames):
		ax = plt.subplot(n_rows, n_cols, i+1);
		for j in range(n_suff_stats):
			ax.plot(np.arange(k_max+1), p_values[i,:,j], label=r'$T_%d(z)$' % (j+1));
		if (AL_final_its[i] is not None):
			ax.plot([AL_final_its[i], AL_final_its[i]], [0,1], 'k--', label='convergence');
			ax.set_title(legendstrs[i], fontsize=fontsize);
		else:
			ax.set_title(legendstrs[i] + ' no converge', fontsize=fontsize);
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_xlabel('aug Lag it', fontsize=fontsize);
		ax.set_ylabel('p value', fontsize=fontsize);
		ax.set_ylim([0,1]);
		ax.legend(fontsize=fontsize);
	plt.tight_layout();
	plt.show();

	return figs


def coloring_from_str(c_str, system, npzfile, AL_final_it):
	cm = plt.cm.get_cmap('viridis');
	if (c_str == 'log_q_z'):
		c = npzfile['log_q_zs'][AL_final_it];
		c_label_str = r'$log(q_\theta)$';
	elif (c_str == 'real part'):
		c = npzfile['T_xs'][AL_final_it, :, 0];
		cm = plt.cm.get_cmap('Reds')
		c_label_str = r'real($\lambda_1$)'
	elif (c_str == 'dE'):
		c = npzfile['T_xs'][AL_final_it, :, 0];
		cm = plt.cm.get_cmap('Greys')
		c_label_str = r'$d_{E,ss}$'
	elif (c_str == 'dP'):
		c = npzfile['T_xs'][AL_final_it, :, 1];
		cm = plt.cm.get_cmap('Blues')
		c_label_str = r'$d_{P,ss}$'
	elif (c_str == 'dS'):
		c = npzfile['T_xs'][AL_final_it, :, 2];
		cm = plt.cm.get_cmap('Reds')
		c_label_str = r'$d_{S,ss}$'
	elif (c_str == 'dV'):
		c = npzfile['T_xs'][AL_final_it, :, 3];
		cm = plt.cm.get_cmap('Greens')
		c_label_str = r'$d_{V,ss}$'

	else:
		# no coloring
		c = np.ones((npzfile['T_xs'].shape[1],))
		c_label_str = '';

	return c, c_label_str, cm;

def dist_from_str(dist_str, f_str, system, npzfile, AL_final_it):
	dist_label_strs = [];
	if (dist_str in ['Zs', 'T_xs']):
		dist = npzfile[dist_str][AL_final_it, :, :];
		if (f_str == 'identity'):
			if (dist_str == 'Zs'):
				dist_label_strs = system.z_labels;
			elif (dist_str == 'T_xs'):
				dist_label_strs = system.T_x_labels;
		elif (f_str == 'PCA'):
			dist, evecs, evals = PCA(dist, dist.shape[1]);
			dist_label_strs = ['PC%d' %i for i in range(1, system.D+1)];
		elif (f_str == 'tSNE'):
			np.random.seed(0);
			dist = TSNE(n_components=2).fit_transform(dist);
			dist_label_strs = ['tSNE 1', 'tSNE 2'];
	else:
		raise NotImplementedError();
	return dist, dist_label_strs;

def filter_outliers(c, num_stds=4):
	c_mean = np.mean(c);
	c_std = np.std(c);
	all_inds = np.arange(c.shape[0]);
	below_inds = all_inds[c < c_mean - num_stds*c_std];
	over_inds = all_inds[c > c_mean + num_stds*c_std];
	plot_inds = all_inds[np.logical_and(c_mean - num_stds*c_std <= c, c <= c_mean + num_stds*c_std)];
	return plot_inds, below_inds, over_inds;

def plot_var_ellipse(ax, x,y):
	mean_x = np.mean(x);
	mean_y = np.mean(y);
	std_x = np.std(x);
	std_y = np.std(y);
	h = plot_ellipse(ax, mean_x, mean_y, std_x, std_y, 'k')
	return h;

def plot_target_ellipse(ax, i, j, system, mu):
	if (system.name == 'linear_2D'):
		if (system.behavior['type'] == 'oscillation'):
			mean_x = mu[j];
			mean_y = mu[i];
			std_x = np.sqrt(mu[j+system.num_suff_stats//2] - mu[j]**2);
			std_y = np.sqrt(mu[i+system.num_suff_stats//2] - mu[i]**2);
	elif (system.name == 'V1_circuit'):
		if (system.behavior['type'] == 'difference'):
			mean_x = mu[j];
			mean_y = mu[i];
			std_x = np.sqrt(mu[j+system.num_suff_stats//2] - mu[j]**2);
			std_y = np.sqrt(mu[i+system.num_suff_stats//2] - mu[i]**2);
		else:
			raise NotImplementedError();
	else:
		raise NotImplementedError();
	plot_ellipse(ax, mean_x, mean_y, std_x, std_y, 'r');

def plot_ellipse(ax, mean_x, mean_y, std_x, std_y, c):
	t = np.arange(0,1,0.01);
	ax.plot(mean_x, mean_y, c=c, marker='+', ms=20);
	rx_t = std_x*np.cos(2*np.pi*t)+mean_x;
	ry_t = std_y*np.sin(2*np.pi*t)+mean_y;
	h = ax.plot(rx_t, ry_t, c);
	return h;

def dsn_pairplots(fnames, dist_str, system, D, f_str='identity', \
	              c_str=None, legendstrs=[], AL_final_its=[], \
	              fontsize=14, ellipses=False, \
	              pfname='temp.png'):
	n_fnames = len(fnames);

	# make sure D is greater than 1
	if (D < 2):
		print('Warning: D must be at least 2. Setting D = 2.')
		D = 2
	# If plotting ellipses, make sure D <= |T(x)|
	if (ellipses and D > system.num_suff_stats//2):
		print('Warning: When plotting elipses, can only pairplot first moments.')
		print('Assuming T(x) = [first moments, second moments].')
		print('Setting D = |T(x)|/2.')
		D = system.num_suff_stats//2

	# make all the legendstrs empty if no input
	if (len(legendstrs)==0):
		legendstrs = n_fnames*[''];
	# take the last aug lag iteration if haven't checked for convergence
	if (len(AL_final_its)==0):
		AL_final_its = n_fnames*[-1];

	figsize = (12,12);
	figs = [];
	for k in range(n_fnames):
		fname = fnames[k];
		AL_final_it = AL_final_its[k];
		if (AL_final_it is None):
			print('%s has not converged so not plotting.' % legendstrs[k]);
			continue;
		npzfile = np.load(fname);
		dist, dist_label_strs = dist_from_str(dist_str, f_str, system, npzfile, AL_final_it);
		c, c_label_str, cm = coloring_from_str(c_str, system, npzfile, AL_final_it);
		plot_inds, below_inds, over_inds = filter_outliers(c, 2);
		fig, axs = plt.subplots(D, D, figsize=figsize);
		for i in range(D):
			for j in range(D):
				ind = D*i+j+1;
				ax = axs[i,j];
				ax.scatter(dist[below_inds,j], dist[below_inds,i], c='w', \
					        edgecolors='k', linewidths=0.25);
				ax.scatter(dist[over_inds,j], dist[over_inds,i], c='k', \
					        edgecolors='k', linewidths=0.25);
				h = ax.scatter(dist[plot_inds,j], dist[plot_inds,i], c=c[plot_inds], cmap=cm, \
					        edgecolors='k', linewidths=0.25);
				if (ellipses):
					plot_target_ellipse(ax, i, j, system, system.mu);
					plot_var_ellipse(ax, dist[:,j], dist[:,i]);
				if (i==(D-1)):
					ax.set_xlabel(dist_label_strs[j], fontsize=fontsize);
				if (j==0):
					ax.set_ylabel(dist_label_strs[i], fontsize=fontsize);

		# add the colorbar
		if (c is not None):
			fig.subplots_adjust(right=0.90)
			cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
			clb = fig.colorbar(h, cax=cbar_ax);
			plt.text(-.2, 1.02*np.max(c[plot_inds]), c_label_str, {'fontsize':fontsize})
			#clb.ax.set_ylabel(c_label_str, rotation=270, fontsize=fontsize);
		plt.suptitle(legendstrs[k], fontsize=fontsize);
		plt.savefig(pfname);
		plt.show();
		figs.append(fig);

	return figs


def pairplot(Z, dims, labels, origin=False, xlims=None, ylims=None, \
	         c=None, c_label=None, cmap=None, \
	         fontsize=12, figsize=(12,12), fname='temp.png'):
	num_dims = len(dims)
	rand_order = np.random.permutation(Z.shape[0])
	Z = Z[rand_order, :]

	if (c is not None):
		c = c[rand_order]
		plot_inds, below_inds, over_inds = filter_outliers(c, 2);

	fig, axs = plt.subplots(num_dims-1, num_dims-1, figsize=figsize);
	for i in range(num_dims-1):
		dim_i = dims[i]
		for j in range(1, num_dims):
			ax = axs[i, j-1]
			if (j > i):
				dim_j = dims[j]
				if ((xlims is not None) and (ylims is not None) and origin):
					ax.plot(xlims, [0, 0], c=0.5*np.ones(3), linestyle='--')
					ax.plot([0, 0], ylims, c=0.5*np.ones(3), linestyle='--')
				if (c is not None):
					ax.scatter(Z[below_inds,dim_j], Z[below_inds, dim_i], c='w', \
						       edgecolors='k', linewidths=0.25);
					ax.scatter(Z[over_inds,dim_j], Z[over_inds, dim_i], c='k', \
						       edgecolors='k', linewidths=0.25);
					h = ax.scatter(Z[plot_inds,dim_j], Z[plot_inds, dim_i], c=c[plot_inds], cmap=cmap, \
						           edgecolors='k', linewidths=0.25);
				else:
					h = ax.scatter(Z[:,dim_j], Z[:, dim_i], \
						           edgecolors='k', linewidths=0.25);
				if (i+1==j):
					ax.set_xlabel(labels[j], fontsize=fontsize);
					ax.set_ylabel(labels[i], fontsize=fontsize);
				if (xlims is not None):
					ax.set_xlim(xlims)
				if (ylims is not None):
					ax.set_ylim(ylims)
			else:
				ax.axis('off')

	if (c is not None):
		fig.subplots_adjust(right=0.90)
		cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
		clb = fig.colorbar(h, cax=cbar_ax);
		plt.text(-.2, 1.02*np.max(c[plot_inds]), c_label, {'fontsize':fontsize})
	#plt.savefig(fname)
	plt.show()
	return fig



def dsn_tSNE(fnames, dist_str, c_str, system, legendstrs=[], AL_final_its=[], \
	              fontsize=14, pfname='temp.png'):
	n_fnames = len(fnames);

	# take the last aug lag iteration if haven't checked for convergence
	if (len(AL_final_its)==0):
		AL_final_its = n_fnames*[-1];

	figsize = (8,8);
	figs = [];
	for k in range(n_fnames):
		fname = fnames[k];
		AL_final_it = AL_final_its[k];
		npzfile = np.load(fname);
		dist, dist_label_strs = dist_from_str(dist_str, 'tSNE', None, npzfile, AL_final_it);
		c, c_label_str, cm = coloring_from_str(c_str, system, npzfile, AL_final_it);
		if (AL_final_it is None):
			print('%s has not converged so not plotting.' % legendstrs[k]);
			continue;
		fig = plt.figure(figsize=figsize);
		ax = plt.subplot(111);
		h = plt.scatter(dist[:,0], dist[:,1], c=c, cmap=cm, \
					edgecolors='k', linewidths=0.25);

		plt.xlabel(dist_label_strs[0], fontsize=fontsize);
		plt.ylabel(dist_label_strs[1], fontsize=fontsize);

		# add the colorbar
		if (c is not None):
			fig.subplots_adjust(right=0.90)
			cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
			clb = fig.colorbar(h, cax=cbar_ax);
			plt.text(-.2, 1.02*np.max(c), c_label_str, {'fontsize':fontsize})
			#clb.ax.set_ylabel(c_label_str, rotation=270, fontsize=fontsize);
		plt.suptitle(legendstrs[k], fontsize=fontsize);
		plt.savefig(pfname);
		plt.show();
		figs.append(fig);
	return figs

def dsn_corrhists(fnames, dist_str, system, D, AL_final_its):
	rs, r2s, dist_label_strs = dsn_correlations(fnames, dist_str, system, D, AL_final_its);
	figs = [];
	figs.append(pairhists(rs, dist_label_strs, 'correlation hists'));
	figs.append(pairhists(r2s, dist_label_strs, r'$r^2$ hists'));
	return figs;

def pairhists(x, dist_label_strs, title_str='', fontsize=16):
	D = x.shape[1];
	hist_ns = [];
	fig, axs = plt.subplots(D, D, figsize=(12,12));
	for i in range(D):
		for j in range(D):
			n, _, _ = axs[i][j].hist(x[:,j,i]);
			if (not (i==j)):
				hist_ns.append(n);


	max_n = np.max(np.array(hist_ns));
	for i in range(D):
		for j in range(D):
			ax = axs[i][j];
			ax.set_xlim([-1, 1]);
			ax.set_ylim([0, max_n]);
			if (i==(D-1)):
				ax.set_xlabel(dist_label_strs[j], fontsize=fontsize);
			if (j==0):
				ax.set_ylabel(dist_label_strs[i], fontsize=fontsize);
	plt.suptitle(title_str, fontsize=fontsize+2)
	plt.show();
	return fig;


def dsn_correlations(fnames, dist_str, system, D, AL_final_its):
    n_fnames = len(fnames);
    rs = np.zeros((n_fnames, D, D));
    r2s = np.zeros((n_fnames, D, D));
    for k in range(n_fnames):
        fname = fnames[k];
        AL_final_it = AL_final_its[k];
        if (AL_final_it is None):
            rs[k,:,:] = np.nan;
            r2s[k,:,:] = np.nan;
            continue;
        npzfile = np.load(fname);
        dist, dist_label_strs = dist_from_str(dist_str, 'identity', system, npzfile, AL_final_it);
        for i in range(D):
            for j in range(D):
                ind = D*i+j+1;
                slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(dist[:,j], dist[:,i]);
                rs[k,i,j] = r_value;
                r2s[k,i,j] = r_value**2;
    return rs, r2s, dist_label_strs;


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
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs








