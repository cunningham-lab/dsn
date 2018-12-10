import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def assess_constraints(fnames, alpha, k_max, mu, n_suff_stats):
	n_fnames = len(fnames);
	p_values = np.zeros((n_fnames, k_max+1, n_suff_stats));
	for i in range(n_fnames):
		fname = fnames[i];
		npzfile = np.load(fname);

		for k in range(k_max+1):
			T_phis = npzfile['T_phis'][k];
			for j in range(n_suff_stats):
				t, p = scipy.stats.ttest_1samp(T_phis[:100,j], mu[j]);
				p_values[i,k,j] = p;

	AL_final_its = [];
	for i in range(n_fnames):
		for j in range(k_max+1):
			con_sat = np.prod(p_values[i,j,:] > alpha);
			if (con_sat==1):
				AL_final_its.append(j);
				break;
			if (j==(k_max)):
				AL_final_its.append(None)
	return p_values, AL_final_its;

def plot_opt(fnames, legendstrs=[], alpha=0.05, plotR2=False, fontsize=14):
	n_fnames = len(fnames);
	# read optimization diagnostics from files
	costs_list = [];
	Hs_list = [];
	R2s_list = [];
	mean_T_phis_list = [];
	T_phis_list = [];
	for i in range(n_fnames):
		fname = fnames[i];
		npzfile = np.load(fname);
		costs = npzfile['costs'];
		Hs = npzfile['Hs'];
		R2s = npzfile['R2s'];
		mean_T_phis = npzfile['mean_T_phis'];
		T_phis = npzfile['T_phis'];

		costs_list.append(costs);
		Hs_list.append(Hs);
		R2s_list.append(R2s);
		mean_T_phis_list.append(mean_T_phis);

		if (i==0):
			mu = npzfile['mu'];
			check_rate = npzfile['check_rate'];
			last_ind = npzfile['it']//check_rate;
			nits = costs.shape[0];
			k_max = npzfile['T_phis'].shape[0]-1;
			iterations = np.arange(0, check_rate*nits, check_rate);
			n_suff_stats = mean_T_phis_list[0].shape[1];
			p_values, AL_final_its = assess_constraints(fnames, alpha, k_max, mu, n_suff_stats);



	# plot cost, entropy and r^2
	num_panels = 3 if plotR2 else 2;
	figsize = (num_panels*4, 4);
	fig1 = plt.figure(figsize=figsize);
	plt.subplot(1,num_panels,1);
	for i in range(n_fnames):
		costs = costs_list[i];
		plt.plot(iterations[:last_ind], costs[:last_ind], label=legendstrs[i]);
	plt.xlabel('iterations', fontsize=fontsize);
	plt.ylabel('cost', fontsize=fontsize);

	plt.subplot(1,num_panels,2);
	for i in range(n_fnames):
		Hs = Hs_list[i];
		plt.plot(iterations[:last_ind], Hs[:last_ind], label=legendstrs[i]);
	plt.xlabel('iterations', fontsize=fontsize);
	plt.ylabel('H', fontsize=fontsize);

	if (plotR2):
		plt.subplot(1,num_panels,3);
		for i in range(n_fnames):
			R2s = R2s_list[i];
			plt.plot(iterations[:last_ind], R2s[:last_ind], label=legendstrs[i]);
		plt.xlabel('iterations', fontsize=fontsize);
		plt.ylabel(r'$r^2$', fontsize=fontsize);

	plt.legend(fontsize=fontsize);
	plt.tight_layout();
	plt.show();


	# plot constraints throughout optimization
	n_cols = 4;
	n_rows = int(np.ceil(n_suff_stats/n_cols));
	figsize = (n_cols*3, n_rows*3);
	fig2 = plt.figure(figsize=figsize);
	for i in range(n_suff_stats):
		plt.subplot(n_rows,n_cols,i+1);
		plt.plot([iterations[0], iterations[last_ind]], [mu[i], mu[i]], 'k--');
		for j in range(n_fnames):
			mean_T_phis = mean_T_phis_list[j];
			plt.plot(iterations[:last_ind], mean_T_phis[:last_ind,i], label=legendstrs[j]);
		plt.ylabel(r"$E[T_%d(z)]$" % (i+1), fontsize=fontsize);
		if (i==(n_cols-1)):
			plt.legend(fontsize=fontsize);
		if (i > n_suff_stats - n_cols - 1):
			plt.xlabel('iterations', fontsize=fontsize);
	plt.tight_layout();
	plt.show();

	# plot the p-value based constraint satisfaction 
	n_cols = 4;
	n_rows = int(np.ceil(n_fnames/n_cols));
	figsize = (n_cols*3, n_rows*3);
	plt.figure(figsize=figsize);
	for i in range(n_fnames):
		plt.subplot(n_rows, n_cols, i+1);
		for j in range(n_suff_stats):
			plt.plot(np.arange(k_max+1), p_values[i,:,j], label=r'$T_%d(z)$' % (j+1));
		if (AL_final_its[i] is not None):
			plt.plot([AL_final_its[i], AL_final_its[i]], [0,1], 'k--', label='convergence');
			plt.title(legendstrs[i], fontsize=fontsize);
		else:
			plt.title(legendstrs[i] + ' no converge', fontsize=fontsize);
		plt.ylabel('p value', fontsize=fontsize);
		plt.xlabel('aug Lag it', fontsize=fontsize);
		plt.ylim([0,1]);
	plt.tight_layout();
	plt.legend();
	plt.show();

	return fig1, fig2;

def plot_phis(fnames, D, labels=[], legendstrs=[], AL_final_its=[],fontsize=14):
	n_fnames = len(fnames);
	if (len(AL_final_its)==0):
		AL_final_its = n_fnames*[-1];
	if (len(labels)==[]):
		labels = D*[''];

	figsize = (12,12);
	figs = [];
	for k in range(n_fnames):
		fname = fnames[k];
		npzfile = np.load(fname);
		phis = npzfile['phis'];
		log_q_phis = npzfile['log_q_phis'];
		AL_final_it = AL_final_its[k];
		if (AL_final_it is None):
			print('%s has not converged so not plotting.' % legendstrs[k]);
			continue;
		fig = plt.figure(figsize=figsize);
		for i in range(D):
			for j in range(D):
				ind = D*i+j+1;
				plt.subplot(D, D, ind);
				plt.scatter(phis[AL_final_it,:,j], phis[AL_final_it,:,i], c=log_q_phis[AL_final_it]);
				if (i==(D-1)):
					plt.xlabel(labels[j], fontsize=fontsize);
				if (j==0):
					plt.ylabel(labels[i], fontsize=fontsize);
		plt.title(legendstrs[k], fontsize=fontsize);
		plt.show();
		figs.append(fig);
	return figs


def plot_T_phis(fnames, n_suff_stats, labels=[], legendstrs=[], AL_final_its=[],fontsize=14):
	n_fnames = len(fnames);
	if (len(AL_final_its)==0):
		AL_final_its = n_fnames*[-1];
	if (len(labels)==[]):
		labels = D*[''];

	figsize = (12,12);
	figs = [];
	for k in range(n_fnames):
		fname = fnames[k];
		npzfile = np.load(fname);
		T_phis = npzfile['T_phis'];
		log_q_phis = npzfile['log_q_phis'];
		AL_final_it = AL_final_its[k];
		if (AL_final_it is None):
			print('%s has not converged so not plotting.' % legendstrs[k]);
			continue;
		fig = plt.figure(figsize=figsize);
		for i in range(n_suff_stats):
			for j in range(n_suff_stats):
				ind = n_suff_stats*i+j+1;
				plt.subplot(n_suff_stats, n_suff_stats, ind);
				plt.scatter(T_phis[AL_final_it,:,j], T_phis[AL_final_it,:,i], c=log_q_phis[AL_final_it]);
				if (i==(n_suff_stats-1)):
					plt.xlabel(labels[j], fontsize=fontsize);
				if (j==0):
					plt.ylabel(labels[i], fontsize=fontsize);
		plt.title(legendstrs[k], fontsize=fontsize);
		plt.show();
		figs.append(fig);
	return figs












