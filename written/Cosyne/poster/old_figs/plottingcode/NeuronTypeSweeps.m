%% Code for figure 1

close all
clear all
%addpath helpers
width = 4*3/4;     % Width in inches
height = 3*3/4;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 0.5;      % LineWidth
msz = 4;       % MarkerSize
setdefaultplot % set the plot parameters
errorbarmult = 4;
grey = [0.5 0.5 0.5];
axisfnt = 14;
axislabelfnt = 8;
figurelabelfnt = 18;
xaxislimit = 101;
lblx = 5;
lbly = 85;
figurelabel2fnt = 14;
lblx2 = 65;
lbly2 = 8;

nneurons = 100;
trial_counts = [1200];
ntrial_counts = length(trial_counts);
binsize = 1000;

for i=1:ntrial_counts;
    ntrials = trial_counts(i);
    %% Load neuron type sweep
    NTSdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/results/NeuronTypeSweep/';
    NeSweepClustered = load([NTSdir, sprintf('%dN_%dT_clustered_NeuronTypeSweep.mat', nneurons, ntrials)]);
    NeSweepNonclust = load([NTSdir, sprintf('%dN_%dT_nonclust_NeuronTypeSweep.mat', nneurons, ntrials)]);

    [NeSweepClustered, maxind_cl] = filterNotYetComputed(NeSweepClustered, 'NeuronTypeSweep');
    [NeSweepNonclust, maxind_non] = filterNotYetComputed(NeSweepNonclust, 'NeuronTypeSweep');

    %% Plot A figs
    % plot Fig 3 A: Dimensionality
    figure % dimensionality
    inds = 0:10:100;
    plot(inds,mean(NeSweepClustered.dimensionality), 'k--');
    hold on
    errbar_rw(gca,std(NeSweepClustered.dimensionality)/sqrt(maxind_cl), 'k');
    xlim([0 101])

    plot(inds,mean(NeSweepNonclust.dimensionality), 'k-');
    errbar_rw(gca,std(NeSweepNonclust.dimensionality)/sqrt(maxind_non),'k');

    xlim([0 101])
    %ylabel('Dimensionality','FontSize',axisfnt)
    if (ntrials == 20000)
        xlabel('E-to-I neuron ratio','FontSize',axisfnt)
    end
    set(gca,'FontSize',axislabelfnt)
    ylim([0 100])
    set(gca,'TickDir','out')
    set(gca,'XTickLabel', {'All I', '20:80', '40:60', '60:40', '80:20', 'All E'});
    text(lblx2, lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
    %text(lblx, lbly, 'A   Dimensionality', 'FontSize', figurelabelfnt);
    print(sprintf('fig3a_%dms_%dN_%dT', binsize, nneurons, ntrials), '-depsc', '-r300');

%     %% Plot Fig 3 B: excitatory percent shared variance
%     figure % percent shared
%     inds = 0:10:100;
%     plot(inds, 100*mean(NeSweepClustered.percentSV_EallTypeAvg),'r--');
%     hold on
%     errbar_rw(gca, 100*std(NeSweepClustered.percentSV_EallTypeAvg)/sqrt(maxind_cl),'r');
% 
%     plot(inds, 100*mean(NeSweepNonclust.percentSV_EallTypeAvg),'r-');
%     errbar_rw(gca, 100*std(NeSweepNonclust.percentSV_EallTypeAvg)/sqrt(maxind_non), 'r');
% 
%     ylim([0 100])
%     ylabel('% Shared Variance','FontSize',axisfnt);
%     xlabel('E-to-I neuron ratio','FontSize',axisfnt);
%     set(gca,'FontSize',axislabelfnt)
%     set(gca,'TickDir','out')
%     set(gca,'XTickLabel', {'All I', '20:80', '40:60', '60:40', '80:20', 'All E'});
%     text(lblx, 104, 'B   E % SV', 'FontSize', figurelabelfnt);
%     print(sprintf('fig3b_%dN_%dT', nneurons, ntrials), '-depsc', '-r300');
% 
%     %% Plot Fig 3 C: inhibitiory percent shared variance
%     figure % percent shared
%     inds = 0:10:100
%     plot(inds, 100*mean(NeSweepClustered.percentSV_IallTypeAvg),'b--');
%     hold on
%     errbar_rw(gca, 100*std(NeSweepClustered.percentSV_IallTypeAvg)/sqrt(maxind_cl),'b');
% 
%     plot(inds, 100*mean(NeSweepNonclust.percentSV_IallTypeAvg),'b-');
%     errbar_rw(gca, 100*std(NeSweepNonclust.percentSV_IallTypeAvg)/sqrt(maxind_non), 'b');
%     
%     ylim([0 100])
%     ylabel('% Shared Variance','FontSize',axisfnt);
%     xlabel('E-to-I neuron ratio','FontSize',axisfnt);
%     set(gca,'XTickLabel', {'All I', '20:80', '40:60', '60:40', '80:20', 'All E'});
%     set(gca,'FontSize',axislabelfnt);
%     set(gca,'TickDir','out')
%     text(lblx, 104, 'C   I % SV', 'FontSize', figurelabelfnt);
%     print(sprintf('fig3c_%dN_%dT', nneurons, ntrials), '-depsc', '-r300');
end
