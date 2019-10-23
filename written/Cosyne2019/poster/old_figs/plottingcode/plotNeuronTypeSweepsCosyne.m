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
axisfnt = 18;
axislabelfnt = 8;
figurelabelfnt = 18;
xaxislimit = 101;
lblx = 5;
lbly = 85;
figurelabel2fnt = 14;
lblx2 = 45;
lbly2 = 8;

nneurons = 100;
trial_counts = [1200,10000];
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
    box off
    inds = 0:10:100;    
    plot(inds,mean(NeSweepClustered.dimensionality), 'k--');
    hold on
    errbar_rw(gca,std(NeSweepClustered.dimensionality)/sqrt(maxind_cl), 'k');
    plot(inds,mean(NeSweepNonclust.dimensionality), 'k-');
    errbar_rw(gca,std(NeSweepNonclust.dimensionality)/sqrt(maxind_non),'k');
    xlim([0 101]);
    ylabel('Dimensionality','FontSize',axisfnt)
    %ylabel('Dimensionality','FontSize',axisfnt)
    if (ntrials == 20000 || ntrials == 5000)
        xlabel('E-to-I neuron ratio','FontSize',axisfnt)
    elseif (ntrials == 1200 || ntrials == 10000)
        xlabel(' ', 'FontSize', axisfnt);
    end
    
    set(gca,'FontSize',axislabelfnt)
    ylim([0 80]);
    set(gca,'TickDir','out');
    set(gca, 'XTick', [0:20:100]);
    set(gca,'XTickLabel', {'All I', '20:80', '40:60', '60:40', '80:20', 'All E'});
    set(gca,'FontSize',axislabelfnt)
    set(gca,'YTick', 0:20:80);
    if (ntrials == 5000)
        text(lblx2, 9*lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
        text(lblx2, 7.8*lbly2, sprintf('100 Neurons', ntrials), 'FontSize', figurelabel2fnt);
%         text(lblx2+10, 3.2*lbly2, sprintf('clustered', ntrials), 'FontSize', figurelabel2fnt);
%         text(lblx2-30, .5*lbly2, sprintf('non-clustered', ntrials), 'FontSize', figurelabel2fnt);
    elseif (ntrials == 20000)
        text(lblx2, 9*lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
    else
        text(lblx2, 9*lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
        set(gca, 'XTickLabels', {''});
    end
    %text(lblx, lbly, 'A   Dimensionality', 'FontSize', figurelabelfnt);
    box off
    print(sprintf('fig3a_%dms_%dN_%dT', binsize, nneurons, ntrials), '-depsc', '-r300');

    %% Plot Fig 3 B: excitatory percent shared variance
    figure % percent shared
    inds = 0:10:100;
    
    plot(inds, 100*mean(NeSweepClustered.percentSV_EallTypeAvg),'r--');
    hold on
    errbar_rw(gca, 100*std(NeSweepClustered.percentSV_EallTypeAvg)/sqrt(maxind_cl),'r');

    plot(inds, 100*mean(NeSweepNonclust.percentSV_EallTypeAvg),'r-');
    errbar_rw(gca, 100*std(NeSweepNonclust.percentSV_EallTypeAvg)/sqrt(maxind_non), 'r');
    
    plot(inds, 100*mean(NeSweepNonclust.percentSV_IallTypeAvg),'b-');
    errbar_rw(gca, 100*std(NeSweepNonclust.percentSV_IallTypeAvg)/sqrt(maxind_non), 'b');
    
    plot(inds, 100*mean(NeSweepClustered.percentSV_IallTypeAvg),'b--');
    errbar_rw(gca, 100*std(NeSweepClustered.percentSV_IallTypeAvg)/sqrt(maxind_cl),'b');

    ylim([0 100])
    set(gca, 'XTick', 0:20:100);
    set(gca,'XTickLabel', {'All I', '20:80', '40:60', '60:40', '80:20', 'All E'});
    set(gca,'FontSize',axislabelfnt)
    %ylabel('% Shared Variance','FontSize',axisfnt);
    ylabel('%SV','FontSize',axisfnt)
    if (ntrials == 20000 || ntrials == 5000)
        xlabel('E-to-I neuron ratio','FontSize',axisfnt)
    elseif (ntrials == 1200 || ntrials == 10000)
        xlabel(' ', 'FontSize', axisfnt);
    end
    set(gca,'YTick', 0:20:100);
    set(gca,'TickDir','out');
    
    % Place plot labels
    switch ntrials
        case 1200
        	%text(lblx2, lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
            set(gca, 'XTickLabels', {''});
        case 5000
            %text(lblx2, 2.4 *lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
            ntsxshift1 = 20;
            ntsxshift2 = 2;
            ntsxshift3 = 16;
%             text(ntsxshift1, 90, sprintf('clustered E', ntrials), 'FontSize', figurelabel2fnt, 'Color', 'red');
%             text(ntsxshift1, 72, sprintf('clustered I', ntrials), 'FontSize', figurelabel2fnt, 'Color', 'blue');
%             text(ntsxshift2, 3, sprintf('non-clustered E', ntrials), 'FontSize', figurelabel2fnt, 'Color', 'red');
%             text(ntsxshift3, 28, sprintf('non-clustered I', ntrials), 'FontSize', figurelabel2fnt, 'Color', 'blue');
            
        case 10000
            set(gca, 'XTickLabels', {''});
            %text(lblx2, 2.1*lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
        case 20000
            %text(lblx2, lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
    end
    %text(lblx, 104, 'B   E % SV', 'FontSize', figurelabelfnt);
    box off
    print(sprintf('fig3b_%dN_%dT', nneurons, ntrials), '-depsc', '-r300');
end
