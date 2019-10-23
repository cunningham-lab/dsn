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
axislabelfnt = 10;
figurelabelfnt = 18;
xaxislimit = 101;
lblx = 5;
lbly = 26


%% Load inhibitory neuron sweep data
ndraws = 10;
NTSdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/results/NeuronTypeSweep/';
NeSweepClustered = load([NTSdir, 'NeSweep_clustered_3.mat']);
NeSweepNonclust = load([NTSdir, 'NeSweep_nonclust_3.mat']);


%% Plot A figs
% plot Fig 3 A: Dimensionality
figure % dimensionality
inds = 0:10:100
scatter(inds,mean(NeSweepClustered.dimensionality),'o',  'k','fill');
hold on
errbar_rw(gca,std(NeSweepClustered.dimensionality)/sqrt(ndraws),'k')
xlim([0 101])

scatter(inds,mean(NeSweepNonclust.dimensionality),'o','MarkerEdgeColor', grey);
errbar_rw(gca,std(NeSweepNonclust.dimensionality)/sqrt(ndraws),grey)
xlim([0 101])
ylabel('Dimensionality','FontSize',axisfnt)
xlabel('E-to-I neuron ratio','FontSize',axisfnt)
set(gca,'FontSize',axislabelfnt)
ylim([0 25])
set(gca,'TickDir','out')
set(gca,'XTickLabel', {'All I', '20:80', '40:60', '60:40', '80:20', 'All E'});
text(lblx, lbly, 'A', 'FontSize', figurelabelfnt);
print('fig3a', '-depsc', '-r300');

% Plot Fig 3 B: excitatory percent shared variance
figure % percent shared
inds = 0:10:100
scatter(inds, 100*mean(NeSweepClustered.percentSV_EallTypeAvg),'o',  'k','fill');
hold on
errbar_rw(gca, 100*std(NeSweepClustered.percentSV_EallTypeAvg)/sqrt(ndraws),'k')

scatter(inds, 100*mean(NeSweepNonclust.percentSV_EallTypeAvg),'o','MarkerEdgeColor', grey);
errbar_rw(gca, 100*std(NeSweepNonclust.percentSV_EallTypeAvg)/sqrt(ndraws),grey)
ylim([0 100])
ylabel('% Shared Variance','FontSize',axisfnt);
xlabel('E-to-I neuron ratio','FontSize',axisfnt);
set(gca,'FontSize',axislabelfnt)
set(gca,'TickDir','out')
set(gca,'XTickLabel', {'All I', '20:80', '40:60', '60:40', '80:20', 'All E'});
text(lblx, 4*lbly, 'B', 'FontSize', figurelabelfnt);
print('fig3b', '-depsc', '-r300');

% Plot Fig 3 C: inhibitiory percent shared variance
figure % percent shared
inds = 0:10:100
scatter(inds, 100*mean(NeSweepClustered.percentSV_IallTypeAvg),'o',  'k','fill');
hold on
errbar_rw(gca, 100*std(NeSweepClustered.percentSV_IallTypeAvg)/sqrt(ndraws),'k')

scatter(inds, 100*mean(NeSweepNonclust.percentSV_IallTypeAvg),'o','MarkerEdgeColor', grey);
errbar_rw(gca, 100*std(NeSweepNonclust.percentSV_IallTypeAvg)/sqrt(ndraws),grey)
ylim([0 100])
ylabel('% Shared Variance','FontSize',axisfnt);
xlabel('E-to-I neuron ratio','FontSize',axisfnt);
set(gca,'XTickLabel', {'All I', '20:80', '40:60', '60:40', '80:20', 'All E'});
set(gca,'FontSize',axislabelfnt)
set(gca,'TickDir','out')
text(lblx, 4*lbly, 'C', 'FontSize', figurelabelfnt);
print('fig3c', '-depsc', '-r300');

