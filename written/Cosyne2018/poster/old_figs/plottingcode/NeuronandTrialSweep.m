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
axisfnt = 20;
axislblfnt = 10;
legendfnt = 14;
figurelabelfnt = 24;
Mscale = 40;
figurelabel2fnt = 14;
lblx2 = 55;
lbly2 = 3;
nneurons = 80;
ntrials = 1200;
binsize = 1000;
circleScale = 30;
circlesize = circleScale*ones(8,1);

%Load sweep data
ndraws = 5;
NSdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/results/NeuronSweeps/';
TSdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/results/TrialSweeps/';
% Load inhibitory neuron sweep data
EclustNS = load([NSdir, sprintf('E_%dms_%dN_%dT_clustered_NeuronSweep.mat', binsize, nneurons, ntrials)]);
EclustTS = load([TSdir, sprintf('E_%dms_%dN_%dT_clustered_TrialSweep.mat', binsize, nneurons, ntrials)]);
EnonclustNS = load([NSdir, sprintf('E_%dms_%dN_%dT_nonclust_NeuronSweep.mat', binsize, nneurons, ntrials)]);
EnonclustTS = load([TSdir, sprintf('E_%dms_%dN_%dT_nonclust_TrialSweep.mat', binsize, nneurons, ntrials)]);

% Load inhibitory neuron sweep data
IclustNS = load([NSdir, sprintf('I_%dms_%dN_%dT_clustered_NeuronSweep.mat', binsize, nneurons, ntrials)]);
IclustTS = load([TSdir, sprintf('I_%dms_%dN_%dT_clustered_TrialSweep.mat', binsize, nneurons, ntrials)]);
InonclustNS = load([NSdir, sprintf('I_%dms_%dN_%dT_nonclust_NeuronSweep.mat', binsize, nneurons, ntrials)]);
InonclustTS = load([TSdir, sprintf('I_%dms_%dN_%dT_nonclust_TrialSweep.mat', binsize, nneurons, ntrials)]);

% Clip off uncomputed matrix
[EclustNS, ndraws] = filterNotYetComputed(EclustNS, 'NeuronSweep');
[EclustTS, ndraws] = filterNotYetComputed(EclustTS, 'TrialSweep');
%[EnonclustNS, ndraws] = filterNotYetComputed(EnonclustNS, 'NeuronSweep');
%[EnonclustTS, ndraws] = filterNotYetComputed(EnonclustTS, 'TrialSweep');
[IclustNS, ndraws] = filterNotYetComputed(IclustNS, 'NeuronSweep');
[IclustTS, ndraws] = filterNotYetComputed(IclustTS, 'TrialSweep');
%[InonclustNS, ndraws] = filterNotYetComputed(InonclustNS, 'NeuronSweep');
%[InonclustTS, ndraws] = filterNotYetComputed(InonclustTS, 'TrialSweep');


%% Plot A figs
% calculate mean and error
lblx = 5;
lbly = 62.4;

% plot Fig 1 A
figure % dimensionality
inds = 10:10:80;
ndraws = size(EclustNS.dimensionality,1);
scatter(inds,mean(EclustNS.dimensionality), circlesize, 'o',  'r','fill');
hold on
errbar_rw(gca,std(EclustNS.dimensionality)/sqrt(ndraws),'r')
xlim([0 85])

ndraws = size(IclustNS.dimensionality,1);
scatter(inds, mean(IclustNS.dimensionality),circlesize, 'o','b','fill');
errbar_rw(gca,std(IclustNS.dimensionality)/sqrt(ndraws),'b')
xlim([0 85])
ylabel('Dimensionality','FontSize',axisfnt)
%xlabel('Neuron Count','FontSize',10)
set(gca,'FontSize',axislblfnt)
ylim([0 25])
set(gca,'TickDir','out')
%text(lblx, lbly, 'A', 'FontSize', figurelabelfnt);
text(lblx2, lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
text(45,20.5,'excitatory', 'Color', 'r', 'FontName', 'Arial', 'FontSize', legendfnt);
text(52,14.5,'inhibitory', 'Color', 'b', 'FontName', 'Arial', 'FontSize', legendfnt);
h = gca;
set(h,'XTick', []);
print(sprintf('fig1a_%dT_%dms_clustered', ntrials, binsize), '-depsc', '-r300');

% % Plot Fig 2 A: non-clustered network network sweep dimensionality
% figure % percent shared
% ndraws = size(EnonclustNS.dimensionality,1);
% scatter(inds, mean(EnonclustNS.dimensionality),'o','r','fill');
% hold on
% errbar_rw(gca,std(EnonclustNS.dimensionality)/sqrt(ndraws),'r')
% xlim([0 85])
% ndraws = size(InonclustNS.dimensionality,1);
% scatter(inds, mean(InonclustNS.dimensionality),'o', 'b','fill');
% errbar_rw(gca, std(InonclustNS.dimensionality)/sqrt(ndraws),'b')
% ylim([0 60])
% ylabel('Dimensionality','FontSize',axisfnt)
% set(gca,'FontSize',axislblfnt)
% set(gca,'TickDir','out')
% text(lblx, lbly, 'A', 'FontSize', figurelabelfnt);
% text(lblx2, lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
% %text(60,14.5,'E-neurons', 'Color', 'r', 'FontName', 'Arial', 'FontSize', legendfnt);
% %text(62,45,'I-neurons', 'Color', 'b', 'FontName', 'Arial', 'FontSize', legendfnt);
% print(sprintf('fig2a_%dT_%dms_clustered', ntrials, binsize), '-depsc', '-r300');
%% Plot B figs
% calculate mean and error
lblx = 5;
lbly = 62.4;

% plot Fig 1 B
figure % dimensionality
if (ntrials == 1200)
    inds = 200:200:1200;
else
    inds = 500:500:5000;
end
circlesize = circleScale*ones(length(inds), 1);
scatter(inds,mean(EclustTS.dimensionality), circlesize, 'o',  'r','fill');
hold on
errbar_rw(gca,std(EclustTS.dimensionality)/sqrt(ndraws),'r')
xlim([0, 1.05*ntrials])

ndraws = size(IclustTS.dimensionality,1);
scatter(inds, mean(IclustTS.dimensionality),circlesize,'o','b','fill');
errbar_rw(gca,std(IclustTS.dimensionality)/sqrt(ndraws),'b')
xlim([0, 1.05*ntrials])
%ylabel('Dimensionality','FontSize',axisfnt)
%xlabel('Neuron Count','FontSize',10)
set(gca,'FontSize',axislblfnt)
ylim([0 25])
set(gca,'TickDir','out')
%text(lblx, lbly, 'B', 'FontSize', figurelabelfnt);
text(ntrials*lblx2/100, lbly2, '80 Neurons', 'FontSize', figurelabel2fnt);
h = gca;
set(h,'XTick', []);
print(sprintf('fig1b_%dT_%dms_clustered', ntrials, binsize), '-depsc', '-r300');


% % Plot Fig 2 B: non-clustered network trial sweep dimensionality
% figure % percent shared
% ndraws = size(EnonclustTS.dimensionality,1);
% scatter(inds, mean(EnonclustTS.dimensionality),'o','r','fill');
% hold on
% errbar_rw(gca,std(EnonclustTS.dimensionality)/sqrt(ndraws),'r')
% xlim([0, 1.05*ntrials])
% 
% ndraws = size(InonclustTS.dimensionality,1);
% scatter(inds, mean(InonclustTS.dimensionality),'o', 'b','fill');
% errbar_rw(gca, std(InonclustTS.dimensionality)/sqrt(ndraws),'b')
% ylim([0 60])
% %ylabel('Dimensionality','FontSize',axisfnt)
% set(gca,'FontSize',axislblfnt)
% set(gca,'TickDir','out')
% text(lblx, lbly, 'B', 'FontSize', figurelabelfnt);
% text(ntrials*lblx2/100, lbly2, '80 Neurons', 'FontSize', figurelabel2fnt);
% print(sprintf('fig2b_%dT_%dms_clustered', ntrials, binsize), '-depsc', '-r300');


%% Plot C figs
% calculate mean and error
lblx = 5;
lbly = 104;

% plot Fig 1 C
figure % dimensionality
inds = 10:10:80;
circlesize = circleScale*ones(length(inds), 1);
ndraws = size(EclustNS.percentSV,1);
scatter(inds,100*mean(EclustNS.percentSV),circlesize, 'o',  'r','fill');
hold on
errbar_rw(gca,100*std(EclustNS.percentSV)/sqrt(ndraws),'r')
xlim([0 85])

ndraws = size(IclustNS.percentSV,1);
scatter(inds, 100*mean(IclustNS.percentSV),circlesize,'o','b','fill');
errbar_rw(gca,100*std(IclustNS.percentSV)/sqrt(ndraws),'b')
xlim([0 85])
ylabel('%SV','FontSize',axisfnt)
xlabel('Neuron Count','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
ylim([0 100])
set(gca,'TickDir','out')
%text(lblx, lbly, 'C', 'FontSize', figurelabelfnt);
text(lblx2, 4*lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
print(sprintf('fig1c_%dT_%dms_clustered', ntrials, binsize), '-depsc', '-r300');

% % Plot Fig 2 C: non-clustered network network sweep percent SV
% figure % percent shared
% ndraws = size(EnonclustNS.percentSV,1);
% scatter(inds, 100*mean(EnonclustNS.percentSV),'o','r','fill');
% hold on
% errbar_rw(gca, 100*std(EnonclustNS.percentSV)/sqrt(ndraws),'r')
% xlim([0 85])
% 
% ndraws = size(InonclustNS.percentSV,1);
% scatter(inds, 100*mean(InonclustNS.percentSV),'o', 'b','fill');
% errbar_rw(gca, 100*std(InonclustNS.percentSV)/sqrt(ndraws),'b')
% ylim([0 100])
% ylabel('% Shared Variance','FontSize',axisfnt)
% xlabel('Neuron Count','FontSize',axisfnt)
% set(gca,'FontSize',axislblfnt)
% set(gca,'TickDir','out')
% text(lblx, lbly, 'C', 'FontSize', figurelabelfnt);
% text(lblx2, lbly2, sprintf('%d Trials', ntrials), 'FontSize', figurelabel2fnt);
% print(sprintf('fig2c_%dT_%dms_clustered', ntrials, binsize), '-depsc', '-r300');

%% Plot D figs
% calculate mean and error
lblx = 5;
lbly = 104;

% plot Fig 1 D
figure % percent SV
if (ntrials == 1200)
    inds = 200:200:1200;
else
    inds = 500:500:5000;
end
circlesize = circleScale*ones(length(inds), 1);
ndraws = size(EclustTS.percentSV,1);
scatter(inds,100*mean(EclustTS.percentSV),circlesize, 'o',  'r','fill');
hold on
errbar_rw(gca,100*std(EclustTS.percentSV)/sqrt(ndraws),'r')
xlim([0, 1.05*ntrials])

ndraws = size(IclustTS.percentSV,1);
scatter(inds, 100*mean(IclustTS.percentSV),circlesize,'o','b','fill');
errbar_rw(gca,100*std(IclustTS.percentSV)/sqrt(ndraws),'b')
xlim([0, 1.05*ntrials])
%ylabel('% Shared Variance','FontSize',axisfnt)
xlabel('Trial Count','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
ylim([0 100])
set(gca,'TickDir','out')
%text(lblx, lbly, 'D', 'FontSize', figurelabelfnt);
text(ntrials*lblx2/100, 4*lbly2, '80 Neurons', 'FontSize', figurelabel2fnt);
print(sprintf('fig1d_%dT_%dms_clustered', ntrials, binsize), '-depsc', '-r300');

% % Plot Fig 2 D: non-clustered network trial sweep percent SV
% figure % percent shared
% ndraws = size(EnonclustTS.percentSV,1);
% scatter(inds, 100*mean(EnonclustTS.percentSV),'o','r','fill');
% hold on
% errbar_rw(gca, 100*std(EnonclustTS.percentSV)/sqrt(ndraws),'r')
% xlim([0, 1.05*ntrials])
% 
% ndraws = size(InonclustTS.percentSV,1);
% scatter(inds, 100*mean(InonclustTS.percentSV),'o', 'b','fill');
% errbar_rw(gca, 100*std(InonclustTS.percentSV)/sqrt(ndraws),'b')
% ylim([0 100])
% ylabel('% Shared Variance','FontSize',axisfnt)
% xlabel('Trial Count','FontSize',axisfnt)
% set(gca,'FontSize',axislblfnt)
% set(gca,'TickDir','out')
% text(lblx, lbly, 'D', 'FontSize', figurelabelfnt);
% text(ntrials*lblx2/100, lbly2, '80 Neurons', 'FontSize', figurelabel2fnt);
% print(sprintf('fig2d_%dT_%dms_clustered', ntrials, binsize), '-depsc', '-r300');
