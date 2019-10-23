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
%setdefaultplot % set the plot parameters
errorbarmult = 4;
grey = [0.5 0.5 0.5];
figurelabelfnt = 18;
axisfnt = 18;
axislblfnt = 14;

%% Load inhibitory neuron sweep data
ndraws = 5;
NSdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/results/NeuronSweeps/';
TSdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/results/TrialSweeps/';
IclustNS = load([NSdir, 'NeuronSweep_clustered_build_2.mat']);
IclustTS = load([TSdir, 'TrialSweep_clustered_1.mat']);
InonclustNS = load([NSdir, 'NeuronSweep_nonclust_build_2.mat']);
InonclustTS = load([TSdir, 'TrialSweep_nonclust_1.mat']);

IclustTS.dimensionality = IclustTS.dimensionality(:,1:6);
IclustTS.percentSV = IclustTS.percentSV(:,1:6);
InonclustTS.dimensionality = InonclustTS.dimensionality(:,1:6);
InonclustTS.percentSV = InonclustTS.percentSV(:,1:6);

% Compute clustered I neuron sweep properties
meanIclustNSdim = mean(IclustNS.dimensionality(1:ndraws, :));
errIclustNSdim = std(IclustNS.dimensionality(1:ndraws, :))/sqrt(ndraws);
meanIclustNSpsv = mean(IclustNS.percentSV(1:ndraws, :));
errIclustNSpsv = std(IclustNS.percentSV(1:ndraws, :))/sqrt(ndraws);

% Compute clustered I trial sweep properties
meanIclustTSdim = mean(IclustTS.dimensionality(1:ndraws, :));
errIclustTSdim = std(IclustTS.dimensionality(1:ndraws, :))/sqrt(ndraws);
meanIclustTSpsv = mean(IclustTS.percentSV(1:ndraws, :));
errIclustTSpsv = std(IclustTS.percentSV(1:ndraws, :))/sqrt(ndraws);

% Compute non-clustered I neuron sweep properties
meanInonclustNSdim = mean(InonclustNS.dimensionality(1:ndraws, :));
errInonclustNSdim = std(InonclustNS.dimensionality(1:ndraws, :))/sqrt(ndraws);
meanInonclustNSpsv = mean(InonclustNS.percentSV(1:ndraws, :));
errInonclustNSpsv = std(InonclustNS.percentSV(1:ndraws, :))/sqrt(ndraws);

% Compute non-clustered I trial sweep properties
meanInonclustTSdim = mean(InonclustTS.dimensionality(1:ndraws, :));
errInonclustTSdim = std(InonclustTS.dimensionality(1:ndraws, :))/sqrt(ndraws);
meanInonclustTSpsv = mean(InonclustTS.percentSV(1:ndraws, :));
errInonclustTSpsv = std(InonclustTS.percentSV(1:ndraws, :))/sqrt(ndraws);

%% Figure 2
% load increase neuron low 1200 trial data
incneurontriallowc0 = load('compfamatnoverincneuron1200noclust');
incneurontriallowc80 = load('compfamatnoverincneuron1200clust');

%% Plot A figs
% calculate mean and error
lblx = 5;
lbly = 26;
xdimun0 = unique(incneurontriallowc0.comparefa.xdimension);
meanval0 = zeros(length(xdimun0),1);
err0 = zeros(length(xdimun0),1);
xdimun80 = unique(incneurontriallowc80.comparefa.xdimension);
meanval80 = zeros(length(xdimun80),1);
err80 = zeros(length(xdimun80),1);
for n = 1:length(xdimun0)
    tind0 = ismember(incneurontriallowc0.comparefa.xdimension,xdimun0(n));
    meanval0(n) = mean(incneurontriallowc0.comparefa.numdimpercentexplainedsc(tind0)-1);
    err0(n) = std(incneurontriallowc0.comparefa.numdimpercentexplainedsc(tind0)-1,[],2)./sum(tind0)^0.5;
end
for n = 1:length(xdimun80)
    tind80 = ismember(incneurontriallowc80.comparefa.xdimension,xdimun80(n));
    meanval80(n) = mean(incneurontriallowc80.comparefa.numdimpercentexplainedsc(tind80)-1);
    err80(n) = std(incneurontriallowc80.comparefa.numdimpercentexplainedsc(tind80)-1,[],2)./sum(tind80)^0.5;
end

% plot Fig 1 A
figure % dimensionality
inds = xdimun80<=80;
scatter(xdimun80(inds),meanval80(inds),'o',  'r','fill');
hold on
errbar_rw(gca,err80(inds)','k')
xlim([0 85])

scatter(xdimun80(inds),meanIclustNSdim(inds),'o','b','fill');
errbar_rw(gca,errIclustNSdim(inds),'b')
xlim([0 85])
ylabel('Dimensionality','FontSize',axisfnt)
%xlabel('Neuron Count','FontSize',10)
set(gca,'FontSize',axislblfnt)
ylim([0 25])
set(gca,'TickDir','out')
text(lblx, lbly, 'A', 'FontSize', figurelabelfnt);
print('fig1a', '-depsc', '-r300');

% Plot Fig 2 A: non-clustered network network sweep dimensionality
figure % percent shared
inds = xdimun0<=1200;
scatter(xdimun0(inds),meanInonclustNSdim(inds),'o','b','fill');
hold on
errbar_rw(gca,errInonclustNSdim(inds),'b')
xlim([0 85])
inds = xdimun80<=1200;
scatter(xdimun80(inds),meanval0(inds),'o', 'r','fill');
errbar_rw(gca,err0(inds)','k')
ylim([0 5])
ylabel('Dimensionality','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
set(gca,'TickDir','out')
text(lblx, lbly/5, 'A', 'FontSize', figurelabelfnt);
print('fig2a', '-depsc', '-r300');

%% Plot B figs

% calculate mean and error
xdimun0 = unique(incneurontriallowc0.comparefa.xdimension);
meanval0 = zeros(length(xdimun0),1);
err0 = zeros(length(xdimun0),1);
xdimun80 = unique(incneurontriallowc80.comparefa.xdimension);
meanval80 = zeros(length(xdimun80),1);
err80 = zeros(length(xdimun80),1);
for n = 1:length(xdimun0)
    tind0 = ismember(incneurontriallowc0.comparefa.xdimension,xdimun0(n));
    meanval0(n) = mean(incneurontriallowc0.comparefa.meanpercentshared(tind0)*100);
    err0(n) = std(incneurontriallowc0.comparefa.meanpercentshared(tind0)*100,[],2)./sum(tind0)^0.5;
end
for n = 1:length(xdimun80)
    tind80 = ismember(incneurontriallowc80.comparefa.xdimension,xdimun80(n));
    meanval80(n) = mean(incneurontriallowc80.comparefa.meanpercentshared(tind80)*100);
    err80(n) = std(incneurontriallowc80.comparefa.meanpercentshared(tind80)*100,[],2)./sum(tind80)^0.5;
end


% Plot Fig 1. B
figure % shared variance
inds = xdimun0<=80;
scatter(xdimun0(inds),100*meanIclustNSpsv(inds), 'o','b','fill');
hold on
errbar_rw(gca, 100*errIclustNSpsv(inds),'b')
xlim([0 85])
inds = xdimun80<=80;
scatter(xdimun80(inds),meanval80(inds),'o', 'r','fill');
%errbar_rw(gca,err80(inds)','k')
xlim([0 85])
ylabel('% Shared Variance','FontSize',axisfnt)
xlabel('Neuron Count','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
ylim([0 100])
set(gca,'TickDir','out')
text(lblx, 4*lbly, 'B', 'FontSize', figurelabelfnt);
print('fig1b', '-depsc', '-r300');

% Plot Fig 2 B: non-clustered network network sweep % SV
figure % percent shared
scatter(xdimun0(inds),100*meanInonclustNSpsv(inds),'o','b','fill');
hold on
errbar_rw(gca,100*errInonclustNSpsv(inds),'b')
xlim([0 85])
inds = xdimun80<=1200;
scatter(xdimun80(inds),meanval0(inds),'o', 'r','fill');
errbar_rw(gca,err0(inds)','k')
xlim([0 85])
ylabel('% Shared Variance','FontSize',axisfnt)
xlabel('Neuron Count','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
ylim([0 10])
set(gca,'TickDir','out')
text(lblx, lbly*2/5, 'B', 'FontSize', figurelabelfnt);
print('fig2b', '-depsc', '-r300');

%% Plot C Figs

% load increase trial 80 neuron data 
inctriallowc0 = load('compfamateigvectrialsnoclust');
inctriallowc80 = load('compfamateigvectrialclust');

% calculate mean and error
xdimun0 = unique(inctriallowc0.comparefa.xdimension);
meanval0 = zeros(length(xdimun0),1);
err0 = zeros(length(xdimun0),1);
xdimun80 = unique(inctriallowc80.comparefa.xdimension);
meanval80 = zeros(length(xdimun80),1);
err80 = zeros(length(xdimun80),1);
for n = 1:length(xdimun0)
    tind0 = ismember(inctriallowc0.comparefa.xdimension,xdimun0(n));
    meanval0(n) = mean(inctriallowc0.comparefa.numdimpercentexplainedsc(tind0)-1);
    err0(n) = std(inctriallowc0.comparefa.numdimpercentexplainedsc(tind0)-1,[],2)./sum(tind0)^0.5;
end
for n = 1:length(xdimun80)
    tind80 = ismember(inctriallowc80.comparefa.xdimension,xdimun80(n));
    meanval80(n) = mean(inctriallowc80.comparefa.numdimpercentexplainedsc(tind80)-1);
    err80(n) = std(inctriallowc80.comparefa.numdimpercentexplainedsc(tind80)-1,[],2)./sum(tind80)^0.5;
end

% Plot Figure 1. C   Trial Sweep Dimensionality
% plot increase trial 80 neuron for simulation data
figure % dimensionality
inds = xdimun0<=1200;
scatter(xdimun0(inds),meanIclustTSdim(inds),'o','b','fill');
hold on
errbar_rw(gca,errIclustTSdim(inds),'b')
xlim([0 1250])
inds = xdimun80<=1200;
scatter(xdimun80(inds),meanval80(inds),'o', 'r','fill');
errbar_rw(gca,err80(inds)','k')
xlim([0 1250])
%ylabel('Dimensionality','FontSize',10)
%xlabel('Trial Count','FontSize',10)
set(gca,'FontSize',axislblfnt)
ylim([0 25])
set(gca,'TickDir','out')
text(12*lblx, lbly, 'C', 'FontSize', figurelabelfnt);
print('fig1c', '-depsc', '-r300');

% Plot Fig 2 C: non-clustered network trial sweep dimensionality
figure % percent shared
inds = xdimun0<=1200;
scatter(xdimun0(inds),meanInonclustTSdim(inds),'o','b','fill');
hold on
errbar_rw(gca,errInonclustTSdim(inds),'b')
xlim([0 1250])
inds = xdimun80<=1200;
scatter(xdimun80(inds),meanval0(inds),'o', 'r','fill');
errbar_rw(gca,err0(inds)','k')
xlim([0 1250])
%ylabel('% Shared Variance','FontSize',10)
%xlabel('Trial Count','FontSize',10)
set(gca,'FontSize',axislblfnt)
ylim([0 5])
set(gca,'TickDir','out')
text(12*lblx, lbly/5, 'C', 'FontSize', figurelabelfnt);
print('fig2c', '-depsc', '-r300');

%% Plot D Figs

% calculate mean and error
xdimun0 = unique(inctriallowc0.comparefa.xdimension);
meanval0 = zeros(length(xdimun0),1);
err0 = zeros(length(xdimun0),1);
xdimun80 = unique(inctriallowc80.comparefa.xdimension);
meanval80 = zeros(length(xdimun80),1);
err80 = zeros(length(xdimun80),1);
for n = 1:length(xdimun0)
    tind0 = ismember(inctriallowc0.comparefa.xdimension,xdimun0(n));
    meanval0(n) = mean(inctriallowc0.comparefa.meanpercentshared(tind0)*100);
    err0(n) = std(inctriallowc0.comparefa.meanpercentshared(tind0)*100,[],2)./sum(tind0)^0.5;
end
for n = 1:length(xdimun80)
    tind80 = ismember(inctriallowc80.comparefa.xdimension,xdimun80(n));
    meanval80(n) = mean(inctriallowc80.comparefa.meanpercentshared(tind80)*100);
    err80(n) = std(inctriallowc80.comparefa.meanpercentshared(tind80)*100,[],2)./sum(tind80)^0.5;
end

% Plot Fig 1 D
figure % percent shared
inds = xdimun0<=1200;
scatter(xdimun0(inds),100*meanIclustTSpsv(inds),'o','b','fill');
hold on
errbar_rw(gca,100*errIclustTSpsv(inds),'b')
xlim([0 1250])
inds = xdimun80<=1200;
scatter(xdimun80(inds),meanval80(inds),'o', 'r','fill');
errbar_rw(gca,err80(inds)','k')
xlim([0 1250])
%ylabel('% Shared Variance','FontSize',10)
xlabel('Trial Count','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
ylim([0 100])
set(gca,'TickDir','out')
text(12*lblx, 4*lbly, 'D', 'FontSize', figurelabelfnt);
print('fig1d', '-depsc', '-r300');


% Plot Fig 2 D: non-clustered network trial sweep % Sv
figure % percent shared
inds = xdimun0<=1200;
scatter(xdimun0(inds),100*meanInonclustTSpsv(inds),'o','b','fill');
hold on
errbar_rw(gca,100*errInonclustTSpsv(inds),'b')
xlim([0 1250])
inds = xdimun80<=1200;
scatter(xdimun80(inds),meanval0(inds),'o', 'r','fill');
errbar_rw(gca,err0(inds)','k')
xlim([0 1250])
%ylabel('% Shared Variance','FontSize',10)
xlabel('Trial Count','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
ylim([0 10])
set(gca,'TickDir','out')
text(12*lblx, lbly*2/5, 'D', 'FontSize', figurelabelfnt);
print('fig2d', '-depsc', '-r300');
