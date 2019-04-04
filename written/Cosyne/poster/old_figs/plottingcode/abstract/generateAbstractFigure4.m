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

%% Load all data
ndraws = 5;
NSdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/results/NeuronSweeps/';
TSdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/results/TrialSweeps/';
INS = load([NSdir, 'INeuronSweep_102l001p12_all36_.mat']);
ITS = load([TSdir, 'ITrialSweep_102l001p12_all36.mat']);
ENS = load([NSdir, 'ENeuronSweep_102l001p12_all36_.mat']);
ETS = load([TSdir, 'ETrialSweep_102l001p12_all36.mat']);


%% Plot Fig 4 A
% calculate mean and error
lblx = 5;
lbly = 26;
% plot Fig 1 A
figure % dimensionality
inds = 5:5:50;
scatter(inds, ENS.dimensionality,'o',  'r','fill');
hold on;
scatter(inds, INS.dimensionality(1:length(inds)),'o',  'b','fill');
xlim([0 52])
ylabel('Dimensionality','FontSize',axisfnt)
xlabel('Neuron Count','FontSize',axisfnt)
set(gca,'FontSize',axisfnt)
ylim([0 25])
set(gca,'TickDir','out')
text(lblx/2, lbly, 'A', 'FontSize', figurelabelfnt);
print('fig4a', '-depsc', '-r300');

%% Plot Fig 4 C
% calculate mean and error
lblx = 5;
lbly = 26;
% plot Fig 1 A
figure % dimensionality
inds = 5:5:50;
scatter(inds, 100*ENS.percentSV,'o',  'r','fill');
hold on;
scatter(inds, 100*INS.percentSV(1:length(inds)),'o',  'b','fill');
xlim([0 52])
ylabel('% Shared Variance','FontSize',axisfnt)
xlabel('Neuron Count','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
ylim([0 100])
set(gca,'TickDir','out')
text(lblx/2, 4*lbly, 'C', 'FontSize', figurelabelfnt);
print('fig4c', '-depsc', '-r300');

%% Plot Fig 4 B
% calculate mean and error
lblx = 5;
lbly = 26;
% plot Fig 1 A
figure % dimensionality
inds = 200:200:1200;
scatter(inds, ETS.dimensionality(1:length(inds)),'o','r','fill');
hold on;
scatter(inds, ITS.dimensionality(1:length(inds)),'o','b','fill');
xlim([0 1300])
ylabel('Dimensionality','FontSize',axisfnt)
%label('Neuron Count','FontSize',10)
set(gca,'FontSize',axislblfnt)
ylim([0 25])
set(gca,'TickDir','out')
text(12*lblx, lbly, 'B', 'FontSize', figurelabelfnt);
print('fig4b', '-depsc', '-r300');

%% Plot Fig 4 C
% calculate mean and error
lblx = 5;
lbly = 26;
% plot Fig 1 A
figure % dimensionality
inds = 200:200:1200;
scatter(inds, 100*ETS.percentSV(1:length(inds)),'o',  'r','fill');
hold on;
scatter(inds, 100*ITS.percentSV(1:length(inds)),'o',  'b','fill');
xlim([0 1300])
ylabel('% Shared Variance','FontSize',axisfnt)
xlabel('Trial Count','FontSize',axisfnt)
set(gca,'FontSize',axislblfnt)
ylim([0 100])
set(gca,'TickDir','out')
text(12*lblx, 4*lbly, 'D', 'FontSize', figurelabelfnt);
print('fig4', '-depsc', '-r300');
