%plotClassAvgWaveforms
%plotClassAvgWaveforms Summary of this function goes here
%   Detailed explanation goes here
expstrs = {'101r001p23', '102l001p12', '102r001p06', '565r001p26', '567l002p09', '567r001p10'};
%expstrs = {'Bo130404_s4ae_fixblank_active',
%            'Bo130405_s5ae_fixblank_active',
%            'Bo130408_s6ae_fixblank_active',
%            'Bo130409_s7ae_fixblank_active',
%            'Bo130410_s8ae_fixblank_active',
%            'Bo130416_s10ae_fixblank_active',
%            'Bo130417_s11ae_fixblank_active',
%            'Bo130418_s12ae_fixblank_active',
%            'Bo130419_s13ae_fixblank_active',
%            'Bo130422_s14ae_fixblank_active',
%            'Bo130423_s15ae_fixblank_active',
%            'Bo130424_s16ae_fixblank_active',
%            'Bo130425_s17ae_fixblank_active',
%            'Bo130426_s18ae_fixblank_active'};

setdefaultplot;
%datadir = '/Users/sbittner/Documents/MATLAB/15_CNBC/src/AdamsCode/data/'
figdir = '/Users/sbittner/Documents/MATLAB/15_CNBC/src/AdamsCode/figures/';
datadir = '/Users/sbittner/Documents/MATLAB/15_CNBC/data/v1/';

PTHRESH = .5;
HISTBINS = 40;
PROPCOLORS = {'k', 'k', 'k'};
PROPSTR = {'time shift of AHP (ms)', 'width of AHP (ms)', 'difference in rise and fall rates'}; 
FNSTR = {'TimeShiftDist', 'AHPWidthDist', 'SlopeDist'};
axislims = [.5,   0, -0.5;
            1.4, 1,  1; ];

wavesAll = [];
okWavesAll = [];
coefValsAll = [];
EwavesAll = [];
IwavesAll = [];
%%
for i=1:length(expstrs)
    expstr = expstrs{i};
    avgfile = [datadir, 'avg_', expstr, '.mat'];
    classfile = [datadir, 'class_', expstr, '.mat'];

    load(avgfile);
    load(classfile);
    

    mu = obj.mu;
    % put in correct order
    [val, ind] = min(mu(:,1));
    if (ind ~= 1)
        mu = [mu(2,:); mu(1,:)];
    end
    
    % update all experiment variables
    wavesAll = [wavesAll, waves];
    okWavesAll = [okWavesAll; okWaves];
    coefValsAll = [coefValsAll; coefVals];
       
    [waveformlen, neurons] = size(waves);

    figure;
    hold on;
    sampsPerMsec = 30;
    x = (1:waveformlen)/sampsPerMsec;
    inds = 1:neurons;
    % Determine E and I neuron for this experiment
    Iinds = inds(P(:,1) >= PTHRESH & okWaves);
    Einds = inds(P(:,2) >= PTHRESH & okWaves);
    Ilen = length(Iinds);
    Elen = length(Einds);
    % Compute the average neuron waveform of each type
    Iwaves = waves(:,Iinds);
    Ewaves = waves(:,Einds);
    
    % Save the E and I neuron average waveforms for all exeriments
    EwavesAll = [EwavesAll, Ewaves];
    IwavesAll = [IwavesAll, Iwaves];
    
    avg_Iwave = mean(Iwaves, 2);
%     avg_Ewave = mean(Ewaves, 2);    

 end
%%
figure;
hold on;
load('temp.mat');

save('temp.mat', 'x', 'avg_Iwave');

classfile = [datadir, 'class_all.mat'];
load(classfile);
mu = obj.mu;
% put in correct order
[val, ind] = min(mu(:,1));
if (ind ~= 1)
    mu = [mu(2,:); mu(1,:)];
end
    
okWavesAll = okWavesAll == 1;
avg_Iwave = mean(IwavesAll, 2);
avg_Ewave = mean(EwavesAll, 2);    
plot(x, avg_Iwave, 'Color', [0,.6,0], 'LineWidth', 5);
axis([0,max(x),min(min(IwavesAll))-5, max(max(IwavesAll))+5]);
ylabel('amplitude mV', 'FontSize', fsz);
xlabel('time (ms)', 'FontSize', fsz);
title('All experiments', 'FontSize', fsz+2);
print([figdir, sprintf('figClassAvgWaves_All')], '-depsc', '-r300');
 ylim([-100,50]);









