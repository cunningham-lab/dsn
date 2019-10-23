function rasterspikes = plotSpikeTrains(dataDir, trials, interval,ind1)
%PLOTCLUSTER Summary of this function goes here
%   Detailed explanation goes here
neurons = 5000; % assumes data-set has 4000 excitatory neurons
dataPath = ['/Users/sbittner/Documents/MATLAB/15_CNBC/data/', dataDir, filesep];
ntrials = length(trials);
totalLength = 2200;
xfont = 24;
yfont = 24;
axisfont = 16;
%interval = 1; % use interval from 1.2s to 2.2s
for i=1:ntrials
    trial = trials(i);
    trialData = load([dataPath, sprintf('%d.mat', trial)]);
    t = trialData.t;
    if (interval)
        t = t - 1200;
        trialLength = 1000;
    else
        trialLength = totalLength;
    end
    rasterspikes = [];
    spacing = 0;
    for j=1:neurons
        tj = t(j,:);
        rasterspikes = [rasterspikes, tj(tj > 0) + spacing];
        spacing = spacing + trialLength;
    end
    subplot(1,2,ind1);
    h = rasterplot_neurons(rasterspikes, neurons, trialLength);
    %title(['\bf ', strrep(dataDir, '_', ''), sprintf('Trial %d', trial)]);
    ax = gca;
    if (interval)
        set(ax, 'XTick', [1, 1000]);
        set(ax, 'YTick', 1000:1000:5000);
        set(ax, 'XTickLabel', {'0.0', '1.0'});
        set(gca, 'FontSize', axisfont);
        xlabel('time (s)', 'FontSize', xfont);
        ylabel('neuron', 'FontSize', yfont);
    else
        set(ax, 'XTick', [1, 1000]);
        set(ax, 'XTickLabel', {'0.0', '1.0'});
        set(gca, 'FontSize', axisfont);
        xlabel('time (s)', 'FontSize', xfont);
        ylabel('neuron', 'FontSize', yfont);
    end

end

