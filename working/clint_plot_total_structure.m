%% Refactored TDA Plot Script
% Automatically compute axis limits from data, add tunable margin, and loop over plots

close all;

SCRIPT_DIR = fileparts(mfilename('fullpath'));
OUT_DIR    = SCRIPT_DIR;

totTbl = readtable(fullfile(OUT_DIR,'tda_summary_hat.csv'));
pairTbl = readtable(fullfile(OUT_DIR,'ripser_wasserstein_pairs_p2.csv'));

%---- Settings ----
mkrs = {'o','s','v','^','p','h'};
styleLUT = struct( ...
    'bert_strat800',{mkrs}, ...
    'bow_strat800',{mkrs},  ...
    'tfidf_strat800',{mkrs} ...
);
styleIDs   = fieldnames(styleLUT);
metricCols = {'base','ws','5nn','10nn','ws 5nn','ws 10nn'};

% Color arrays
clrMaps = {{'#000075','#4363d8','#42d4f4'}; {'#e6194B','#f58231','#911eb4'}; {'#808000','#3cb44b','#bfef45'}};
clrCA = cellfun(@(m)validatecolor(flip(m),'multiple'), clrMaps, 'UniformOutput', false);

% Marker picker function
pickMarker = @(ds,met) find([ ...
    nnz(ds=='_')==0, ...               % base
    nnz(ds=='_')==1&&contains(met,'ws'), ...
    nnz(ds=='_')==1&&contains(met,'5nn'),...
    nnz(ds=='_')==1&&contains(met,'10nn'),...
    nnz(ds=='_')==2&&contains(met,'5nn'),...
    nnz(ds=='_')==2&&contains(met,'10nn')],1,'first');

sz = 100;              % scatter size
marginFactor = 0.1;    % fraction of log-range to pad axes

% Define plots: fields xCol, yCol, labels, title
plots = struct( ...
    'xCol', {'tp_0','entropy_0','tp_012'}, ...
    'yCol', {'tp_12','entropy_12','entropy_012'}, ...
    'xlabel', {'TP_{0}','E_{0}(D)','TP_{0,1,2}'}, ...
    'ylabel', {'TP_{1,2}','E_{1,2}(D)','E_{0,1,2}(D)'}, ...
    'title', {'High vs Low-Dim Total Persistence', ...
              'High vs Low-Dim Persistent Entropy', ...
              'Full Persistent Entropy vs Total Persistence'} ...
);

% Loop over each plot configuration
for i = 1:numel(plots)
    cfg = plots(i);
    figure('Name',cfg.title); hold on; grid on;
    xlabel(cfg.xlabel); ylabel(cfg.ylabel);
    title(cfg.title);
    
    % Extract positive data for axis limits
    xData = totTbl.(cfg.xCol);
    yData = totTbl.(cfg.yCol);
    xPos = xData(xData>0);
    yPos = yData(yData>0);
    logX = log10([min(xPos), max(xPos)]);
    logY = log10([min(yPos), max(yPos)]);
    % Expand by marginFactor of the log-range
    xLim = 10.^(logX + [-1,1]*marginFactor*(logX(2)-logX(1)));
    yLim = 10.^(logY + [-1,1]*marginFactor*(logY(2)-logY(1)));
    axis([xLim, yLim]);
    set(gca, 'XScale','log','YScale','log','FontWeight','bold','FontSize',12);
    
    % Plot points
    for ii = 1:height(totTbl)
        ds = totTbl.dataset{ii};
        if ~isfield(styleLUT, ds), continue; end
        met = totTbl.metric{ii};
        mkIdx = pickMarker(met, met);
        if isempty(mkIdx), mkIdx = 1; end
        mk  = styleLUT.(ds){mkIdx};
        clrIdx = contains(met,'cos') + 2*contains(met,'mp1') + 3*contains(met,'mpinf');
        clr = clrCA{strcmp(styleIDs, ds)}(clrIdx, :);
        x = xData(ii); if x==0, x = xLim(1); end
        y = yData(ii); if y==0, y = yLim(1); end
        scatter(x, y, sz, clr, mk, 'LineWidth', 1.5);
    end
end

%--- Optional: Style grid legend (unchanged) ---
figure('Name','Style grid legend'); hold on;
for ii = 1:height(totTbl)
    ds = totTbl.dataset{ii}; if ~isfield(styleLUT, ds), continue; end
    met = totTbl.metric{ii};
    mkIdx = pickMarker(met, met); if isempty(mkIdx), mkIdx=1; end
    mk = styleLUT.(ds){mkIdx};
    clrIdx = contains(met,'cos') + 2*contains(met,'mp1') + 3*contains(met,'mpinf');
    clr = clrCA{strcmp(styleIDs, ds)}(clrIdx, :);
    x0 = mkIdx;
    y0 = 3*find(strcmp(styleIDs, ds)) - (4 - clrIdx);
    scatter(x0, y0, sz, clr, mk, 'LineWidth', 1.5);
end
axis off;  % tweak labels as needed
