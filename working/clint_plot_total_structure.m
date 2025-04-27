close all

SCRIPT_DIR = fileparts(mfilename('fullpath'));
OUT_DIR    = SCRIPT_DIR;

totTbl  = readtable(fullfile(OUT_DIR,'tda_summary.csv'));
pairTbl  = readtable(fullfile(OUT_DIR,'ripser_wasserstein_pairs_p2.csv'));


mkrs = {'o','s','v','^','p','h'};
styleLUT = struct( ...
    'bert_strat800',  {mkrs}, ...
    'bow_strat800',   {mkrs}, ...
    'tfidf_strat800', {mkrs} );

styleIDs   = fieldnames(styleLUT);
palette    = lines(numel(styleIDs));                  % one colour / dataset
metricCols = {'base','ws','5nn','10nn','ws 5nn','ws 10nn'};

pickMarker = @(ds,metric) find([ ...
        nnz(ds=='_')==0                                     , ... % base
        nnz(ds=='_')==1 && contains(metric,'ws')            , ...
        nnz(ds=='_')==1 && contains(metric,'5nn')           , ...
        nnz(ds=='_')==1 && contains(metric,'10nn')          , ...
        nnz(ds=='_')==2 && contains(metric,'5nn')           , ...
        nnz(ds=='_')==2 && contains(metric,'10nn')],1,'first');

clrPalletes = {@cool,@abyss ,@winter};
nBaseMetrics = 3;
clrCA = cellfun(@(f)flipud(f(nBaseMetrics)),clrPalletes,'UniformOutput',0);

map = {{'#000075','#4363d8','#42d4f4'};{'#e6194B','#f58231','#911eb4'};{'#808000','#3cb44b','#bfef45'}};
clrCA = cellfun(@(map)validatecolor(flip(map), 'multiple'),map,'UniformOutput',0);

%% 
figure('Name','Total persistence'); hold on; grid on
xlabel('TP_{0}'); ylabel('TP_{1,2}')
axLim = 10.^[0 5 -2.5 5]; axis(axLim);
title('High vs Low-Dim Total Peristance')

sz = 100;

for ii = 1:height(totTbl)
    isZero = 0;
    ds = totTbl.dataset{ii};          
    if ~isfield(styleLUT,ds),  continue, end
    metric = totTbl.metric{ii};

    mkIdx = pickMarker(metric,metric);       
    if isempty(mkIdx), mkIdx = 1; end
    mk = styleLUT.(ds){mkIdx};
    clrIdx = contains(metric,'cos')+2*contains(metric,'mp1')+3*contains(metric,'mpinf');
    clr = clrCA{strcmp(styleIDs,ds)}(clrIdx,:);

    x = totTbl.tp_0(ii);
    y = totTbl.tp_12(ii);
    
    if x == 0, x = axLim(1); end
    if y == 0, y = axLim(3); end
    assert(x >= axLim(1) && x <= axLim(2))
    assert(y >= axLim(3) && y <= axLim(4))
    scatter(x,y, sz, clr, mk, 'LineWidth', 1.5)
end
set(gca,'XScale','log','YScale','log','FontWeight','bold','FontSize',12)

%% 
figure('Name','Total persistence'); hold on; grid on
xlabel('E_{0}(D)'); ylabel('E_{1,2}(D)')
axLim = 10.^[0 5 log10(1.5) log10(10)]; axis(axLim);
title('High vs Low-Dim Persistant Entropy')

sz = 100;

for ii = 1:height(totTbl)
    isZero = 0;
    ds = totTbl.dataset{ii};          
    if ~isfield(styleLUT,ds),  continue, end
    metric = totTbl.metric{ii};

    mkIdx = pickMarker(metric,metric);       
    if isempty(mkIdx), mkIdx = 1; end
    mk = styleLUT.(ds){mkIdx};
    clrIdx = contains(metric,'cos')+2*contains(metric,'mp1')+3*contains(metric,'mpinf');
    clr = clrCA{strcmp(styleIDs,ds)}(clrIdx,:);

    x = totTbl.tp_0(ii);
    y = totTbl.entropy_0(ii);
    y = totTbl.tp_12(ii);
    y = totTbl.entropy_12(ii);
    
    if x == 0, x = axLim(1); end
    if y == 0, y = axLim(3); end
    assert(x >= axLim(1) && x <= axLim(2))
    assert(y >= axLim(3) && y <= axLim(4))
    scatter(x,y, sz, clr, mk, 'LineWidth', 1.5)
end
set(gca,'XScale','log','YScale','log','FontWeight','bold','FontSize',12)
%% 
figure('Name','Total persistence'); hold on; grid on
ylabel('E_{0,1,2}(D)'); xlabel('TP_{0,1,2}')
axLim = 10.^[0.5 5 log10(1.5) log10(10)]; axis(axLim);
title('Full Persistant Entropy vs Total Persistance')

sz = 100;

for ii = 1:height(totTbl)
    isZero = 0;
    ds = totTbl.dataset{ii};          
    if ~isfield(styleLUT,ds),  continue, end
    metric = totTbl.metric{ii};

    mkIdx = pickMarker(metric,metric);       
    if isempty(mkIdx), mkIdx = 1; end
    mk = styleLUT.(ds){mkIdx};
    clrIdx = contains(metric,'cos')+2*contains(metric,'mp1')+3*contains(metric,'mpinf');
    clr = clrCA{strcmp(styleIDs,ds)}(clrIdx,:);

    x = totTbl.tp_012(ii);
    y = totTbl.entropy_012(ii);
    
    if x == 0, x = axLim(1); end
    if y == 0, y = axLim(3); end
    scatter(x,y, sz, clr, mk, 'LineWidth', 1.5)
end
set(gca,'XScale','log','YScale','log','FontWeight','bold','FontSize',12)

%%
nRows = numel(styleIDs); nCols = numel(metricCols);
f= figure('Name','Style grid legend'); hold on; f.Position(3:4) = [501.6000000000004,215.2];
for ii = 1:height(totTbl)
    ds = totTbl.dataset{ii};          
    if ~isfield(styleLUT,ds),  continue, end
    metric = totTbl.metric{ii};

    mkIdx = pickMarker(metric,metric);       
    if isempty(mkIdx), mkIdx = 1; end
    mk = styleLUT.(ds){mkIdx};
    clrIdx = contains(metric,'cos')+2*contains(metric,'mp1')+3*contains(metric,'mpinf');
    clr = clrCA{strcmp(styleIDs,ds)}(clrIdx,:);

    x0 = mkIdx;
    y0 = 3*find(strcmp(styleIDs,ds))+clrIdx;

    scatter(x0, y0, sz, clr, mk, 'LineWidth', 1.5)
end
axis on
y0 = 12.6;
ca = {'HorizontalAlignment','center','VerticalAlignment','bottom','FontWeight','bold'};
text(1,y0,'Base',ca{:})
text(2,y0,'ws',ca{:})
text(3,y0,'5nn',ca{:})
text(4,y0,'10nn',ca{:})
text(5,y0,'ws 5nn',ca{:})
text(6,y0,'ws 10nn',ca{:})

x0 = 0.5;
ca = {'HorizontalAlignment','center','VerticalAlignment','middle','FontWeight','bold'};
text(x0,4,'Cos',ca{:})
text(x0,5,'L1',ca{:})
text(x0,6,'LInf',ca{:})
text(x0,7,'Cos',ca{:})
text(x0,8,'L1',ca{:})
text(x0,9,'LInf',ca{:})
text(x0,10,'Cos',ca{:})
text(x0,11,'L1',ca{:})
text(x0,12,'LInf',ca{:})

x0 = -0.2;
text(x0,6,'BERT',ca{:})
text(x0,9,'BoW',ca{:})
text(x0,12,'TF-IDF',ca{:})

axis([0 6 4 12.5])
axis off
% axis on

