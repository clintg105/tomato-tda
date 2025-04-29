
close all, clc

SCRIPT_DIR = fileparts(mfilename('fullpath'));
OUT_DIR    = SCRIPT_DIR;
pairTbl    = readtable(fullfile(OUT_DIR,'ripser_pairs_dists_hat.csv'));

mkrs = {'o','s','v','^','p','h'};
styleLUT = struct( ...
    'bert_strat800',  {mkrs}, ...
    'bow_strat800',   {mkrs}, ...
    'tfidf_strat800', {mkrs});
styleIDs = fieldnames(styleLUT);

clrMap = { ...
    {'#000075','#4363d8','#42d4f4'}; ...
    {'#e6194B','#f58231','#911eb4'}; ...
    {'#808000','#3cb44b','#bfef45'}};
clrCA = cellfun( ...
    @(c)validatecolor(flip(c), 'multiple'), ...
    clrMap,'UniformOutput',false);

pickMarker = @(str) find([ ...
        nnz(str=='_')==0                      , ... % base
        nnz(str=='_')==1 && contains(str,'ws'), ...
        nnz(str=='_')==1 && contains(str,'5'), ...
        nnz(str=='_')==1 && contains(str,'10'), ...
        nnz(str=='_')==2 && contains(str,'5'), ...
        nnz(str=='_')==2 && contains(str,'10')],1);

dimLane = @(m)   ...
        contains(m,'_012') *4 + ...
        (contains(m,'_2') & ~contains(m,'_12'))*3 + ...
        (contains(m,'_1') & ~contains(m,'_01'))*2 + ...
        contains(m,'_0');                     

%% 
% suffixes  = {'_0','_1','_2','_012'};
% laneLbl   = {'0','1','2','0,1,2'};
% 
% varNames   = pairTbl.Properties.VariableNames;
% distPrefix = unique(regexprep(varNames(endsWith(varNames,'_0')),'_0$',''));
% 
% catGrps = unique(pairTbl.cat_grp,'stable');
% 
% lane = @(sfx) find(strcmp(sfx(2:end), {'0','1','2','012'})); 
% 
% for p = 1:numel(distPrefix)
%     pref = distPrefix{p};
% 
%     % global X-limits for this distance family (all cat_grps)
%     vals = [];  
%     for s = 1:numel(suffixes)
%         col = sprintf('%s%s',pref,suffixes{s});
%         if ismember(col,varNames)
%             v  = pairTbl.(col);
%             vals = [vals ; v(isfinite(v) & v>0)];
%         end
%     end
%     if isempty(vals), warning('%s: all NaN',pref), continue, end
%     xLim = 10.^[floor(log10(min(vals)))  ceil(log10(max(vals)))];
% 
%     % one figure per cat_grp
%     for g = 1:numel(catGrps)
%         grp = catGrps{g};
%         T   = pairTbl(strcmp(pairTbl.cat_grp,grp),:);
% 
%         figure('Name',[pref,' – ',grp]), clf, hold on, grid on
%         set(gca,'XScale','log','YLim',[0.5 4.5], ...
%                 'YTick',1:4,'YTickLabel',laneLbl, ...
%                 'FontWeight','bold','FontSize',12)
%         xlabel([upper(pref),'  (distance)'])
%         ylabel('diagram dimension set')
%         title(sprintf('%s ‖ %s',strrep(pref,'_','\_'),strrep(grp,'_','\_')))
% 
%         % —— plot every table row × suffix that has a finite value —
%         for ii = 1:height(T)
%             ds     = T.dataset{ii};
%             if ~isfield(styleLUT,ds), continue, end
%             metric = T.metric{ii};
%             mkIdx  = pickMarker(metric); if isempty(mkIdx), mkIdx=1; end
%             mk     = styleLUT.(ds){mkIdx};
%             clrIdx = 1 + contains(metric,'mp1') + 2*contains(metric,'mpinf');
%             clr    = clrCA{strcmp(styleIDs,ds)}(clrIdx,:);
% 
%             for s = 1:numel(suffixes)
%                 col = sprintf('%s%s',pref,suffixes{s});
%                 if ~ismember(col,varNames), continue, end
%                 val = T.(col)(ii);
%                 if ~isfinite(val) || val==0, continue, end
%                 y = lane(suffixes{s}) + (rand-0.5)*0.6;   % jitter within lane
%                 scatter(val,y,120,clr,mk,'LineWidth',1.5)
%             end
%         end
%         xlim(xLim)
%     end
% end


%% 
distSel = {'wd','bn','landL2','imgL2'};  
suffixes = {'_0','_1','_2','_012'};
laneLbl  = {'0','1','2','0,1,2'};
catGrps  = unique(pairTbl.cat_grp,'stable');

% assert(numel(distSel)==2,'distSel must have exactly two prefixes');

nR = numel(catGrps); nC = numel(distSel);
xLim = nan(nC,2);
for f = 1:nC
    vals = [];
    for sfx = suffixes
        col = sprintf('%s%s',distSel{f},sfx{1});
        if ismember(col,pairTbl.Properties.VariableNames)
            v = pairTbl.(col);
            vals = [vals ; v(isfinite(v) & v>0)];
        end
    end
    if isempty(vals), continue; end
    xLim(f,:) = 10.^[floor(log10(min(vals))), ceil(log10(max(vals)))];
end

f = figure; f.Position([3 4]) = [1175.2,1308.8];
t  = tiledlayout(nR,nC,'TileSpacing','tight','Padding','tight');

for r = 1:nR
    grp = catGrps{r};
    T   = pairTbl(strcmp(pairTbl.cat_grp,grp),:);

    for c = 1:nC
        pref = distSel{c};
        nexttile
        hold on, grid on, box on
        set(gca,'XScale','log','YLim',[0.5 4.5], ...
            'YTick',1:4,'YTickLabel',laneLbl, ...
            'FontSize',10,'FontWeight','bold')
        if r==1
            title(upper(pref));
        end
        % if r~=nR
        %     xticklabels(repmat({''},1,numel(xticks())));
        % end
        if c==1
            ylabel(strrep(grp,'_','\_')); 
        end
        if c~=1
            yticklabels(repmat({''},1,numel(yticks())));
        end
        xlabel('')
        xlim(xLim(c,:))
        yline([1 2 3]+0.5,'-k')

        for ii = 1:height(T)
            ds     = T.dataset{ii};
            if ~isfield(styleLUT,ds), continue, end
            metric = T.metric{ii};
            mkIdx  = pickMarker(metric); if isempty(mkIdx), mkIdx=1; end
            mk     = styleLUT.(ds){mkIdx};
            clrIdx = 1 + contains(metric,'mp1') + 2*contains(metric,'mpinf');
            clr    = clrCA{strcmp(styleIDs,ds)}(clrIdx,:);

            for sfx = suffixes
                col = sprintf('%s%s',pref,sfx{1});
                if ~ismember(col,T.Properties.VariableNames), continue, end
                val = T.(col)(ii);
                if ~isfinite(val) || val==0, continue, end
                y = find(strcmp(sfx{1}(2:end),{'0','1','2','012'})) ...
                    + (rand-0.5)*0.6;
                scatter(val,y,100,clr,mk,'LineWidth',1.4)
            end
        end
        xlim(10.^[-3 3])
    end
end