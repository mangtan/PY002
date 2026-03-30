function report = train_wine_svm(dataFile, outputDir)
% 基于 SVM 的意大利葡萄酒种类识别（MATLAB 训练脚本）
% 用法：
%   report = train_wine_svm()
%   report = train_wine_svm('../wine/wine.data', '../outputs_matlab')

rng(42);

if nargin < 1 || isempty(dataFile)
    baseDir = fileparts(mfilename("fullpath"));
    dataFile = fullfile(baseDir, "..", "wine", "wine.data");
end
if nargin < 2 || isempty(outputDir)
    baseDir = fileparts(mfilename("fullpath"));
    outputDir = fullfile(baseDir, "..", "outputs_matlab");
end
if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

raw = readmatrix(dataFile, "FileType", "text", "Delimiter", ",");
y = raw(:, 1);
X = raw(:, 2:end);
labelValues = unique(y)';
[labelNames, labelMap] = buildLabelNames(dataFile, labelValues);

featureNames = { ...
    "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", ...
    "Magnesium", "Total phenols", "Flavanoids", ...
    "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", ...
    "Hue", "OD280_OD315", "Proline" ...
};
featureNamesZh = { ...
    "酒精", "苹果酸", "灰分", "灰分碱度", "镁", "总酚", ...
    "黄酮", "非黄烷类酚", "原花青素", "色度", "色调", ...
    "稀释葡萄酒OD280/OD315", "脯氨酸" ...
};

[Xz, mu, sigma] = zscore(X);
sigma(sigma == 0) = 1;

nSamples = size(Xz, 1);
nFeatures = size(Xz, 2);

% 1) 特征方法对比：RFE / PCA / 模型重要性
cv10 = cvpartition(y, "KFold", 10);
featureResults = struct("method", {}, "bestScore", {}, "selectedDim", {}, "selectedIdx", {});

% RFE（线性SVM权重递归消除）
rfeRank = computeRFERanking(Xz, y);
[rfeBestScore, rfeBestDim] = searchBestDimByCV(Xz, y, rfeRank, cv10);
featureResults(end+1) = struct( ...
    "method", "RFE", ...
    "bestScore", rfeBestScore, ...
    "selectedDim", rfeBestDim, ...
    "selectedIdx", rfeRank(1:rfeBestDim) ...
);

% PCA
[coeffPCA, scorePCA, ~, ~, ~, pcaMu] = pca(Xz);
[pcaBestScore, pcaBestDim] = searchBestPCADimByCV(Xz, y, cv10);
featureResults(end+1) = struct( ...
    "method", "PCA", ...
    "bestScore", pcaBestScore, ...
    "selectedDim", pcaBestDim, ...
    "selectedIdx", 1:pcaBestDim ...
);

% 模型重要性（TreeBagger OOB importance）
impRank = computeImportanceRanking(Xz, y);
[impBestScore, impBestDim] = searchBestDimByCV(Xz, y, impRank, cv10);
featureResults(end+1) = struct( ...
    "method", "ModelImportance", ...
    "bestScore", impBestScore, ...
    "selectedDim", impBestDim, ...
    "selectedIdx", impRank(1:impBestDim) ...
);

[bestFeature, featureBarFig] = pickBestFeatureMethod(featureResults);
saveFigureHD(featureBarFig, fullfile(outputDir, "feature_method_compare.png"));
close(featureBarFig);

% 构造最终特征变换配置
selector = struct();
selector.method = bestFeature.method;
selector.selectedDim = bestFeature.selectedDim;
selector.useLDA = false;

switch selector.method
    case "RFE"
        selector.selectedIdx = rfeRank(1:selector.selectedDim);
        Xsel = Xz(:, selector.selectedIdx);
    case "ModelImportance"
        selector.selectedIdx = impRank(1:selector.selectedDim);
        Xsel = Xz(:, selector.selectedIdx);
    case "PCA"
        selector.coeff = coeffPCA(:, 1:selector.selectedDim);
        selector.pcaMu = pcaMu;
        Xsel = scorePCA(:, 1:selector.selectedDim);
    otherwise
        error("未知特征方法: %s", selector.method);
end

% 开题要求：若维度 > 20，进一步 LDA（当前 UCI Wine 通常不会触发）
if size(Xsel, 2) > 20
    selector.useLDA = true;
    ldaModel = fitcdiscr(Xsel, y, "DiscrimType", "linear");
    % 对于 K 类，LDA 最多 K-1 维。这里通过 transform 获得降维空间
    [~, ldaScore] = resubPredict(ldaModel);
    selector.ldaModel = ldaModel;
    Xsel = ldaScore;
end

% 2) 核函数与超参数对比（网格搜索 + 5折）
kernelResults = struct("kernel", {}, "bestScore", {}, "bestParams", {});
cv5 = cvpartition(y, "KFold", 5);
CGrid = logspace(-3, 3, 7);
gammaRbfGrid = logspace(-4, 2, 7);
gammaSigGrid = logspace(-3, 0, 7);
coef0Grid = linspace(0, 3, 7);

% Linear
[bestLinearScore, bestLinearC] = searchLinearKernel(Xsel, y, cv5, CGrid);
kernelResults(end+1) = struct("kernel", "linear", "bestScore", bestLinearScore, ...
    "bestParams", struct("C", bestLinearC));

% RBF
[bestRbfScore, bestRbfC, bestRbfGamma] = searchRbfKernel(Xsel, y, cv5, CGrid, gammaRbfGrid);
kernelResults(end+1) = struct("kernel", "rbf", "bestScore", bestRbfScore, ...
    "bestParams", struct("C", bestRbfC, "gamma", bestRbfGamma));

% Sigmoid（优先 libsvm；若无则标记不可用）
hasLibsvm = ensureLibsvm();
if hasLibsvm
    [bestSigScore, bestSigC, bestSigGamma, bestSigCoef0] = ...
        searchSigmoidKernelLibsvm(Xsel, y, CGrid, gammaSigGrid, coef0Grid);
    kernelResults(end+1) = struct("kernel", "sigmoid", "bestScore", bestSigScore, ...
        "bestParams", struct("C", bestSigC, "gamma", bestSigGamma, "coef0", bestSigCoef0));
else
    kernelResults(end+1) = struct("kernel", "sigmoid", "bestScore", NaN, ...
        "bestParams", struct("available", false, "reason", "libsvm not found"));
end

[bestKernel, kernelBarFig] = pickBestKernel(kernelResults);
saveFigureHD(kernelBarFig, fullfile(outputDir, "kernel_compare.png"));
close(kernelBarFig);

% 3) 最终评估策略
if nSamples < 50
    evalStrategy = "LOOCV";
elseif nSamples <= 200
    evalStrategy = "Repeated10x10CV";
else
    evalStrategy = "Stratified10Fold";
end

[predAll, acc] = evaluateFinal(Xsel, y, bestKernel, evalStrategy);
cm = confusionmat(y, predAll);

cmFig = figure("Visible", "off");
imagesc(cm);
applyChineseStyle(cmFig);
title("混淆矩阵");
xlabel("预测类别");
ylabel("真实类别");
colorbar;
axis tight;
xticks(1:numel(labelValues));
yticks(1:numel(labelValues));
shortNames = toShortLabelNames(labelNames);
xticklabels(shortNames);
yticklabels(shortNames);
xtickangle(35);
for i = 1:size(cm, 1)
    for j = 1:size(cm, 2)
        if cm(i, j) > 0
            text(j, i, num2str(cm(i, j)), "HorizontalAlignment", "center", "FontSize", 10);
        end
    end
end
saveFigureHD(cmFig, fullfile(outputDir, "confusion_matrix.png"));
close(cmFig);

% 4) 训练最终模型并保存
modelData = struct();
modelData.featureNames = featureNames;
modelData.featureNamesZh = featureNamesZh;
modelData.mu = mu;
modelData.sigma = sigma;
modelData.selector = selector;
modelData.bestKernel = bestKernel;
modelData.evalStrategy = evalStrategy;
modelData.labelValues = labelValues;
modelData.labelNames = labelNames;
modelData.labelMap = labelMap;
classExamples = buildClassExamples(X, y, labelValues, labelNames);
modelData.classExamples = classExamples;
modelData.featureStats = struct( ...
    "min", min(X, [], 1), ...
    "max", max(X, [], 1), ...
    "mean", mean(X, 1) ...
);

if bestKernel.kernel == "sigmoid" && hasLibsvm
    cmd = sprintf("-s 0 -t 3 -c %g -g %g -r %g -b 1 -q", ...
        bestKernel.bestParams.C, bestKernel.bestParams.gamma, bestKernel.bestParams.coef0);
    modelData.modelType = "libsvm";
    modelData.model = svmtrainQuiet(y, Xsel, cmd);
    if isempty(modelData.model)
        modelData.model = svmtrain(y, Xsel, cmd);
    end
else
    modelData.modelType = "fitcecoc";
    if bestKernel.kernel == "rbf"
        kernelScale = 1 / sqrt(2 * bestKernel.bestParams.gamma);
        t = templateSVM( ...
            "KernelFunction", "gaussian", ...
            "KernelScale", kernelScale, ...
            "BoxConstraint", bestKernel.bestParams.C, ...
            "Standardize", false);
    elseif bestKernel.kernel == "linear"
        t = templateSVM( ...
            "KernelFunction", "linear", ...
            "BoxConstraint", bestKernel.bestParams.C, ...
            "Standardize", false);
    else
        t = templateSVM( ...
            "KernelFunction", char(bestKernel.kernel), ...
            "BoxConstraint", bestKernel.bestParams.C, ...
            "Standardize", false);
    end
    modelData.model = fitcecoc(Xsel, y, "Learners", t, "Coding", "onevsone");
end

modelPath = fullfile(outputDir, "model.mat");
save(modelPath, "modelData");

report = struct();
report.data.path = string(dataFile);
report.data.nSamples = nSamples;
report.data.nFeatures = nFeatures;
report.featureSelection.allMethods = featureResults;
report.featureSelection.bestMethod = bestFeature;
report.kernelSearch.allKernels = kernelResults;
report.kernelSearch.bestKernel = bestKernel;
report.finalEvaluation.strategy = evalStrategy;
report.finalEvaluation.accuracy = acc;
report.finalEvaluation.confusionMatrix = cm;
report.labelMap = labelMap;
report.artifacts.model = string(modelPath);
report.artifacts.featureBar = string(fullfile(outputDir, "feature_method_compare.png"));
report.artifacts.kernelBar = string(fullfile(outputDir, "kernel_compare.png"));
report.artifacts.confusionMatrix = string(fullfile(outputDir, "confusion_matrix.png"));
examplesPath = fullfile(outputDir, "class_examples.csv");
writeClassExamplesCsv(classExamples, featureNames, examplesPath);
report.artifacts.classExamples = string(examplesPath);

jsonPath = fullfile(outputDir, "report.json");
fid = fopen(jsonPath, "w");
fwrite(fid, jsonencode(report, "PrettyPrint", true), "char");
fclose(fid);

fprintf("训练完成，模型已保存: %s\n", modelPath);
fprintf("评估准确率 (%s): %.4f\n", evalStrategy, acc);
fprintf("报告文件: %s\n", jsonPath);
end

function hasLibsvm = ensureLibsvm()
hasLibsvm = hasLibsvmMexLocal();
if hasLibsvm
    return;
end

if exist("setup_libsvm", "file") == 2
    try
        hasLibsvm = setup_libsvm();
    catch
        hasLibsvm = false;
    end
    if hasLibsvm
        return;
    end
end

baseDir = fileparts(mfilename("fullpath"));
cand = {
    fullfile(baseDir, "..", "..", "third_party", "libsvm", "matlab"), ...
    fullfile(baseDir, "..", "third_party", "libsvm", "matlab")
};

for i = 1:numel(cand)
    d = cand{i};
    if exist(d, "dir") ~= 7
        continue;
    end

    addpath(d);
    rehash;
    hasLibsvm = hasLibsvmMexLocal();
    if hasLibsvm
        return;
    end

    % 尝试自动编译 mex（兼容老逻辑，优先调用 setup_libsvm）
    try
        if exist("setup_libsvm", "file") == 2
            hasLibsvm = setup_libsvm();
        else
            old = pwd;
            cd(d);
            if exist("make", "file") == 2
                make;
            end
            cd(old);
            hasLibsvm = hasLibsvmMexLocal();
        end
    catch
        if exist('old', 'var') == 1
            try
                cd(old);
            catch
            end
        end
    end

    rehash;
    hasLibsvm = hasLibsvmMexLocal();
    if hasLibsvm
        return;
    end
end
end

function tf = hasLibsvmMexLocal()
p1 = which('svmtrain');
p2 = which('svmpredict');
tf = ~isempty(p1) && ~isempty(p2);
if tf
    tf = endsWith(p1, ['.', mexext]) && endsWith(p2, ['.', mexext]);
end
end

function classExamples = buildClassExamples(X, y, labelValues, labelNames)
classExamples = struct("label", {}, "name", {}, "features", {});
for i = 1:numel(labelValues)
    c = labelValues(i);
    Xi = X(y == c, :);
    if isempty(Xi)
        continue;
    end
    mu = mean(Xi, 1);
    d = sum((Xi - mu) .^ 2, 2);
    [~, idx] = min(d);
    classExamples(end+1) = struct( ... %#ok<AGROW>
        "label", c, ...
        "name", char(string(labelNames{i})), ...
        "features", Xi(idx, :) ...
    );
end
end

function writeClassExamplesCsv(classExamples, featureNames, outputPath)
n = numel(classExamples);
if n == 0
    return;
end

labelCol = zeros(n, 1);
nameCol = cell(n, 1);
feat = zeros(n, numel(featureNames));
for i = 1:n
    labelCol(i) = classExamples(i).label;
    nameCol{i} = char(string(classExamples(i).name));
    feat(i, :) = classExamples(i).features;
end

tbl = table(labelCol, nameCol, 'VariableNames', {'label', 'class_name'});
for j = 1:numel(featureNames)
    colName = matlab.lang.makeValidName(char(string(featureNames{j})));
    tbl.(colName) = feat(:, j);
end
writetable(tbl, outputPath);
end

function [labelNames, labelMap] = buildLabelNames(dataFile, labelValues)
labelNames = defaultLabelNames(labelValues);

mapFile = fullfile(fileparts(dataFile), "label_map.json");
if exist(mapFile, "file") == 2
    try
        m = jsondecode(fileread(mapFile));
        for i = 1:numel(labelValues)
            val = tryGetLabelMapValue(m, labelValues(i));
            if ~isempty(val)
                labelNames{i} = normalizeWineClassName(char(string(val)), labelValues(i));
            end
        end
    catch
        % ignore map parse failure and keep default names
    end
end

for i = 1:numel(labelValues)
    labelNames{i} = normalizeWineClassName(labelNames{i}, labelValues(i));
end
labelMap = struct("label", num2cell(labelValues), "name", labelNames);
end

function val = tryGetLabelMapValue(m, label)
val = [];
if ~isstruct(m)
    return;
end

key = sprintf("%d", label);
alt1 = ['x', key]; % matlab jsondecode for numeric keys
fields = fieldnames(m);

if isfield(m, key)
    val = m.(key);
    return;
end
if isfield(m, alt1)
    val = m.(alt1);
    return;
end

% fallback: match trailing digits in field names
for i = 1:numel(fields)
    fn = fields{i};
    tok = regexp(fn, '(\d+)$', 'tokens', 'once');
    if ~isempty(tok) && str2double(tok{1}) == label
        val = m.(fn);
        return;
    end
end
end

function labelNames = defaultLabelNames(labelValues)
labelNames = cell(1, numel(labelValues));
for i = 1:numel(labelValues)
    v = labelValues(i);
    labelNames{i} = defaultMappedNameZh(v);
end
end

function out = normalizeWineClassName(inName, label)
s = strtrim(char(string(inName)));
if isempty(s)
    out = defaultMappedNameZh(label);
    return;
end

if strcmp(s, sprintf("第%d类", label)) || ...
        strcmp(s, sprintf("UCI_Class_%d", label)) || ...
        strcmp(s, sprintf("Cultivar %d（UCI原始标签）", label))
    out = defaultMappedNameZh(label);
    return;
end

switch lower(strrep(s, '''', ''))
    case "barolo"
        out = "巴罗洛";
    case "grignolino"
        out = "格里尼奥利诺";
    case "barbera"
        out = "巴贝拉";
    case "sangiovese"
        out = "桑娇维塞";
    case "nebbiolo"
        out = "内比奥罗";
    case "montepulciano"
        out = "蒙特布查诺";
    case "aglianico"
        out = "阿利亚尼科";
    case "primitivo"
        out = "普里米蒂沃";
    case {"nero davola", "nero daavola"}
        out = "黑达沃拉";
    case "corvina"
        out = "科维纳";
    case "dolcetto"
        out = "多尔切托";
    otherwise
        out = s;
end

out = char(string(out));
end

function name = defaultMappedNameZh(label)
if label == 1
    name = "巴罗洛";
elseif label == 2
    name = "格里尼奥利诺";
elseif label == 3
    name = "巴贝拉";
elseif label == 4
    name = "桑娇维塞";
elseif label == 5
    name = "内比奥罗";
elseif label == 6
    name = "巴贝拉";
elseif label == 7
    name = "蒙特布查诺";
elseif label == 8
    name = "阿利亚尼科";
elseif label == 9
    name = "普里米蒂沃";
elseif label == 10
    name = "黑达沃拉";
elseif label == 11
    name = "科维纳";
elseif label == 12
    name = "多尔切托";
else
    name = sprintf("第%d类", label);
end
name = char(string(name));
end

function rankIdx = computeRFERanking(X, y)
p = size(X, 2);
remain = 1:p;
removed = [];

while numel(remain) > 1
    t = templateSVM("KernelFunction", "linear", "BoxConstraint", 1, "Standardize", false);
    mdl = fitcecoc(X(:, remain), y, "Learners", t, "Coding", "onevsone");

    w = zeros(1, numel(remain));
    for i = 1:numel(mdl.BinaryLearners)
        bl = mdl.BinaryLearners{i};
        if isprop(bl, "Beta") && ~isempty(bl.Beta)
            w = w + abs(bl.Beta(:))';
        else
            w = w + eps;
        end
    end
    [~, idxMin] = min(w);
    removed(end+1) = remain(idxMin); %#ok<AGROW>
    remain(idxMin) = [];
end

rankIdx = [remain, fliplr(removed)];
end

function rankIdx = computeImportanceRanking(X, y)
bag = TreeBagger(200, X, y, ...
    "Method", "classification", ...
    "OOBPrediction", "on", ...
    "OOBPredictorImportance", "on");
imp = bag.OOBPermutedPredictorDeltaError;
[~, rankIdx] = sort(imp, "descend");
end

function [bestScore, bestDim] = searchBestDimByCV(X, y, rankIdx, cvp)
p = size(X, 2);
bestScore = -inf;
bestDim = 2;
for k = 2:p
    Xk = X(:, rankIdx(1:k));
    acc = linearCVAccuracy(Xk, y, cvp);
    if acc > bestScore || (abs(acc - bestScore) < 1e-9 && k < bestDim)
        bestScore = acc;
        bestDim = k;
    end
end
end

function [bestScore, bestDim] = searchBestPCADimByCV(X, y, cvp)
p = size(X, 2);
[coeff, ~, ~, ~, ~, pcaMu] = pca(X);
bestScore = -inf;
bestDim = 2;
for k = 2:p
    Xk = (X - pcaMu) * coeff(:, 1:k);
    acc = linearCVAccuracy(Xk, y, cvp);
    if acc > bestScore || (abs(acc - bestScore) < 1e-9 && k < bestDim)
        bestScore = acc;
        bestDim = k;
    end
end
end

function acc = linearCVAccuracy(X, y, cvp)
t = templateSVM("KernelFunction", "linear", "BoxConstraint", 1, "Standardize", false);
mdl = fitcecoc(X, y, "Learners", t, "Coding", "onevsone", "CVPartition", cvp);
loss = kfoldLoss(mdl);
acc = 1 - loss;
end

function [bestScore, bestC] = searchLinearKernel(X, y, cvp, CGrid)
bestScore = -inf;
bestC = CGrid(1);
for c = CGrid
    t = templateSVM("KernelFunction", "linear", "BoxConstraint", c, "Standardize", false);
    mdl = fitcecoc(X, y, "Learners", t, "Coding", "onevsone", "CVPartition", cvp);
    score = 1 - kfoldLoss(mdl);
    if score > bestScore
        bestScore = score;
        bestC = c;
    end
end
end

function [bestScore, bestC, bestGamma] = searchRbfKernel(X, y, cvp, CGrid, gammaGrid)
bestScore = -inf;
bestC = CGrid(1);
bestGamma = gammaGrid(1);
for c = CGrid
    for g = gammaGrid
        kernelScale = 1 / sqrt(2 * g);
        t = templateSVM("KernelFunction", "gaussian", ...
            "KernelScale", kernelScale, ...
            "BoxConstraint", c, ...
            "Standardize", false);
        mdl = fitcecoc(X, y, "Learners", t, "Coding", "onevsone", "CVPartition", cvp);
        score = 1 - kfoldLoss(mdl);
        if score > bestScore
            bestScore = score;
            bestC = c;
            bestGamma = g;
        end
    end
end
end

function [bestScore, bestC, bestGamma, bestCoef0] = searchSigmoidKernelLibsvm(X, y, CGrid, gammaGrid, coef0Grid)
bestScore = -inf;
bestC = CGrid(1);
bestGamma = gammaGrid(1);
bestCoef0 = coef0Grid(1);
for c = CGrid
    for g = gammaGrid
        for r = coef0Grid
            cmd = sprintf("-s 0 -t 3 -c %g -g %g -r %g -v 5 -q", c, g, r);
            accRaw = svmtrainQuiet(y, X, cmd);
            score = normalizeLibsvmScore(accRaw);
            if isnan(score)
                % 兼容部分环境下 -v 返回模型而不是数值的情况：回退到手动5折
                score = manualLibsvmCvScore(X, y, c, g, r, 5);
            end
            if score > bestScore
                bestScore = score;
                bestC = c;
                bestGamma = g;
                bestCoef0 = r;
            end
        end
    end
end
end

function out = svmtrainQuiet(yIn, XIn, cmdIn)
try
    [~, out] = evalc('svmtrain(yIn, XIn, cmdIn);');
catch
    out = [];
end
end

function pred = svmpredictQuiet(yIn, XIn, modelIn, cmdIn)
try
    [~, pred] = evalc('svmpredict(yIn, XIn, modelIn, cmdIn);');
catch
    pred = [];
end
end

function score = normalizeLibsvmScore(accRaw)
score = NaN;
if isnumeric(accRaw) || islogical(accRaw)
    if isempty(accRaw)
        return;
    end
    v = double(accRaw(1));
    if ~isfinite(v)
        return;
    end
    if v > 1
        score = v / 100.0;
    else
        score = v;
    end
end
end

function score = manualLibsvmCvScore(X, y, c, g, r, k)
cvp = cvpartition(y, "KFold", k);
total = 0;
correct = 0;
cmdTrain = sprintf("-s 0 -t 3 -c %g -g %g -r %g -q", c, g, r);
for i = 1:cvp.NumTestSets
    tr = training(cvp, i);
    te = test(cvp, i);
    mdl = svmtrainQuiet(y(tr), X(tr, :), cmdTrain);
    if isempty(mdl)
        score = NaN;
        return;
    end
    yy = y(te);
    pred = svmpredictQuiet(yy, X(te, :), mdl, "-q");
    if isempty(pred)
        score = NaN;
        return;
    end
    pred = reshape(pred, [], 1);
    yy = reshape(yy, [], 1);
    n = min(numel(pred), numel(yy));
    correct = correct + sum(pred(1:n) == yy(1:n));
    total = total + n;
end
score = correct / max(total, 1);
end

function [bestFeature, fig] = pickBestFeatureMethod(featureResults)
scores = arrayfun(@(x) x.bestScore, featureResults);
dims = arrayfun(@(x) x.selectedDim, featureResults);
methods = arrayfun(@(x) x.method, featureResults, "UniformOutput", false);

[~, idx] = sortrows([-(scores(:)), dims(:)], [1, 2]);
bestFeature = featureResults(idx(1));

fig = figure("Visible", "off", "Position", [100 100 1200 700]);
applyChineseStyle(fig);
bar(scores);
set(gca, "XTickLabel", mapFeatureMethodNames(methods), "FontSize", 12);
ylim([0, 1]);
ylabel("准确率");
title("特征方法对比（10折交叉验证）");
grid on;
end

function [bestKernel, fig] = pickBestKernel(kernelResults)
validMask = arrayfun(@(x) ~isnan(x.bestScore), kernelResults);
valid = kernelResults(validMask);
scores = arrayfun(@(x) x.bestScore, valid);
[~, idx] = max(scores);
bestKernel = valid(idx);

kernelNames = arrayfun(@(x) x.kernel, kernelResults, "UniformOutput", false);
plotScores = arrayfun(@(x) x.bestScore, kernelResults);
plotScores(isnan(plotScores)) = 0;

fig = figure("Visible", "off", "Position", [100 100 1200 700]);
applyChineseStyle(fig);
bar(plotScores);
set(gca, "XTickLabel", mapKernelNames(kernelNames), "FontSize", 12);
ylim([0, 1]);
ylabel("准确率");
title("核函数对比（网格搜索+5折交叉验证）");
grid on;
end

function [predAll, acc] = evaluateFinal(X, y, bestKernel, strategy)
classes = unique(y)';
n = numel(y);
predAll = zeros(n, 1);

switch strategy
    case "LOOCV"
        cvp = cvpartition(n, "LeaveOut");
        for i = 1:cvp.NumTestSets
            tr = training(cvp, i);
            te = test(cvp, i);
            mdl = trainOne(X(tr, :), y(tr), bestKernel);
            predAll(te) = predictOne(mdl, X(te, :), bestKernel);
        end
    case "Repeated10x10CV"
        voteCount = zeros(n, numel(classes));
        for rep = 1:10
            cvp = cvpartition(y, "KFold", 10);
            for i = 1:cvp.NumTestSets
                tr = training(cvp, i);
                te = test(cvp, i);
                mdl = trainOne(X(tr, :), y(tr), bestKernel);
                pred = predictOne(mdl, X(te, :), bestKernel);
                teIdx = find(te);
                for r = 1:numel(pred)
                    clsIdx = find(classes == pred(r), 1);
                    voteCount(teIdx(r), clsIdx) = voteCount(teIdx(r), clsIdx) + 1;
                end
            end
        end
        [~, maxIdx] = max(voteCount, [], 2);
        predAll = classes(maxIdx)';
    otherwise % Stratified10Fold
        cvp = cvpartition(y, "KFold", 10);
        for i = 1:cvp.NumTestSets
            tr = training(cvp, i);
            te = test(cvp, i);
            mdl = trainOne(X(tr, :), y(tr), bestKernel);
            predAll(te) = predictOne(mdl, X(te, :), bestKernel);
        end
end

acc = mean(predAll == y);
end

function mdl = trainOne(Xtr, ytr, bestKernel)
if bestKernel.kernel == "sigmoid" && isfield(bestKernel.bestParams, "gamma")
    cmd = sprintf("-s 0 -t 3 -c %g -g %g -r %g -b 0 -q", ...
        bestKernel.bestParams.C, bestKernel.bestParams.gamma, bestKernel.bestParams.coef0);
    mdl.type = "libsvm";
    mdl.obj = svmtrainQuiet(ytr, Xtr, cmd);
    if isempty(mdl.obj)
        mdl.obj = svmtrain(ytr, Xtr, cmd);
    end
    return;
end

if bestKernel.kernel == "rbf"
    kernelScale = 1 / sqrt(2 * bestKernel.bestParams.gamma);
    t = templateSVM("KernelFunction", "gaussian", ...
        "KernelScale", kernelScale, ...
        "BoxConstraint", bestKernel.bestParams.C, ...
        "Standardize", false);
elseif bestKernel.kernel == "linear"
    t = templateSVM("KernelFunction", "linear", ...
        "BoxConstraint", bestKernel.bestParams.C, ...
        "Standardize", false);
else
    t = templateSVM("KernelFunction", char(bestKernel.kernel), ...
        "BoxConstraint", bestKernel.bestParams.C, ...
        "Standardize", false);
end
mdl.type = "fitcecoc";
mdl.obj = fitcecoc(Xtr, ytr, "Learners", t, "Coding", "onevsone");
end

function pred = predictOne(mdl, Xte, bestKernel)
if mdl.type == "libsvm"
    dummy = zeros(size(Xte, 1), 1);
    pred = svmpredictQuiet(dummy, Xte, mdl.obj, "-q");
    if isempty(pred)
        pred = svmpredict(dummy, Xte, mdl.obj, "-q");
    end
else
    pred = predict(mdl.obj, Xte);
end
pred = reshape(pred, [], 1);
if bestKernel.kernel == "sigmoid"
    pred = round(pred);
end
end

function out = mapFeatureMethodNames(in)
out = in;
for i = 1:numel(in)
    s = char(string(in{i}));
    if strcmpi(s, "RFE")
        out{i} = "RFE";
    elseif strcmpi(s, "PCA")
        out{i} = "PCA";
    elseif strcmpi(s, "ModelImportance")
        out{i} = "模型重要性";
    else
        out{i} = s;
    end
end
end

function out = mapKernelNames(in)
out = in;
for i = 1:numel(in)
    s = char(string(in{i}));
    if strcmpi(s, "linear")
        out{i} = "线性核";
    elseif strcmpi(s, "rbf")
        out{i} = "RBF核";
    elseif strcmpi(s, "sigmoid")
        out{i} = "Sigmoid核";
    else
        out{i} = s;
    end
end
end

function out = toShortLabelNames(labelNames)
out = cell(size(labelNames));
for i = 1:numel(labelNames)
    s = char(string(labelNames{i}));
    s = strrep(s, "（UCI原始标签）", "");
    out{i} = s;
end
end

function applyChineseStyle(fig)
try
    set(fig, "Color", "w");
    set(findall(fig, "-property", "FontName"), "FontName", "PingFang SC");
    set(findall(fig, "-property", "FontSize"), "FontSize", 12);
catch
    % keep default style if font setting fails
end
end

function saveFigureHD(fig, outPath)
try
    exportgraphics(fig, outPath, "Resolution", 360);
catch
    print(fig, outPath, "-dpng", "-r360");
end
end
