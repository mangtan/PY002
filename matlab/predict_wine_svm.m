function resultTable = predict_wine_svm(inputData, modelFile, outputCsv)
% MATLAB 预测脚本
% 用法：
%   predict_wine_svm([14.23,1.71,...], '../outputs_matlab/model.mat')
%   predict_wine_svm('../wine/wine_expanded.data', '../outputs_matlab/model.mat', '../outputs_matlab/predict.csv')

if nargin < 2 || isempty(modelFile)
    baseDir = fileparts(mfilename("fullpath"));
    modelFile = fullfile(baseDir, "..", "outputs_matlab", "model.mat");
end
if nargin < 3 || isempty(outputCsv)
    baseDir = fileparts(mfilename("fullpath"));
    outputCsv = fullfile(baseDir, "..", "outputs_matlab", "predict_result.csv");
end

s = load(modelFile, "modelData");
modelData = s.modelData;

X = parseInput(inputData);
if size(X, 2) ~= 13
    error("输入数据必须是 13 列特征。");
end

Xz = (X - modelData.mu) ./ modelData.sigma;
Xz(:, modelData.sigma == 0) = X(:, modelData.sigma == 0);

Xt = applySelector(Xz, modelData.selector);

if string(modelData.modelType) == "libsvm"
    dummy = zeros(size(Xt, 1), 1);
    [pred, ~, prob] = svmpredict(dummy, Xt, modelData.model, "-b 1 -q");
    pred = reshape(pred, [], 1);
    conf = max(prob, [], 2);
else
    [pred, score] = predict(modelData.model, Xt);
    pred = reshape(pred, [], 1);
    conf = softmaxConfidence(score);
end

className = labelsToNames(pred, modelData);
resultTable = table(pred, className, conf, 'VariableNames', {'pred_label', 'class_name', 'confidence'});
if shouldWriteOutput(outputCsv)
    writetable(resultTable, outputCsv);
end
end

function X = parseInput(inputData)
if isnumeric(inputData)
    X = inputData;
    if isvector(X)
        X = reshape(X, 1, []);
    end
    return;
end

if isstring(inputData) || ischar(inputData)
    p = char(inputData);
    if exist(p, "file") == 2
        raw = readmatrix(p, "FileType", "text", "Delimiter", ",");
        if size(raw, 2) == 14
            X = raw(:, 2:end);
        else
            X = raw;
        end
        return;
    end
    vals = str2double(split(string(inputData), ","));
    vals = vals(~isnan(vals));
    if numel(vals) ~= 13
        error("字符串输入必须是 13 个逗号分隔特征值。");
    end
    X = reshape(vals, 1, []);
    return;
end

error("不支持的输入类型。");
end

function Xt = applySelector(Xz, selector)
switch selector.method
    case "RFE"
        Xt = Xz(:, selector.selectedIdx);
    case "ModelImportance"
        Xt = Xz(:, selector.selectedIdx);
    case "PCA"
        Xt = (Xz - selector.pcaMu) * selector.coeff;
    otherwise
        error("未知特征方法: %s", selector.method);
end

if isfield(selector, "useLDA") && selector.useLDA
    error("当前预测脚本未启用 LDA 二次降维路径。");
end
end

function conf = softmaxConfidence(score)
if isempty(score)
    conf = ones(0, 1);
    return;
end
s = score - max(score, [], 2);
ex = exp(s);
prob = ex ./ sum(ex, 2);
conf = max(prob, [], 2);
end

function tf = shouldWriteOutput(outputCsv)
if isempty(outputCsv)
    tf = false;
    return;
end
if isstring(outputCsv)
    tf = all(strlength(outputCsv) > 0);
    return;
end
if ischar(outputCsv)
    tf = ~isempty(strtrim(outputCsv));
    return;
end
tf = true;
end

function className = labelsToNames(pred, modelData)
pred = reshape(pred, [], 1);
className = cell(size(pred, 1), 1);

hasMap = isfield(modelData, "labelValues") && isfield(modelData, "labelNames");
if hasMap
    labels = modelData.labelValues(:);
    if iscell(modelData.labelNames)
        names = modelData.labelNames(:);
    else
        names = cellstr(string(modelData.labelNames(:)));
    end
    for i = 1:numel(pred)
        idx = find(labels == pred(i), 1);
        if ~isempty(idx)
            className{i} = normalizeLabelName(char(string(names{idx})), pred(i));
        else
            className{i} = defaultMappedName(pred(i));
        end
    end
    return;
end

for i = 1:numel(pred)
    className{i} = defaultMappedName(pred(i));
end
end

function out = normalizeLabelName(name, label)
out = normalizeWineClassName(name, label);
end

function name = defaultMappedName(label)
if label == 1
    name = "巴罗洛";
elseif label == 2
    name = "格里尼奥利诺";
elseif label == 3
    name = "桑娇维塞";
elseif label == 4
    name = "内比奥罗";
elseif label == 5
    name = "巴贝拉";
elseif label == 6
    name = "蒙特布查诺";
elseif label == 7
    name = "阿利亚尼科";
elseif label == 8
    name = "普里米蒂沃";
elseif label == 9
    name = "黑达沃拉";
elseif label == 10
    name = "科维纳";
elseif label == 11
    name = "多尔切托";
else
    name = sprintf("第%d类", label);
end
name = char(string(name));
end

function out = normalizeWineClassName(inName, label)
s = strtrim(char(string(inName)));
if isempty(s)
    out = defaultMappedName(label);
    return;
end

if strcmp(s, sprintf("第%d类", label)) || strcmp(s, sprintf("UCI_Class_%d", label)) || ...
        strcmp(s, sprintf("Cultivar %d（UCI原始标签）", label))
    out = defaultMappedName(label);
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
