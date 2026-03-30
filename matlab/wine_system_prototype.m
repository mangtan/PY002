function wine_system_prototype()
% 用户友好版 GUI：
% 1) 13个特征逐项填写（中文+英文）
% 2) 结果显示：类别编号 + 类别名称 + 置信度
% 3) 展示类别映射与特征说明

baseDir = fileparts(mfilename("fullpath"));
expandedModel = fullfile(baseDir, "..", "outputs_matlab_expanded", "model.mat");
baseModel = fullfile(baseDir, "..", "outputs_matlab", "model.mat");
if exist(expandedModel, "file") == 2
    defaultModel = expandedModel;
    defaultOut = fullfile(baseDir, "..", "outputs_matlab_expanded", "predict_result.csv");
else
    defaultModel = baseModel;
    defaultOut = fullfile(baseDir, "..", "outputs_matlab", "predict_result.csv");
end

[featureEn, featureZh] = defaultFeatureMeta();
sample = [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065];

fig = uifigure("Name", "意大利葡萄酒种类识别系统原型", ...
    "Position", [80 20 1360 930]);

% 顶部：模型文件
uilabel(fig, "Text", "模型文件", "Position", [20 878 80 24], "FontWeight", "bold");
modelField = uieditfield(fig, "text", "Position", [95 878 930 24], "Value", defaultModel);
uibutton(fig, "Text", "浏览", "Position", [1040 878 90 24], ...
    "ButtonPushedFcn", @(~, ~) chooseModel(fig));
uibutton(fig, "Text", "刷新模型信息", "Position", [1140 878 120 24], ...
    "ButtonPushedFcn", @(~, ~) refreshModelInfo(fig));

% 左侧：单样本输入
inputPanel = uipanel(fig, "Title", "单样本输入（逐项填写）", ...
    "Position", [20 170 620 700], "FontWeight", "bold");

inputLabels = gobjects(13, 1);
inputFields = gobjects(13, 1);
for i = 1:13
    col = floor((i - 1) / 7);      % 左7项 + 右6项
    row = mod(i - 1, 7);
    xBase = 18 + col * 300;
    y = 550 - row * 72;

    labelText = sprintf("%d. %s (%s)", i, featureZh{i}, featureEn{i});
    inputLabels(i) = uilabel(inputPanel, "Text", labelText, "Position", [xBase y 280 22]);
    inputFields(i) = uieditfield(inputPanel, "numeric", ...
        "Position", [xBase y - 28 120 26], "Value", sample(i));
end

uibutton(inputPanel, "Text", "填充示例", "Position", [18 20 90 30], ...
    "ButtonPushedFcn", @(~, ~) fillSample(fig));
uibutton(inputPanel, "Text", "清空输入", "Position", [118 20 90 30], ...
    "ButtonPushedFcn", @(~, ~) clearInputs(fig));
uibutton(inputPanel, "Text", "单样本预测", "Position", [218 20 120 30], ...
    "ButtonPushedFcn", @(~, ~) runSinglePredict(fig));
uibutton(inputPanel, "Text", "批量CSV预测", "Position", [348 20 120 30], ...
    "ButtonPushedFcn", @(~, ~) runBatchPredict(fig));
uilabel(inputPanel, "Text", "示例类别", "Position", [472 26 60 22]);
exampleDropdown = uidropdown(inputPanel, "Position", [536 20 158 30], ...
    "Items", {'1 - 巴罗洛'});

% 右上：预测结果
resultPanel = uipanel(fig, "Title", "预测结果", ...
    "Position", [660 700 680 170], "FontWeight", "bold");
uilabel(resultPanel, "Text", "类别编号：", "Position", [20 90 90 26], "FontWeight", "bold");
predValueLabel = uilabel(resultPanel, "Text", "-", "Position", [110 90 220 26], "FontSize", 16);

uilabel(resultPanel, "Text", "类别名称：", "Position", [20 55 90 26], "FontWeight", "bold");
nameValueLabel = uilabel(resultPanel, "Text", "-", "Position", [110 55 520 26], "FontSize", 14);

uilabel(resultPanel, "Text", "置信度：", "Position", [20 20 90 26], "FontWeight", "bold");
confValueLabel = uilabel(resultPanel, "Text", "-", "Position", [110 20 220 26], "FontSize", 16);

% 右中：类别映射
mapPanel = uipanel(fig, "Title", "类别映射（模型标签 -> 名称）", ...
    "Position", [660 430 680 260], "FontWeight", "bold");
classMapTable = uitable(mapPanel, ...
    "Position", [10 10 660 220], ...
    "ColumnName", {"类别编号", "类别名称"}, ...
    "ColumnWidth", {120, 520}, ...
    "RowName", [], ...
    "Data", cell(0, 2));

% 右下：特征说明
featurePanel = uipanel(fig, "Title", "特征说明", ...
    "Position", [660 170 680 250], "FontWeight", "bold");
featureInfoTable = uitable(featurePanel, ...
    "Position", [10 10 660 210], ...
    "ColumnName", {"序号", "中文特征", "英文特征", "数据范围"}, ...
    "ColumnWidth", {60, 170, 230, 190}, ...
    "RowName", [], ...
    "Data", cell(13, 4));

% 底部：日志
uilabel(fig, "Text", "运行日志", "Position", [20 145 100 24], "FontWeight", "bold");
logArea = uitextarea(fig, "Position", [20 20 1320 125], "Editable", "off");
logArea.Value = ["系统已启动。"; "请确认 model.mat 已训练生成。"];

app = struct();
app.featureEn = featureEn;
app.featureZh = featureZh;
app.sample = sample;
app.defaultOut = defaultOut;
app.modelField = modelField;
app.inputLabels = inputLabels;
app.inputFields = inputFields;
app.predValueLabel = predValueLabel;
app.nameValueLabel = nameValueLabel;
app.confValueLabel = confValueLabel;
app.classMapTable = classMapTable;
app.featureInfoTable = featureInfoTable;
app.logArea = logArea;
app.exampleDropdown = exampleDropdown;
app.exampleLabelVals = 1;
app.classExamples = struct("label", {}, "name", {}, "features", {});
setappdata(fig, "app", app);

refreshModelInfo(fig);
end

function runSinglePredict(fig)
app = getappdata(fig, "app");
modelPath = app.modelField.Value;

if exist(modelPath, "file") ~= 2
    appendLog(fig, "单样本预测失败: 模型文件不存在。");
    return;
end

x = zeros(1, 13);
for i = 1:13
    v = app.inputFields(i).Value;
    if isnan(v)
        appendLog(fig, sprintf("单样本预测失败: 第%d项特征为空。", i));
        return;
    end
    x(i) = v;
end

try
    tbl = predict_wine_svm(x, modelPath, "");
    predVal = getFieldOrIndex(tbl, "pred_label", 1);
    className = getFieldOrIndex(tbl, "class_name", 2);
    confVal = getFieldOrIndex(tbl, "confidence", min(3, width(tbl)));
    className = resolveClassNameFromMap(app, predVal, className);

    app.predValueLabel.Text = num2str(predVal);
    app.nameValueLabel.Text = char(string(className));
    app.confValueLabel.Text = sprintf("%.4f", confVal);
    appendLog(fig, sprintf("单样本预测完成: 类别=%d（%s）, 置信度=%.4f", ...
        predVal, char(string(className)), confVal));
catch ME
    appendLog(fig, "单样本预测失败: " + ME.message);
end
end

function runBatchPredict(fig)
app = getappdata(fig, "app");
modelPath = app.modelField.Value;

if exist(modelPath, "file") ~= 2
    appendLog(fig, "批量预测失败: 模型文件不存在。");
    return;
end

[f, p] = uigetfile("*.csv;*.data", "选择批量输入文件");
if isequal(f, 0)
    appendLog(fig, "已取消批量预测。");
    return;
end

inFile = fullfile(p, f);
outFile = app.defaultOut;
try
    tbl = predict_wine_svm(inFile, modelPath, outFile);
    appendLog(fig, "批量预测完成，输出文件: " + outFile);

    % 简单汇总
    if any(strcmp(tbl.Properties.VariableNames, "class_name"))
        names = tbl.class_name;
        if isstring(names)
            names = cellstr(names);
        end
        [u, ~, g] = unique(names);
        cnt = accumarray(g, 1);
        for i = 1:numel(u)
            appendLog(fig, sprintf("  %s: %d", char(string(u{i})), cnt(i)));
        end
    end
catch ME
    appendLog(fig, "批量预测失败: " + ME.message);
end
end

function chooseModel(fig)
app = getappdata(fig, "app");
[f, p] = uigetfile("*.mat", "选择模型文件");
if isequal(f, 0)
    appendLog(fig, "已取消选择模型。");
    return;
end
app.modelField.Value = fullfile(p, f);
setappdata(fig, "app", app);
appendLog(fig, "模型文件已切换: " + app.modelField.Value);
refreshModelInfo(fig);
end

function refreshModelInfo(fig)
app = getappdata(fig, "app");
modelPath = app.modelField.Value;

[featureEn, featureZh] = defaultFeatureMeta();
rangeText = repmat({"-"}, 13, 1);
labelVals = [];
labelNames = {};
classExamples = struct("label", {}, "name", {}, "features", {});

if exist(modelPath, "file") == 2
    try
        s = load(modelPath, "modelData");
        modelData = s.modelData;

        if isfield(modelData, "featureNames") && numel(modelData.featureNames) == 13
            featureEn = toCellStr(modelData.featureNames);
        end
        if isfield(modelData, "featureNamesZh") && numel(modelData.featureNamesZh) == 13
            featureZh = toCellStr(modelData.featureNamesZh);
        end
        if isfield(modelData, "featureStats")
            fs = modelData.featureStats;
            if isfield(fs, "min") && isfield(fs, "max")
                for i = 1:13
                    rangeText{i} = sprintf("[%.3g, %.3g]", fs.min(i), fs.max(i));
                end
            end
        end
        if isfield(modelData, "labelValues")
            labelVals = modelData.labelValues(:);
        end
        if isfield(modelData, "labelNames")
            labelNames = toCellStr(modelData.labelNames(:));
        end
        if isfield(modelData, "classExamples")
            classExamples = normalizeClassExamples(modelData.classExamples);
        end
        if isempty(classExamples)
            examplesCsv = fullfile(fileparts(modelPath), "class_examples.csv");
            if exist(examplesCsv, "file") == 2
                classExamples = readClassExamplesCsv(examplesCsv, labelVals);
            end
        end
        if isempty(classExamples)
            classExamples = readClassExamplesFromReport(modelPath, labelVals, labelNames);
        end
        appendLog(fig, "模型信息已加载。");
    catch ME
        appendLog(fig, "加载模型信息失败: " + ME.message);
    end
else
    appendLog(fig, "模型文件不存在，使用默认特征说明。");
end

for i = 1:13
    zh = char(string(featureZh{i}));
    en = char(string(featureEn{i}));
    app.inputLabels(i).Text = sprintf("%d. %s (%s)", i, zh, en);
end

featData = cell(13, 4);
for i = 1:13
    featData{i, 1} = i;
    featData{i, 2} = char(string(featureZh{i}));
    featData{i, 3} = char(string(featureEn{i}));
    featData{i, 4} = char(string(rangeText{i}));
end
app.featureInfoTable.Data = featData;

if isempty(labelVals) && ~isempty(classExamples)
    labelVals = reshape([classExamples.label], [], 1);
    labelNames = cell(numel(classExamples), 1);
    for i = 1:numel(classExamples)
        labelNames{i} = normalizeWineClassNameUi(char(string(classExamples(i).name)), labelVals(i));
    end
end

if isempty(labelVals)
    labelVals = [1; 2; 3];
    labelNames = defaultMappedNames(labelVals);
end
if isempty(labelNames) || numel(labelNames) ~= numel(labelVals)
    labelNames = defaultMappedNames(labelVals);
end

labelNames = applyLabelMapFromReport(modelPath, labelVals, labelNames);

for i = 1:numel(labelVals)
    labelNames{i} = normalizeWineClassNameUi(labelNames{i}, labelVals(i));
end

mapData = cell(numel(labelVals), 2);
for i = 1:numel(labelVals)
    mapData{i, 1} = labelVals(i);
    mapData{i, 2} = char(string(labelNames{i}));
end
app.classMapTable.Data = mapData;

    items = cell(1, numel(labelVals));
    for i = 1:numel(labelVals)
        items{i} = char(sprintf("%d - %s", labelVals(i), char(string(labelNames{i}))));
    end
    if isempty(items)
        items = {'1 - 巴罗洛'};
    end
    app.exampleDropdown.Items = items(:)';
    app.exampleDropdown.Value = items{1};
app.exampleLabelVals = labelVals(:)';
app.classExamples = classExamples;
setappdata(fig, "app", app);

if numel(labelVals) == 3
    appendLog(fig, "说明：UCI 原始数据只有 class 1/2/3；当前名称使用常见文献命名（巴罗洛/格里尼奥利诺/巴贝拉）。");
end
if ~isempty(classExamples)
    appendLog(fig, sprintf("已加载 %d 个类别示例，可在“示例类别”中切换后点“填充示例”。", numel(classExamples)));
else
    appendLog(fig, "当前模型未包含类别示例，请重新训练后再试。");
end
end

function fillSample(fig)
app = getappdata(fig, "app");
valStr = char(string(app.exampleDropdown.Value));
label = parseLabelFromDropdown(valStr);
if isnan(label)
    for i = 1:13
        app.inputFields(i).Value = app.sample(i);
    end
    appendLog(fig, "示例类别解析失败，已填充默认示例。");
    return;
end

if ~isempty(app.classExamples)
    idx = find([app.classExamples.label] == label, 1);
    if ~isempty(idx)
        feat = app.classExamples(idx).features;
        for i = 1:min(13, numel(feat))
            app.inputFields(i).Value = feat(i);
        end
        appendLog(fig, sprintf("已填充类别 %d 示例：%s", label, char(string(app.classExamples(idx).name))));
        return;
    end
end

for i = 1:13
    app.inputFields(i).Value = app.sample(i);
end
appendLog(fig, sprintf("未找到类别 %d 示例，已填充默认示例。", label));
end

function clearInputs(fig)
app = getappdata(fig, "app");
for i = 1:13
    app.inputFields(i).Value = NaN;
end
app.predValueLabel.Text = "-";
app.nameValueLabel.Text = "-";
app.confValueLabel.Text = "-";
appendLog(fig, "已清空输入。");
end

function val = getFieldOrIndex(tbl, fieldName, idx)
if any(strcmp(tbl.Properties.VariableNames, fieldName))
    v = tbl{1, fieldName};
else
    v = tbl{1, idx};
end

if iscell(v)
    val = v{1};
elseif isstring(v)
    val = char(v(1));
else
    val = v(1);
end
end

function name = resolveClassNameFromMap(app, label, fallbackName)
name = normalizeWineClassNameUi(char(string(fallbackName)), label);
try
    d = app.classMapTable.Data;
    if isempty(d)
        return;
    end
    for i = 1:size(d, 1)
        if isequal(d{i, 1}, label)
            name = normalizeWineClassNameUi(char(string(d{i, 2})), label);
            return;
        end
    end
catch
    % keep fallback
end
end

function appendLog(fig, msg)
app = getappdata(fig, "app");
logArea = app.logArea;

old = string(logArea.Value);
old = old(:);
newLine = "[" + string(datestr(now, "HH:MM:SS")) + "] " + string(msg);
logArea.Value = [old; newLine];
drawnow;
end

function out = toCellStr(in)
if iscell(in)
    out = cell(size(in));
    for i = 1:numel(in)
        out{i} = char(string(in{i}));
    end
elseif isstring(in)
    out = cellstr(in);
else
    out = cell(size(in));
    for i = 1:numel(in)
        out{i} = char(string(in(i)));
    end
end
out = out(:)';
end

function [featureEn, featureZh] = defaultFeatureMeta()
featureEn = { ...
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', ...
    'Magnesium', 'Total phenols', 'Flavanoids', ...
    'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', ...
    'Hue', 'OD280/OD315 of diluted wines', 'Proline' ...
};
featureZh = { ...
    '酒精', '苹果酸', '灰分', '灰分碱度', '镁', '总酚', ...
    '黄酮', '非黄烷类酚', '原花青素', '色度', '色调', ...
    '稀释葡萄酒OD280/OD315', '脯氨酸' ...
};
end

function names = defaultMappedNames(labelVals)
names = cell(numel(labelVals), 1);
for i = 1:numel(labelVals)
    names{i} = defaultMappedName(labelVals(i));
end
end

function name = defaultMappedName(label)
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

function out = normalizeWineClassNameUi(inName, label)
s = strtrim(char(string(inName)));
if isempty(s)
    out = defaultMappedName(label);
    return;
end

if strcmp(s, sprintf("第%d类", label)) || ...
        strcmp(s, sprintf("UCI_Class_%d", label)) || ...
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

function label = parseLabelFromDropdown(s)
label = NaN;
tok = regexp(char(s), '^\s*(\d+)\s*-', 'tokens', 'once');
if ~isempty(tok)
    label = str2double(tok{1});
end
end

function out = normalizeClassExamples(in)
out = struct("label", {}, "name", {}, "features", {});
if isempty(in)
    return;
end
if ~isstruct(in)
    return;
end
for i = 1:numel(in)
    if ~isfield(in, "label") || ~isfield(in, "features")
        continue;
    end
    one = struct();
    one.label = double(in(i).label);
    if isfield(in, "name")
        one.name = normalizeWineClassNameUi(char(string(in(i).name)), one.label);
    else
        one.name = defaultMappedName(one.label);
    end
    one.features = double(reshape(in(i).features, 1, []));
    if numel(one.features) >= 13
        one.features = one.features(1:13);
        out(end+1) = one; %#ok<AGROW>
    end
end
end

function out = readClassExamplesCsv(csvPath, labelVals)
out = struct("label", {}, "name", {}, "features", {});
try
    t = readtable(csvPath);
    if ~all(ismember({'label', 'class_name'}, t.Properties.VariableNames))
        return;
    end
    if isempty(labelVals)
        target = unique(t.label)';
    else
        target = labelVals(:)';
    end
    featCols = setdiff(t.Properties.VariableNames, {'label', 'class_name'}, 'stable');
    for i = 1:numel(target)
        c = target(i);
        idx = find(t.label == c, 1);
        if isempty(idx)
            continue;
        end
        one = struct();
        one.label = c;
        if iscell(t.class_name)
            one.name = normalizeWineClassNameUi(char(string(t.class_name{idx})), c);
        else
            one.name = normalizeWineClassNameUi(char(string(t.class_name(idx))), c);
        end
        fv = zeros(1, numel(featCols));
        for j = 1:numel(featCols)
            fv(j) = t.(featCols{j})(idx);
        end
        if numel(fv) >= 13
            one.features = fv(1:13);
            out(end+1) = one; %#ok<AGROW>
        end
    end
catch
    % ignore
end
end

function out = readClassExamplesFromReport(modelPath, labelVals, labelNames)
out = struct("label", {}, "name", {}, "features", {});
try
    reportPath = fullfile(fileparts(modelPath), "report.json");
    if exist(reportPath, "file") ~= 2
        return;
    end
    r = jsondecode(fileread(reportPath));
    if ~isfield(r, "data") || ~isfield(r.data, "path")
        return;
    end
    dataPath = resolveInputPath(char(string(r.data.path)), reportPath, modelPath);
    if exist(dataPath, "file") ~= 2
        dataPath = resolveDefaultDataPath(modelPath);
        if exist(dataPath, "file") ~= 2
            return;
        end
    end

    raw = readmatrix(dataPath, "FileType", "text", "Delimiter", ",");
    if size(raw, 2) < 14
        return;
    end
    y = raw(:, 1);
    X = raw(:, 2:14);

    if isempty(labelVals)
        labelVals = unique(y);
    end
    labelVals = labelVals(:)';

    if isempty(labelNames) || numel(labelNames) ~= numel(labelVals)
        labelNames = cell(numel(labelVals), 1);
        for i = 1:numel(labelVals)
            labelNames{i} = defaultMappedName(labelVals(i));
        end
    end

    for i = 1:numel(labelVals)
        c = labelVals(i);
        Xi = X(y == c, :);
        if isempty(Xi)
            continue;
        end
        mu = mean(Xi, 1);
        d = sum((Xi - mu) .^ 2, 2);
        [~, idx] = min(d);

        one = struct();
        one.label = c;
        one.name = normalizeWineClassNameUi(char(string(labelNames{i})), c);
        one.features = Xi(idx, :);
        out(end+1) = one; %#ok<AGROW>
    end
catch
    % ignore
end
end

function labelNames = applyLabelMapFromReport(modelPath, labelVals, labelNames)
try
    reportPath = fullfile(fileparts(modelPath), "report.json");
    if exist(reportPath, "file") ~= 2
        return;
    end
    r = jsondecode(fileread(reportPath));
    if ~isfield(r, "data") || ~isfield(r.data, "path")
        return;
    end
    dataPath = resolveInputPath(char(string(r.data.path)), reportPath, modelPath);
    mapFile = "";
    if exist(dataPath, "file") == 2
        mapFile = string(fullfile(fileparts(dataPath), "label_map.json"));
    end
    if strlength(mapFile) == 0 || exist(char(mapFile), "file") ~= 2
        projRoot = fileparts(fileparts(modelPath));
        cand = { ...
            fullfile(projRoot, "wine", "label_map.json"), ...
            fullfile(fileparts(modelPath), "label_map.json") ...
        };
        for k = 1:numel(cand)
            if exist(cand{k}, "file") == 2
                mapFile = string(cand{k});
                break;
            end
        end
    end
    if strlength(mapFile) == 0 || exist(char(mapFile), "file") ~= 2
        return;
    end

    m = jsondecode(fileread(char(mapFile)));
    for i = 1:numel(labelVals)
        val = tryGetLabelMapValueUi(m, labelVals(i));
        if ~isempty(val)
            labelNames{i} = normalizeWineClassNameUi(char(string(val)), labelVals(i));
        end
    end
catch
    % keep original names
end
end

function outPath = resolveInputPath(inPath, reportPath, modelPath)
outPath = char(string(inPath));
if exist(outPath, "file") == 2
    return;
end

rpDir = fileparts(reportPath);
mpDir = fileparts(modelPath);
cand = { ...
    fullfile(rpDir, outPath), ...
    fullfile(mpDir, outPath), ...
    fullfile(fileparts(mpDir), outPath) ...
};
for i = 1:numel(cand)
    if exist(cand{i}, "file") == 2
        outPath = cand{i};
        return;
    end
end
end

function p = resolveDefaultDataPath(modelPath)
projRoot = fileparts(fileparts(modelPath));
cand = { ...
    fullfile(projRoot, "wine", "wine_expanded.data"), ...
    fullfile(projRoot, "wine", "wine.data") ...
};
p = '';
for i = 1:numel(cand)
    if exist(cand{i}, "file") == 2
        p = cand{i};
        return;
    end
end
p = '';
end

function val = tryGetLabelMapValueUi(m, label)
val = [];
if ~isstruct(m)
    return;
end
key = sprintf("%d", label);
alt1 = ['x', key];
fields = fieldnames(m);
if isfield(m, key)
    val = m.(key);
    return;
end
if isfield(m, alt1)
    val = m.(alt1);
    return;
end
for i = 1:numel(fields)
    fn = fields{i};
    tok = regexp(fn, '(\d+)$', 'tokens', 'once');
    if ~isempty(tok) && str2double(tok{1}) == label
        val = m.(fn);
        return;
    end
end
end
