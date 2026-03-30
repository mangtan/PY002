function ok = setup_libsvm()
% 一键配置 libsvm (MATLAB)
% 用法：
%   addpath('<project>/project/matlab');
%   ok = setup_libsvm();

ok = false;
baseDir = fileparts(mfilename("fullpath"));
cand = {
    fullfile(baseDir, "..", "..", "third_party", "libsvm", "matlab"), ...
    fullfile(baseDir, "..", "third_party", "libsvm", "matlab")
};

target = '';
for i = 1:numel(cand)
    if exist(cand{i}, "dir") == 7
        target = cand{i};
        break;
    end
end

if isempty(target)
    fprintf("未找到 libsvm/matlab 目录。\n");
    fprintf("请确认你已拉取项目内的 third_party/libsvm\n");
    return;
end

addpath(target);
rehash;

if ~hasLibsvmMex()
    old = pwd;
    cleanupObj = onCleanup(@() cd(old)); %#ok<NASGU>
    cd(target);

    % Apple Silicon + 新版链接器下，默认 make.m 可能失败；增加回退编译方案
    plans = {
        struct("name", "默认 mex 参数", "extra", {{}}), ...
        struct("name", "LINKFLAGS + -Wl,-ld_classic", ...
               "extra", {{'LINKFLAGS=$LINKFLAGS -Wl,-ld_classic'}}), ...
        struct("name", "LDFLAGS + -Wl,-ld_classic", ...
               "extra", {{'LDFLAGS=$LDFLAGS -Wl,-ld_classic'}}), ...
        struct("name", "LINKFLAGS + -ld_classic", ...
               "extra", {{'LINKFLAGS=$LINKFLAGS -ld_classic'}}), ...
        struct("name", "LDFLAGS + -ld_classic", ...
               "extra", {{'LDFLAGS=$LDFLAGS -ld_classic'}})
    };

    for i = 1:numel(plans)
        try
            fprintf("尝试编译方案 %d/%d: %s\n", i, numel(plans), plans{i}.name);
            compileLibsvmMex(plans{i}.extra);
        catch ME
            fprintf("方案失败: %s\n", ME.message);
        end
        rehash;
        if hasLibsvmMex()
            break;
        end
    end
end

ok = hasLibsvmMex();
if ok
    fprintf("libsvm 已可用。\n");
    fprintf("svmtrain: %s\n", which("svmtrain"));
    fprintf("svmpredict: %s\n", which("svmpredict"));
    try
        savepath;
    catch
    end
else
    fprintf("libsvm 仍不可用，请检查 mex 编译环境。\n");
end
end

function compileLibsvmMex(extraArgs)
if nargin < 1
    extraArgs = {};
end

mexArgs = {'-largeArrayDims'};
if ~isempty(extraArgs)
    mexArgs = [extraArgs(:).', mexArgs]; %#ok<AGROW>
end

% 基础读写函数
mex(mexArgs{:}, 'libsvmread.c');
mex(mexArgs{:}, 'libsvmwrite.c');

% 核心训练/预测函数
mex(mexArgs{:}, '-I..', 'svmtrain.c', '../svm.cpp', 'svm_model_matlab.c');
mex(mexArgs{:}, '-I..', 'svmpredict.c', '../svm.cpp', 'svm_model_matlab.c');
end

function tf = hasLibsvmMex()
p1 = which('svmtrain');
p2 = which('svmpredict');
tf = ~isempty(p1) && ~isempty(p2);
if tf
    tf = endsWith(p1, ['.', mexext]) && endsWith(p2, ['.', mexext]);
end
end
