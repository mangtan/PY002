# 基于 SVM 的意大利葡萄酒种类识别（UCI Wine）

## MATLAB 实现（毕设主线）

- `matlab/train_wine_svm.m`: MATLAB 训练与评估脚本
- `matlab/predict_wine_svm.m`: MATLAB 推理脚本
- `matlab/wine_system_prototype.m`: MATLAB 可视化系统原型（GUI）

在 MATLAB 命令行执行（建议在仓库根目录）：

```matlab
addpath("project/matlab");
report = train_wine_svm("project/wine/wine.data", "project/outputs_matlab");
```

扩充数据训练：

```matlab
addpath("project/matlab");
report = train_wine_svm("project/wine/wine_expanded.data", "project/outputs_matlab");
```

单样本预测：

```matlab
addpath("project/matlab");
tbl = predict_wine_svm([14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065], ...
    "project/outputs_matlab/model.mat");
```

打开系统原型界面：

```matlab
addpath("project/matlab");
wine_system_prototype;
```

## 目录结构

- `wine/wine.data`: UCI Wine 原始数据
- `src/train.py`: 训练与评估脚本
- `src/predict.py`: 推理脚本
- `outputs/`: 模型、图表、报告输出目录

## 环境要求

- Python 3.9+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `joblib`

## 训练

```bash
python3 project/src/train.py --data project/wine/wine.data --output-dir project/outputs
```

训练完成后会生成：

- `project/outputs/model.joblib`
- `project/outputs/report.json`
- `project/outputs/feature_method_compare.png`
- `project/outputs/kernel_compare.png`
- `project/outputs/confusion_matrix.png`

## 预测

### 单样本预测（13个特征）

```bash
python3 project/src/predict.py \
  --model project/outputs/model.joblib \
  --sample "14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065"
```

### 批量预测（CSV每行13列）

```bash
python3 project/src/predict.py \
  --model project/outputs/model.joblib \
  --input-csv your_input.csv \
  --output-csv project/outputs/predict_result.csv
```

## 扩充数据集（合成）

```bash
python3 project/src/generate_expanded_dataset.py \
  --input project/wine/wine.data \
  --output-data project/wine/wine_expanded.data \
  --output-meta project/wine/wine_expanded_with_meta.csv \
  --output-map project/wine/label_map.json \
  --per-class 120 \
  --shift-scale 2.5 \
  --noise-scale 0.28
```

输出说明：

- `wine_expanded.data`: 与 UCI `wine.data` 相同格式（首列标签+13特征，无表头）
- `wine_expanded_with_meta.csv`: 含 `variety/source` 元信息
- `label_map.json`: 标签编号到类别名称映射
