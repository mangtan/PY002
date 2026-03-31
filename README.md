# 基于 SVM 的意大利葡萄酒种类识别（UCI Wine）

## MATLAB 实现（毕设主线）

- `matlab/train_wine_svm.m`: MATLAB 训练与评估脚本
- `matlab/predict_wine_svm.m`: MATLAB 推理脚本
- `matlab/wine_system_prototype.m`: MATLAB 可视化系统原型（GUI）

在 MATLAB 命令行执行（建议在仓库根目录，即包含 `matlab/src/wine` 的目录）：

```matlab
addpath("matlab");
report = train_wine_svm("wine/wine.data", "outputs_matlab");
```

扩充数据训练：

```matlab
addpath("matlab");
report = train_wine_svm("wine/wine_expanded.data", "outputs_matlab_expanded");
```

单样本预测：

```matlab
addpath("matlab");
tbl = predict_wine_svm([14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065], ...
    "outputs_matlab_expanded/model.mat");
```

打开系统原型界面：

```matlab
addpath("matlab");
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
python3 src/train.py --data wine/wine.data --output-dir outputs
```

训练完成后会生成：

- `outputs/model.joblib`
- `outputs/report.json`
- `outputs/feature_method_compare.png`
- `outputs/kernel_compare.png`
- `outputs/confusion_matrix.png`

## 预测

### 单样本预测（13个特征）

```bash
python3 src/predict.py \
  --model outputs/model.joblib \
  --sample "14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065"
```

### 批量预测（CSV每行13列，或14列含首列标签）

```bash
python3 src/predict.py \
  --model outputs/model.joblib \
  --input-csv your_input.csv \
  --output-csv outputs/predict_result.csv
```

## 扩充数据集（合成）

```bash
python3 src/generate_expanded_dataset.py \
  --input wine/wine.data \
  --output-data wine/wine_expanded.data \
  --output-meta wine/wine_expanded_with_meta.csv \
  --output-map wine/label_map.json \
  --per-class 120 \
  --shift-scale 2.5 \
  --noise-scale 0.28
```

输出说明：

- `wine_expanded.data`: 与 UCI `wine.data` 相同格式（首列标签+13特征，无表头）
- `wine_expanded_with_meta.csv`: 含 `variety/source` 元信息
- `label_map.json`: 标签编号到类别名称映射
