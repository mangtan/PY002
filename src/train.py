#!/usr/bin/env python3
"""
UCI Wine 毕设项目训练脚本（按开题方案实现）
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.base import clone


FEATURE_NAMES = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 UCI Wine SVM 分类模型")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("project/wine/wine.data"),
        help="UCI Wine 数据路径（csv，无表头）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("project/outputs"),
        help="输出目录",
    )
    return parser.parse_args()


def load_data(data_path: Path):
    df = pd.read_csv(data_path, header=None)
    y = df.iloc[:, 0].astype(int).values
    x = df.iloc[:, 1:].astype(float).values
    return x, y


def evaluate_feature_methods(x, y):
    """
    三种特征方案：RFE / PCA / 基于模型重要性
    评估方式：10 折交叉验证准确率
    """
    base_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    n_features = x.shape[1]

    method_results = []

    # 1) RFE
    best_rfe = {"score": -1.0, "k": None, "time": None}
    for k in range(2, n_features + 1):
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "selector",
                    RFE(
                        estimator=LinearSVC(dual=False, max_iter=20000, random_state=42),
                        n_features_to_select=k,
                    ),
                ),
                ("clf", SVC(kernel="linear", C=1.0, random_state=42)),
            ]
        )
        start = time.perf_counter()
        scores = cross_val_score(pipe, x, y, cv=base_cv, scoring="accuracy", n_jobs=-1)
        cost = time.perf_counter() - start
        score = float(scores.mean())
        if score > best_rfe["score"] or (
            np.isclose(score, best_rfe["score"]) and k < best_rfe["k"]
        ):
            best_rfe = {"score": score, "k": k, "time": cost}
    method_results.append(
        {
            "method": "RFE",
            "best_score": best_rfe["score"],
            "selected_dim": best_rfe["k"],
            "time_sec": best_rfe["time"],
        }
    )

    # 2) PCA
    best_pca = {"score": -1.0, "k": None, "time": None}
    for k in range(2, n_features + 1):
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("selector", PCA(n_components=k, random_state=42)),
                ("clf", SVC(kernel="linear", C=1.0, random_state=42)),
            ]
        )
        start = time.perf_counter()
        scores = cross_val_score(pipe, x, y, cv=base_cv, scoring="accuracy", n_jobs=-1)
        cost = time.perf_counter() - start
        score = float(scores.mean())
        if score > best_pca["score"] or (
            np.isclose(score, best_pca["score"]) and k < best_pca["k"]
        ):
            best_pca = {"score": score, "k": k, "time": cost}
    method_results.append(
        {
            "method": "PCA",
            "best_score": best_pca["score"],
            "selected_dim": best_pca["k"],
            "time_sec": best_pca["time"],
        }
    )

    # 3) 基于模型重要性
    best_imp = {"score": -1.0, "k": None, "time": None}
    for k in range(2, n_features + 1):
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "selector",
                    SelectFromModel(
                        estimator=RandomForestClassifier(
                            n_estimators=300,
                            random_state=42,
                            n_jobs=-1,
                        ),
                        threshold=-np.inf,
                        max_features=k,
                    ),
                ),
                ("clf", SVC(kernel="linear", C=1.0, random_state=42)),
            ]
        )
        start = time.perf_counter()
        scores = cross_val_score(pipe, x, y, cv=base_cv, scoring="accuracy", n_jobs=-1)
        cost = time.perf_counter() - start
        score = float(scores.mean())
        if score > best_imp["score"] or (
            np.isclose(score, best_imp["score"]) and k < best_imp["k"]
        ):
            best_imp = {"score": score, "k": k, "time": cost}
    method_results.append(
        {
            "method": "ModelImportance",
            "best_score": best_imp["score"],
            "selected_dim": best_imp["k"],
            "time_sec": best_imp["time"],
        }
    )

    # 选择最优：先准确率，再维度，再耗时
    best = sorted(
        method_results,
        key=lambda r: (-r["best_score"], r["selected_dim"], r["time_sec"]),
    )[0]
    return method_results, best


def build_selector(method_name: str, k: int):
    if method_name == "RFE":
        return RFE(
            estimator=LinearSVC(dual=False, max_iter=20000, random_state=42),
            n_features_to_select=k,
        )
    if method_name == "PCA":
        return PCA(n_components=k, random_state=42)
    if method_name == "ModelImportance":
        return SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
            threshold=-np.inf,
            max_features=k,
        )
    raise ValueError(f"未知特征方法: {method_name}")


def run_kernel_search(x, y, selector):
    """
    核函数对比：Linear / RBF / Sigmoid
    超参数优化：网格搜索 + 5 折交叉验证
    """
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c_grid = np.logspace(-3, 3, 7)
    gamma_rbf_grid = np.logspace(-4, 2, 7)
    gamma_sigmoid_grid = np.logspace(-3, 0, 7)
    coef0_grid = np.linspace(0, 3, 7)

    kernel_defs = {
        "linear": {
            "svc__C": c_grid,
        },
        "rbf": {
            "svc__C": c_grid,
            "svc__gamma": gamma_rbf_grid,
        },
        "sigmoid": {
            "svc__C": c_grid,
            "svc__gamma": gamma_sigmoid_grid,
            "svc__coef0": coef0_grid,
        },
    }

    kernel_results = []
    best_overall = None

    for kernel_name, grid in kernel_defs.items():
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("selector", selector),
                ("svc", SVC(kernel=kernel_name, probability=True, random_state=42)),
            ]
        )
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="accuracy",
            cv=cv5,
            n_jobs=-1,
        )
        gs.fit(x, y)
        row = {
            "kernel": kernel_name,
            "best_score": float(gs.best_score_),
            "best_params": {k: float(v) for k, v in gs.best_params_.items()},
        }
        kernel_results.append(row)
        if best_overall is None or row["best_score"] > best_overall["best_score"]:
            best_overall = {
                "kernel": kernel_name,
                "best_score": row["best_score"],
                "best_params": gs.best_params_,
            }
    return kernel_results, best_overall


def get_final_cv_strategy(n_samples: int):
    if n_samples < 50:
        return "LOOCV", LeaveOneOut()
    if 50 <= n_samples <= 200:
        return "Repeated10x10CV", RepeatedStratifiedKFold(
            n_splits=10, n_repeats=10, random_state=42
        )
    return "Stratified10Fold", StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def repeated_cv_majority_predict(model, x, y, cv):
    """
    Repeated CV 下，每个样本会被多次预测。
    这里对每个样本的多次预测结果做多数投票，得到最终标签。
    """
    votes = [[] for _ in range(len(y))]
    for train_idx, test_idx in cv.split(x, y):
        fold_model = clone(model)
        fold_model.fit(x[train_idx], y[train_idx])
        fold_pred = fold_model.predict(x[test_idx])
        for idx, pred in zip(test_idx, fold_pred):
            votes[idx].append(int(pred))

    y_pred = np.zeros_like(y)
    for i, sample_votes in enumerate(votes):
        if not sample_votes:
            raise RuntimeError("样本在 Repeated CV 中没有得到预测结果")
        # 多数票；票数相同则取类别编号较小者
        count = Counter(sample_votes)
        y_pred[i] = sorted(count.items(), key=lambda t: (-t[1], t[0]))[0][0]
    return y_pred


def draw_bar(values, labels, title, output_path: Path):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def draw_confusion(cm, labels, output_path: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_data(args.data)
    n_samples, n_features = x.shape

    feature_results, best_feature = evaluate_feature_methods(x, y)
    selector = build_selector(best_feature["method"], best_feature["selected_dim"])

    # 若特征维度超过 20，按开题做 LDA 进一步降维（本数据一般不会触发）
    use_lda = best_feature["selected_dim"] > 20

    kernel_results, best_kernel = run_kernel_search(x, y, selector)

    # 构建最终模型
    final_steps = [
        ("scaler", StandardScaler()),
        ("selector", selector),
    ]
    if use_lda:
        # 数据集是 3 类，LDA 最多降到 2 维
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        final_steps.append(("lda", LinearDiscriminantAnalysis()))
    final_steps.append(
        (
            "svc",
            SVC(
                kernel=best_kernel["kernel"],
                probability=True,
                random_state=42,
            ),
        )
    )

    final_model = Pipeline(steps=final_steps)
    final_model.set_params(**best_kernel["best_params"])

    eval_name, eval_cv = get_final_cv_strategy(n_samples)
    if eval_name == "Repeated10x10CV":
        y_pred = repeated_cv_majority_predict(final_model, x, y, eval_cv)
    else:
        y_pred = cross_val_predict(final_model, x, y, cv=eval_cv, n_jobs=-1)

    acc = float(accuracy_score(y, y_pred))
    cls_report = classification_report(y, y_pred, output_dict=True, digits=4)
    cm = confusion_matrix(y, y_pred)

    final_model.fit(x, y)
    model_path = args.output_dir / "model.joblib"
    joblib.dump(final_model, model_path)

    report = {
        "data": {
            "path": str(args.data),
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "classes": sorted(np.unique(y).tolist()),
            "feature_names": FEATURE_NAMES,
        },
        "feature_selection": {
            "all_methods": feature_results,
            "best_method": best_feature,
        },
        "kernel_search": {
            "all_kernels": kernel_results,
            "best_kernel": {
                "kernel": best_kernel["kernel"],
                "best_score": float(best_kernel["best_score"]),
                "best_params": {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in best_kernel["best_params"].items()
                },
            },
        },
        "final_evaluation": {
            "strategy": eval_name,
            "accuracy": acc,
            "classification_report": cls_report,
            "confusion_matrix": cm.tolist(),
        },
        "artifacts": {
            "model": str(model_path),
            "feature_bar": str(args.output_dir / "feature_method_compare.png"),
            "kernel_bar": str(args.output_dir / "kernel_compare.png"),
            "confusion_matrix": str(args.output_dir / "confusion_matrix.png"),
        },
    }

    report_path = args.output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 画图输出
    draw_bar(
        [m["best_score"] for m in feature_results],
        [m["method"] for m in feature_results],
        "Feature Method Comparison (10-Fold Accuracy)",
        args.output_dir / "feature_method_compare.png",
    )
    draw_bar(
        [k["best_score"] for k in kernel_results],
        [k["kernel"] for k in kernel_results],
        "Kernel Comparison (GridSearch 5-Fold Accuracy)",
        args.output_dir / "kernel_compare.png",
    )
    draw_confusion(
        cm,
        labels=[str(c) for c in sorted(np.unique(y))],
        output_path=args.output_dir / "confusion_matrix.png",
    )

    print(f"训练完成，模型已保存: {model_path}")
    print(f"评估准确率 ({eval_name}): {acc:.4f}")
    print(f"报告文件: {report_path}")


if __name__ == "__main__":
    main()
