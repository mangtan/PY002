#!/usr/bin/env python3
"""
UCI Wine 毕设项目预测脚本
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="加载训练好的模型进行预测")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/model.joblib"),
        help="训练好的模型路径",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="输入csv路径（每行13列特征，或14列含首列标签）",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="单样本输入，13个特征用逗号分隔",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/predict_result.csv"),
        help="批量预测输出路径",
    )
    return parser.parse_args()


def parse_single_sample(sample_str: str):
    values = [float(v.strip()) for v in sample_str.split(",")]
    if len(values) != 13:
        raise ValueError("单样本必须包含13个特征值")
    return np.array(values, dtype=float).reshape(1, -1)


def main():
    args = parse_args()
    model = joblib.load(args.model)

    if args.sample is None and args.input_csv is None:
        raise ValueError("请提供 --sample 或 --input-csv 其中之一")

    if args.sample is not None:
        x = parse_single_sample(args.sample)
        pred = model.predict(x)
        prob = model.predict_proba(x)
        conf = float(np.max(prob, axis=1)[0])
        print(f"预测类别: {int(pred[0])}")
        print(f"置信度: {conf:.4f}")
        return

    df = pd.read_csv(args.input_csv, header=None)
    if df.shape[1] == 13:
        x = df.values.astype(float)
    elif df.shape[1] == 14:
        x = df.iloc[:, 1:].values.astype(float)
    else:
        raise ValueError("输入CSV必须是13列特征，或14列（首列标签+13特征）")
    pred = model.predict(x)
    prob = model.predict_proba(x)
    conf = np.max(prob, axis=1)

    out_df = df.copy()
    out_df["pred_label"] = pred
    out_df["confidence"] = conf
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"批量预测完成，输出文件: {args.output_csv}")


if __name__ == "__main__":
    main()
