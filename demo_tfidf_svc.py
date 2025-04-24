#!/usr/bin/env python

"""
示例：
python demo_tfidf_svc.py --csv df_file.csv --text-col 0 --label-col 1 --test-size 0.25

    --no-header            # 如果 CSV 没有表头就加上
"""

import argparse, os, random, jieba, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# ---------- 分词器 ----------
def jieba_tokenizer(text: str):
    """结巴分词；把换行统一替换为分号"""
    return list(jieba.cut(text.replace("\n", ";")))

# ---------- CSV 读取 ----------
def load_csv(path: str, text_idx: int, label_idx: int, no_header: bool):
    """
    读取 CSV 并取指定列。
    参数 no_header=True 时，会用 header=None 读取，**不会把首行当标题**。
    """
    df = pd.read_csv(path, header=None if no_header else 0)
    texts  = df.iloc[:, text_idx].astype(str)
    labels = df.iloc[:, label_idx].astype(str)
    return texts, labels

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser("TF-IDF + LinearSVC Demo")
    ap.add_argument("--csv", required=True, help="CSV 文件路径")
    ap.add_argument("--text-col", type=int, default=0, help="文本列索引 (0-based)")
    ap.add_argument("--label-col", type=int, default=1, help="标签列索引 (0-based)")
    ap.add_argument("--no-header", action="store_true", help="CSV 无表头时加此标志")
    ap.add_argument("--test-size", type=float, default=0.2, help="测试集占比")
    ap.add_argument("--random-seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    # 1. 读数据
    texts, labels = load_csv(args.csv, args.text_col, args.label_col, args.no_header)

    # 打印 3 条样例，检查是否取对列
    print("\n>>> 样例 text / label")
    for idx in random.sample(range(len(texts)), k=min(3, len(texts))):
        print(f"[{idx}] {texts.iloc[idx][:60]} ...  ==>  {labels.iloc[idx]}")
    print("-" * 60)

    # 2. 划分训练 / 测试
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=labels
    )

    # 3. TF-IDF 构建
    tfidf = TfidfVectorizer(
        tokenizer=jieba_tokenizer,
        token_pattern=None,
        min_df=2,          # 剔除只出现 1 次的词
        max_df=0.9         # 剔除极高频停用词
    )
    X_tr_vec = tfidf.fit_transform(X_tr)
    X_te_vec = tfidf.transform(X_te)

    # 4. LinearSVC 训练
    clf = LinearSVC(class_weight="balanced", max_iter=10000)
    clf.fit(X_tr_vec, y_tr)

    # 5. 评估
    y_pred = clf.predict(X_te_vec)
    print("\n=== 分类报告 ===")
    print(classification_report(y_te, y_pred))

    # 6. 保存模型与词表
    out_dir = "demo_model"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tfidf_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tfidf.vocabulary_, f, ensure_ascii=False)
    np.save(os.path.join(out_dir, "tfidf_idf.npy"), tfidf.idf_)
    joblib.dump(clf, os.path.join(out_dir, "linear_svc.joblib"))
    print(f"模型文件已保存到 {out_dir}/")

if __name__ == "__main__":
    main()