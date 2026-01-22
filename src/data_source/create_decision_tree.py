# -*- coding: utf-8 -*-
"""
最終版：max_depth=1のシンプルな決定木
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pydotplus
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# データ読み込み
dataset = pd.read_csv(os.path.join(current_dir, "result/feature_vector_unlabeled.csv"), index_col=0)
features_df = pd.read_csv(os.path.join(current_dir, "result/feature_vector_index_map.csv"))
original_df = pd.read_csv(os.path.join(current_dir, "data/complete_dataset_jp.csv"), encoding='utf-8-sig')

X = dataset.values
features = features_df['feature'].tolist()

# ラベル作成（title OR content）と反転
title_has = original_df['title'].str.contains('内部者取引', na=False)
content_has = original_df['content'].str.contains('内部者取引', na=False)
labels_original = (title_has | content_has).astype(int).values
labels = 1 - labels_original  # 色を反転

print("=" * 70)
print("最終版決定木を作成（max_depth=1、色反転版）")
print("=" * 70)
print(f"\n内部者取引あり: {labels_original.sum()}件 ({labels_original.sum()/len(labels)*100:.1f}%)")
print(f"内部者取引なし: {len(labels) - labels_original.sum()}件 ({(len(labels)-labels_original.sum())/len(labels)*100:.1f}%)")

# 訓練・テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, random_state=42, test_size=0.3, stratify=labels
)

# 決定木（max_depth=1）
clf = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
clf.fit(X_train, y_train)

# 評価
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n正解率: {accuracy:.4f}")
print(f"決定木の深さ: {clf.get_depth()}")
print(f"葉ノード数: {clf.get_n_leaves()}")

# 決定木の可視化
print("\n決定木を生成中...")

valid_features = [str(f) if pd.notna(f) else f"feature_{i}" for i, f in enumerate(features)]
features_short = [f[:20] if len(f) > 20 else f for f in valid_features]

dot_data = tree.export_graphviz(
    clf, out_file=None,
    feature_names=features_short,
    class_names=['内部者取引', '非内部者取引'],  # 反転版
    filled=True, rounded=True,
    special_characters=True
)

graph = pydotplus.graph_from_dot_data(dot_data)

output_folder = os.path.join(current_dir, "result/")
pdf_path = os.path.join(output_folder, "decision_tree_final.pdf")
png_path = os.path.join(output_folder, "decision_tree_final.png")

graph.write_pdf(pdf_path)
graph.write_png(png_path)

print(f"決定木をPDFとして保存: {pdf_path}")
print(f"決定木をPNGとして保存: {png_path}")

# ラベルを保存
labels_df = pd.DataFrame({
    'index': range(len(labels_original)),
    'label': labels_original,
    'label_name': ['内部者取引' if l == 1 else '非内部者取引' for l in labels_original]
})
labels_path = os.path.join(output_folder, "labels_final.csv")
labels_df.to_csv(labels_path, index=False)
print(f"ラベルを保存: {labels_path}")

# 結果サマリー
results_path = os.path.join(output_folder, "results_final.txt")
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=== 最終版：キーワードベース分類結果 ===\n\n")
    f.write("方法: title OR content に「内部者取引」が含まれるかどうかで分類\n")
    f.write("決定木: max_depth=1（最もシンプル）\n")
    f.write("色: 反転版（オレンジ=内部者取引、青=非内部者取引）\n\n")
    f.write(f"内部者取引あり: {labels_original.sum()}件 ({labels_original.sum()/len(labels)*100:.1f}%)\n")
    f.write(f"内部者取引なし: {len(labels) - labels_original.sum()}件 ({(len(labels)-labels_original.sum())/len(labels)*100:.1f}%)\n\n")
    f.write(f"正解率: {accuracy:.4f}\n")
    f.write(f"決定木の深さ: {clf.get_depth()}\n")
    f.write(f"葉ノード数: {clf.get_n_leaves()}\n\n")
    f.write("色の意味:\n")
    f.write("  🟧 オレンジ = 内部者取引\n")
    f.write("  🟦 青 = 非内部者取引\n\n")
    f.write("読み方:\n")
    f.write("  内部者取引 ≤ 0.5\n")
    f.write("  ├─ True（左・青）: 単語なし（0） → 非内部者取引\n")
    f.write("  └─ False（右・オレンジ）: 単語あり（1） → 内部者取引\n")

print(f"結果サマリーを保存: {results_path}")

print("\n" + "=" * 70)
print("処理完了！")
print("=" * 70)
print("\n色の意味:")
print("  🟧 オレンジ = 内部者取引")
print("  🟦 青 = 非内部者取引")
