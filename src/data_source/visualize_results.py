# -*- coding: utf-8 -*-
"""
キーワードベース分類の可視化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os

matplotlib.use('Agg')
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
file_data = os.path.join(current_dir, "result/feature_vector_unlabeled.csv")
file_labels = os.path.join(current_dir, "result/labels_final.csv")  # 最終版を使用
file_original = os.path.join(current_dir, "data/complete_dataset_jp.csv")
output_folder = os.path.join(current_dir, "result/")

print("=" * 70)
print("最終版のラベルを使って可視化を再生成中...")
print("=" * 70)
print("\nデータを読み込み中...")
dataset = pd.read_csv(file_data, index_col=0)
labels_df = pd.read_csv(file_labels)
original_df = pd.read_csv(file_original, encoding='utf-8-sig')

X = dataset.values
labels = labels_df['label'].values

print(f"データ件数: {len(X)}")
print(f"特徴量数: {X.shape[1]}")
print(f"内部者取引あり: {labels.sum()}件")
print(f"内部者取引なし: {len(labels) - labels.sum()}件")

# カラーパレット（直感的：内部者取引=赤）
colors = ['#4A90E2', '#FF6B6B']  # 青系（非内部者取引）、赤系（内部者取引）
label_names = ['非内部者取引', '内部者取引']

# 図1: ラベル分布（円グラフ）
print("\n図1: ラベル分布を作成中...")
fig, ax = plt.subplots(figsize=(10, 8))
counts = [len(labels) - labels.sum(), labels.sum()]
explode = (0.05, 0.05)
wedges, texts, autotexts = ax.pie(
    counts,
    labels=label_names,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    startangle=90,
    textprops={'fontsize': 14, 'weight': 'bold'}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
ax.set_title('内部者取引キーワードによる分類結果', fontsize=18, weight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'viz1_label_distribution.png'), dpi=300, bbox_inches='tight')
print(f"保存: viz1_label_distribution.png")
plt.close()

# 図2: 時系列分布
print("\n図2: 時系列分布を作成中...")
try:
    original_df['date'] = pd.to_datetime(original_df['date'], format='mixed', errors='coerce')
    original_df['year'] = original_df['date'].dt.year
    original_df['label'] = labels

    # NaN（日付が不正な行）を除外
    valid_dates = original_df.dropna(subset=['year'])

    year_counts = valid_dates.groupby(['year', 'label']).size().unstack(fill_value=0)
except Exception as e:
    print(f"  時系列分析でエラー: {e}")
    print("  図2をスキップします")
    year_counts = None

if year_counts is not None:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = year_counts.index
    width = 0.4
    x_pos = np.arange(len(x))

    ax.bar(x_pos - width/2, year_counts[0] if 0 in year_counts.columns else 0,
           width, label='非内部者取引', color=colors[0], alpha=0.8)
    ax.bar(x_pos + width/2, year_counts[1] if 1 in year_counts.columns else 0,
           width, label='内部者取引', color=colors[1], alpha=0.8)

    ax.set_xlabel('年', fontsize=14, weight='bold')
    ax.set_ylabel('件数', fontsize=14, weight='bold')
    ax.set_title('年別の内部者取引案件数の推移', fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'viz2_yearly_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"保存: viz2_yearly_distribution.png")
    plt.close()

# 図3, 4, 5: スキップ（不要）

# 図6: 上位特徴量の出現頻度比較（上位5位まで、順位付き）
print("\n図6: 特徴量頻度比較を作成中（上位5位）...")
features_df = pd.read_csv(os.path.join(current_dir, "result/feature_vector_index_map.csv"))
features = features_df['feature'].tolist()

# 各グループでの特徴量頻度
freq_insider = X[labels == 1].sum(axis=0)
freq_non_insider = X[labels == 0].sum(axis=0)

# 比率を計算
total_insider = labels.sum()
total_non_insider = len(labels) - labels.sum()
ratio_insider = freq_insider / total_insider
ratio_non_insider = freq_non_insider / total_non_insider

# 差が大きい上位5特徴量を抽出（差の絶対値で並べ替え）
diff = ratio_insider - ratio_non_insider
top_indices = np.argsort(np.abs(diff))[-5:][::-1]

top_features = [features[i] if pd.notna(features[i]) else f'feature_{i}' for i in top_indices]
top_ratio_insider = [ratio_insider[i] for i in top_indices]
top_ratio_non_insider = [ratio_non_insider[i] for i in top_indices]

# 順位付きラベルを作成
ranked_labels = [f'{i+1}位: {f[:25]}' for i, f in enumerate(top_features)]

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = np.arange(len(ranked_labels))
width = 0.35

# 非内部者取引（青）
ax.barh(y_pos - width/2, top_ratio_non_insider, width,
        label='非内部者取引', color=colors[0], alpha=0.85, edgecolor='white', linewidth=1)
# 内部者取引（赤）
ax.barh(y_pos + width/2, top_ratio_insider, width,
        label='内部者取引', color=colors[1], alpha=0.85, edgecolor='white', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(ranked_labels, fontsize=13, weight='bold')
ax.set_xlabel('1件あたりの平均出現率', fontsize=14, weight='bold')
ax.set_title('グループ間で差が大きい上位5特徴量', fontsize=16, weight='bold', pad=20)
ax.legend(fontsize=13, loc='best')
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)

# Y軸を反転（1位が上に）
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'viz6_feature_frequency.png'), dpi=300, bbox_inches='tight')
print(f"保存: viz6_feature_frequency.png")
plt.close()

print("\n" + "=" * 70)
print("すべての可視化が完了しました！")
print("=" * 70)
print(f"\n出力フォルダ: {output_folder}")
print("\n生成されたファイル（シンプル版）:")
print("  1. viz1_label_distribution.png - ラベル分布（円グラフ）")
print("  2. viz2_yearly_distribution.png - 年別分布")
print("  3. viz6_feature_frequency.png - 特徴量頻度比較（上位5位、順位付き）")
print("\n改善点:")
print("  ✅ 内部者取引 = 赤（直感的！）")
print("  ✅ viz6は上位5位まで、順位表示付き")
print("  ✅ viz3, 4, 5は削除（複雑で不要）")
print("\n使用したラベル: labels_final.csv")
print("=" * 70)
