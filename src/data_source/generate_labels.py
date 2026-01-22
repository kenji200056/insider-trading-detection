# -*- coding: utf-8 -*-
"""
ラベル生成スクリプト（labels_final.csv）
決定木で選ばれた最重要特徴量を使用
"""
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, "data/complete_dataset_jp.csv")
output_file = os.path.join(current_dir, "result/labels_final.csv")
top_feature_file = os.path.join(current_dir, "result/top_feature.txt")

# データ読み込み
df = pd.read_csv(input_file, encoding='utf-8-sig')

# 最重要特徴量を読み込み
if os.path.exists(top_feature_file):
    with open(top_feature_file, 'r', encoding='utf-8') as f:
        top_feature = f.read().strip()
    print(f"📖 最重要特徴量を読み込み: '{top_feature}'")
else:
    print(f"❌ エラー: {top_feature_file} が見つかりません")
    print(f"   ヒント: 先に generate_classification.py を実行してください")
    sys.exit(1)

# 最重要特徴量で分類
title_has = df['title'].str.contains(top_feature, na=False)
content_has = df['content'].str.contains(top_feature, na=False)
labels = (title_has | content_has).astype(int)

# labels_final.csvとして保存
labels_df = pd.DataFrame({'label': labels})
labels_df.to_csv(output_file, index=False)

print(f"✅ labels_final.csv を生成しました: {output_file}")
print(f"   分類基準: '{top_feature}'")
print(f"   class=1: {labels.sum()}件")
print(f"   class=0: {len(labels) - labels.sum()}件")
