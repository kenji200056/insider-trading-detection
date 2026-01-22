# -*- coding: utf-8 -*-
"""
パターン可視化スクリプト
11個の代表パターンをグラフ化して論文用図を生成
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 日本語フォント設定
font_candidates = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in font_candidates:
    if font in available_fonts:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False

def load_patterns(pattern_file, sizes_file):
    """パターンデータを読み込み"""
    patterns = pd.read_csv(pattern_file, header=None).values
    sizes = pd.read_csv(sizes_file, header=None).values.flatten()
    return patterns, sizes

def visualize_patterns(patterns, sizes, output_file):
    """11パターンを3x4のグリッドで可視化"""
    
    n_patterns = len(patterns)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=300)
    axes = axes.flatten()
    
    # パターンの解釈（形状から推測）
    interpretations = [
        "急騰型（スパイク）",
        "長期変動型",
        "緩やかな上昇型",
        "初期急騰型",
        "後半急騰型",
        "中盤ピーク型",
        "後半集中型",
        "中盤集中型",
        "前半ピーク型",
        "階段状上昇型",
        "複合ピーク型"
    ]
    
    for i in range(n_patterns):
        ax = axes[i]
        size = int(sizes[i]) if i < len(sizes) else len(patterns[i])
        pattern = patterns[i, :size]
        
        # パターンをプロット
        days = np.arange(1, len(pattern) + 1)
        ax.plot(days, pattern, 'b-', linewidth=2, alpha=0.8)
        ax.fill_between(days, 0, pattern, alpha=0.3, color='blue')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # タイトルと軸ラベル
        ax.set_title(f'パターン {i+1}: {interpretations[i]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('日数', fontsize=8)
        ax.set_ylabel('正規化誤差', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # 最後のセルは非表示（11パターンなので）
    axes[11].axis('off')
    
    plt.suptitle('K-meansクラスタリングにより抽出された11個の代表パターン', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 保存: {output_file}")
    return output_file

def main():
    print("\n" + "="*60)
    print("📊 パターン可視化")
    print("="*60 + "\n")
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # パターン読み込み
    pattern_file = 'pattern.csv'
    sizes_file = 'pattern_sizes.csv'
    
    if not os.path.exists(pattern_file):
        print(f"❌ {pattern_file} が見つかりません")
        return
    
    patterns, sizes = load_patterns(pattern_file, sizes_file)
    print(f"📂 パターン数: {len(patterns)}")
    print(f"📂 パターンサイズ: {sizes}")
    
    # 可視化
    output_file = '../errors/pattern_gallery.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    visualize_patterns(patterns, sizes, output_file)
    
    print("\n🎉 完了!")

if __name__ == '__main__':
    main()
