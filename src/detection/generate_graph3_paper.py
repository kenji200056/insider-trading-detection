# -*- coding: utf-8 -*-
"""
論文用 graph3-paper 生成スクリプト
graph3と同じデザイン、ただしTrue系=緑、False系=薄ピンク
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors

# 日本語フォント設定
font_candidates = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in font_candidates:
    if font in available_fonts:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False

def generate_graph3_paper():
    """論文用の混同行列グラフを生成"""
    
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 結果データ
    result = {
        'threshold': 0.35,
        'tp': 28,
        'fn': 5,
        'fp': 1,
        'tn': 124,
        'precision': 0.966,
        'recall': 0.848,
        'f1': 0.903
    }
    
    # 混同行列の配置（graph3と同じ）
    # [TP, FN]
    # [FP, TN]
    cm_display = np.array([
        [result['tp'], result['fn']],
        [result['fp'], result['tn']]
    ])
    
    # True系=低値色（緑）、False系=高値色（赤っぽい）のマスク
    # TP(0,0)=True, FN(0,1)=False, FP(1,0)=False, TN(1,1)=True
    # graph3のRdYlGn_rでは高値=赤、低値=緑
    # 今回はTrue=緑（低値側）、False=赤（高値側）にしたいので
    # True=0, False=1 にして RdYlGn を使う
    color_mask = np.array([
        [0, 1],  # TP=緑, FN=赤
        [1, 0]   # FP=赤, TN=緑
    ])
    
    # プロフェッショナルなシックカラー（低彩度、落ち着いた色調）
    colors_list = ['#5a9a6e', '#c98989']  # 濃いめのシックな緑、濃いめのシックなローズ
    cmap = mcolors.ListedColormap(colors_list)
    
    # 超横長図（graph3と同じスタイル）
    fig, ax = plt.subplots(figsize=(28, 4), dpi=300)
    
    im = ax.imshow(color_mask, cmap=cmap, alpha=1.0)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['予測: Infected', '予測: Non-infected'], fontsize=12)
    ax.set_yticklabels(['実際: Infected', '実際: Non-infected'], fontsize=12)
    
    # タイトルなし（graph3との違い）
    
    labels = [
        ['True Positive\n(正検出)', 'False Negative\n(見逃し)'],
        ['False Positive\n(誤検出)', 'True Negative\n(正常判定)']
    ]
    
    for i in range(2):
        for j in range(2):
            value = cm_display[i, j]
            # 全て黒文字
            ax.text(j, i, f'{value}社\n\n{labels[i][j]}',
                   ha='center', va='center', fontsize=14, color='black')
    
    # 精度指標なし
    
    ax.set_facecolor('#FFFFFF')
    fig.patch.set_facecolor('#FFFFFF')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph3-paper.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ graph3-paper.png を生成しました！")
    print(f"   場所: {os.path.abspath(os.path.join(output_dir, 'graph3-paper.png'))}")

if __name__ == '__main__':
    generate_graph3_paper()
