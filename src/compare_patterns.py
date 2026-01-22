# -*- coding: utf-8 -*-
"""
正常企業（Non-infected）からパターンを抽出して比較
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import glob
from sklearn.cluster import KMeans

# 日本語フォント設定
font_candidates = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in font_candidates:
    if font in available_fonts:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False

def extract_patterns(prediction_output_dir, company_list, n_clusters=11, label=""):
    """企業リストからパターンを抽出"""
    
    all_errors = []
    for company in company_list:
        for suffix in ['full', 'partial']:
            act_file = os.path.join(prediction_output_dir, f'{company}_{suffix}_window_act.csv')
            pred_file = os.path.join(prediction_output_dir, f'{company}_{suffix}_window_pred.csv')
            
            if os.path.exists(act_file) and os.path.exists(pred_file):
                try:
                    act_data = pd.read_csv(act_file, header=None).values.flatten()
                    pred_data = pd.read_csv(pred_file, header=None).values.flatten()
                    min_len = min(len(act_data), len(pred_data))
                    error = act_data[:min_len] - pred_data[:min_len]
                    all_errors.append({'company': company, 'error': error})
                except:
                    pass
    
    print(f"  📂 {label}: {len(all_errors)}ファイル読み込み")
    
    # 異常区間抽出
    window_size = 50
    segments = []
    for data in all_errors:
        error = data['error']
        threshold = np.percentile(np.abs(error), 90)
        for i in range(len(error) - window_size):
            segment = error[i:i+window_size]
            if np.max(np.abs(segment)) > threshold and np.std(segment) > 0:
                normalized = (segment - np.mean(segment)) / np.std(segment)
                segments.append(normalized)
    
    print(f"  📊 抽出した異常区間: {len(segments)}")
    
    # K-meansクラスタリング
    if len(segments) < n_clusters:
        patterns = segments[:n_clusters] if segments else []
    else:
        padded = np.array([np.pad(s, (0, window_size - len(s)), 'constant') for s in segments])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(padded)
        
        patterns = []
        for i in range(n_clusters):
            cluster_idx = np.where(labels == i)[0]
            if len(cluster_idx) > 0:
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(padded[cluster_idx] - center, axis=1)
                best = cluster_idx[np.argmin(distances)]
                patterns.append(segments[best])
    
    print(f"  ✅ 生成パターン数: {len(patterns)}")
    return patterns

def visualize_comparison(infected_patterns, non_infected_patterns, output_file):
    """Infected vs Non-infected パターン比較"""
    
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), dpi=300)
    
    # 上段: Infected (インサイダー取引企業)
    for i in range(min(6, len(infected_patterns))):
        ax = axes[0, i]
        pattern = infected_patterns[i]
        days = np.arange(1, len(pattern) + 1)
        ax.plot(days, pattern, 'r-', linewidth=2)
        ax.fill_between(days, 0, pattern, alpha=0.3, color='red')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f'Infected P{i+1}', fontsize=10, fontweight='bold', color='red')
        ax.set_xlabel('日数', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # 下段: Non-infected (正常企業)
    for i in range(min(6, len(non_infected_patterns))):
        ax = axes[1, i]
        pattern = non_infected_patterns[i]
        days = np.arange(1, len(pattern) + 1)
        ax.plot(days, pattern, 'g-', linewidth=2)
        ax.fill_between(days, 0, pattern, alpha=0.3, color='green')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f'Non-infected P{i+1}', fontsize=10, fontweight='bold', color='green')
        ax.set_xlabel('日数', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    axes[0, 0].set_ylabel('正規化誤差\n（インサイダー取引）', fontsize=10, color='red')
    axes[1, 0].set_ylabel('正規化誤差\n（正常企業）', fontsize=10, color='green')
    
    plt.suptitle('インサイダー取引企業 vs 正常企業 の代表パターン比較', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 保存: {output_file}")

def main():
    print("\n" + "="*60)
    print("📊 Infected vs Non-infected パターン比較")
    print("="*60 + "\n")
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    prediction_output_dir = 'prediction/output'
    infected_csv = 'data_source/data/infected/infected_data.csv'
    non_infected_csv = 'data_source/data/non_infected/non-infected.csv'
    
    # 企業リスト取得
    files = glob.glob(os.path.join(prediction_output_dir, '*_window_act.csv'))
    all_companies = set(os.path.basename(f).split('_')[0] for f in files)
    
    infected_df = pd.read_csv(infected_csv)
    infected_codes = set(infected_df['stock_code'].dropna().astype(int).astype(str).tolist())
    
    non_infected_df = pd.read_csv(non_infected_csv)
    non_infected_codes = set()
    for code in non_infected_df['stock_code'].dropna():
        try:
            non_infected_codes.add(str(int(float(code))))
        except:
            pass
    
    infected_in_data = list(all_companies & infected_codes)
    non_infected_in_data = list(all_companies & non_infected_codes)
    
    print(f"📂 Infected企業: {len(infected_in_data)}社")
    print(f"📂 Non-infected企業: {len(non_infected_in_data)}社")
    
    # パターン抽出
    print("\n🔧 Infectedからパターン抽出...")
    infected_patterns = extract_patterns(prediction_output_dir, infected_in_data, 6, "Infected")
    
    print("\n🔧 Non-infectedからパターン抽出...")
    non_infected_patterns = extract_patterns(prediction_output_dir, non_infected_in_data, 6, "Non-infected")
    
    # 比較可視化
    print("\n📊 比較グラフ生成...")
    output_file = 'detection/errors/pattern_comparison.png'
    visualize_comparison(infected_patterns, non_infected_patterns, output_file)
    
    print("\n🎉 完了!")

if __name__ == '__main__':
    main()
