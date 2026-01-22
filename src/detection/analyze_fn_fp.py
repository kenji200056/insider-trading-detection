# -*- coding: utf-8 -*-
"""
FN/FP分析・可視化スクリプト（JSON読み込み版）

main_detection.pyで保存されたfn_fp_companies.jsonからFN/FP企業を読み込み、
各ケースの詳細分析グラフを生成する。

使用方法:
    1. まずmain_detection.pyを実行してfn_fp_companies.jsonを作成
    2. その後このスクリプトを実行

出力:
    - visualizations/fn_analysis_X.png: 各FNケースの詳細分析
    - visualizations/fp_analysis_X.png: 各FPケースの詳細分析
    - visualizations/fn_fp_summary.png: 全件の比較概要
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定
font_candidates = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in font_candidates:
    if font in available_fonts:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False


def load_error_data(prediction_output_dir, company):
    """企業の予測誤差データを読み込む"""
    for suffix in ['full', 'partial']:
        act_file = os.path.join(prediction_output_dir, f'{company}_{suffix}_window_act.csv')
        pred_file = os.path.join(prediction_output_dir, f'{company}_{suffix}_window_pred.csv')
        
        if os.path.exists(act_file) and os.path.exists(pred_file):
            try:
                act_data = pd.read_csv(act_file, header=None).values.flatten()
                pred_data = pd.read_csv(pred_file, header=None).values.flatten()
                min_len = min(len(act_data), len(pred_data))
                return act_data[:min_len] - pred_data[:min_len]
            except:
                pass
    return None


def generate_patterns(prediction_output_dir, n_clusters=11):
    """訓練Infected企業からパターンを生成（簡易版）"""
    
    # 全企業から予測誤差を収集
    files = glob.glob(os.path.join(prediction_output_dir, '*_full_window_act.csv'))
    
    segments = []
    window_size = 50
    
    for act_file in files[:50]:  # 処理簡略化のため50社まで
        company = os.path.basename(act_file).split('_')[0]
        error = load_error_data(prediction_output_dir, company)
        
        if error is not None and len(error) >= window_size:
            threshold = np.percentile(np.abs(error), 90)
            for i in range(len(error) - window_size):
                segment = error[i:i+window_size]
                if np.max(np.abs(segment)) > threshold and np.std(segment) > 0:
                    normalized = (segment - np.mean(segment)) / np.std(segment)
                    segments.append(normalized)
    
    if len(segments) < n_clusters:
        return [np.zeros(window_size) for _ in range(n_clusters)]
    
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
        else:
            patterns.append(np.zeros(window_size))
    
    return patterns


def calculate_ncc_scores(error_data, patterns):
    """各パターンとの最大NCCスコアを計算"""
    window_size = 50
    max_ncc_scores = {i: 0.0 for i in range(len(patterns))}
    
    if error_data is None or len(error_data) < window_size:
        return max_ncc_scores, 0
    
    overlap = 10
    total_matches = 0
    
    for i in range(0, len(error_data) - window_size, overlap):
        segment = error_data[i:i+window_size]
        if np.std(segment) == 0:
            continue
        
        for p_idx, pattern in enumerate(patterns):
            p_len = min(len(pattern), window_size)
            seg = segment[:p_len]
            pat = pattern[:p_len]
            
            if len(seg) > 0 and np.std(seg) > 0 and np.std(pat) > 0:
                ncc = np.corrcoef(seg, pat)[0, 1]
                max_ncc_scores[p_idx] = max(max_ncc_scores[p_idx], ncc)
                if ncc > 0.7:
                    total_matches += 1
    
    return max_ncc_scores, total_matches


def generate_analysis_graph(company, error_data, patterns, max_ncc_scores, total_matches,
                            stock_name_map, case_type, case_number, output_dir):
    """各ケースの詳細分析グラフを生成（3パネル版）"""
    
    if error_data is None:
        print(f"   ⚠️ {company}: 予測誤差データなし")
        return None
    
    # 最も類似するパターンを特定
    best_pattern_idx = max(max_ncc_scores, key=max_ncc_scores.get)
    best_ncc = max_ncc_scores[best_pattern_idx]
    best_pattern = patterns[best_pattern_idx]
    
    # 銘柄名取得
    stock_name = stock_name_map.get(company, '不明')
    
    # ケースタイプのラベル
    if case_type == 'FN':
        case_label = f'【FN-{case_number}】{stock_name} ({company}) - Infected（見逃し）'
        title_color = '#e74c3c'
    else:
        case_label = f'【FP-{case_number}】{stock_name} ({company}) - Non-infected（誤検出）'
        title_color = '#e67e22'
    
    window_size = 50
    
    # 最も類似する部分を見つける
    if np.std(error_data) > 0:
        error_normalized = (error_data - np.mean(error_data)) / np.std(error_data)
    else:
        error_normalized = error_data
    
    best_match_start = 0
    best_match_ncc = -1
    
    for i in range(len(error_normalized) - window_size):
        segment = error_normalized[i:i+window_size]
        if np.std(segment) > 0:
            pat = best_pattern[:min(len(best_pattern), window_size)]
            seg = segment[:len(pat)]
            if len(seg) == len(pat) and np.std(seg) > 0:
                ncc = np.corrcoef(seg, pat)[0, 1]
                if ncc > best_match_ncc:
                    best_match_ncc = ncc
                    best_match_start = i
    
    # グラフ作成（3パネル）
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 0.8], hspace=0.35, wspace=0.25)
    
    # ===== 上部左: 予測誤差全体 =====
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(error_data, color='#3498db', linewidth=1.5, label='予測誤差 $e_t = p_t - \\hat{p}_t$', alpha=0.8)
    
    # マッチング領域をハイライト
    if best_match_start + window_size <= len(error_data):
        ax1.axvspan(best_match_start, best_match_start + window_size, alpha=0.25, color='#ffeb3b', label='最類似マッチング領域')
        ax1.axvline(x=best_match_start, color='#e74c3c', linestyle='-', linewidth=2, alpha=0.8)
        ax1.axvline(x=best_match_start + window_size, color='#e74c3c', linestyle='-', linewidth=2, alpha=0.8)
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('時間 (日)', fontsize=12)
    ax1.set_ylabel('予測誤差 ($e_t$)', fontsize=12)
    ax1.set_title(case_label, fontsize=14, fontweight='bold', color=title_color, pad=15)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== 中部左: マッチング領域の拡大比較 =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    if best_match_start + window_size <= len(error_data):
        segment_actual = error_data[best_match_start:best_match_start+window_size]
        
        # 正規化して比較
        seg_normalized = (segment_actual - np.mean(segment_actual)) / np.std(segment_actual) if np.std(segment_actual) > 0 else segment_actual
        pat_normalized = best_pattern[:window_size]
        
        x_range = np.arange(window_size)
        ax2.plot(x_range, seg_normalized, color='#3498db', linewidth=3, label='予測誤差（正規化）', marker='o', markersize=3)
        ax2.plot(x_range, pat_normalized, color='#e74c3c', linewidth=3, label=f'パターン P{best_pattern_idx}', linestyle='--', marker='s', markersize=3)
        ax2.fill_between(x_range, seg_normalized, pat_normalized, alpha=0.2, color='gray')
    else:
        ax2.text(0.5, 0.5, 'データ不足', ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    
    ax2.set_xlabel('時間 (日) - マッチング領域内', fontsize=11)
    ax2.set_ylabel('正規化値', fontsize=11)
    ax2.set_title(f'マッチング領域拡大（NCC = {best_ncc:.3f}）', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ===== 中部右: パターン単体表示 =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    x_range = np.arange(len(best_pattern[:window_size]))
    ax3.plot(x_range, best_pattern[:window_size], color='#e74c3c', linewidth=3, marker='o', markersize=4)
    ax3.fill_between(x_range, 0, best_pattern[:window_size], alpha=0.3, color='#e74c3c')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    ax3.set_xlabel('時間 (日)', fontsize=11)
    ax3.set_ylabel('正規化予測誤差', fontsize=11)
    ax3.set_title(f'最類似パターン P{best_pattern_idx} の形状', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ===== 下部: パターン別NCCスコア =====
    ax4 = fig.add_subplot(gs[2, :])
    
    pattern_names = [f'P{i}' for i in range(len(patterns))]
    ncc_values = [max_ncc_scores.get(i, 0) for i in range(len(patterns))]
    colors = ['#e74c3c' if v >= 0.7 else '#3498db' for v in ncc_values]
    
    bars = ax4.bar(pattern_names, ncc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='閾値 (τ=0.7)')
    
    # 最大値をハイライト
    bars[best_pattern_idx].set_edgecolor('#000')
    bars[best_pattern_idx].set_linewidth(3)
    
    ax4.set_xlabel('パターン', fontsize=11)
    ax4.set_ylabel('最大NCCスコア', fontsize=11)
    ax4.set_title('各パターンとの最大相関（NCCスコア）', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ===== 考察テキストボックス =====
    if case_type == 'FN':
        if best_ncc < 0.5:
            analysis_text = f"考察: 最大NCC={best_ncc:.3f}と非常に低く、インサイダー取引の典型的パターンと異なる形状。取引期間が短い可能性。"
        elif best_ncc < 0.7:
            analysis_text = f"考察: 最大NCC={best_ncc:.3f}で閾値0.7に僅かに届かず。パターンは類似しているが検出漏れ。"
        else:
            analysis_text = f"考察: NCC={best_ncc:.3f}で閾値超過だが、マッチング回数{total_matches}回が少なく分類器で除外。"
    else:
        analysis_text = f"考察: 最大NCC={best_ncc:.3f}、マッチング{total_matches}回。決算発表等の合法イベントがインサイダーパターンに類似した可能性。"
    
    # 統計情報ボックス
    stats_text = f"総マッチング回数: {total_matches}回 | 最大NCC: {best_ncc:.3f} (P{best_pattern_idx}) | 閾値超過: {sum(1 for v in ncc_values if v >= 0.7)}個"
    
    fig.text(0.5, 0.02, f"{stats_text}\n{analysis_text}", ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.95, edgecolor='#999', linewidth=1.5),
             wrap=True)
    
    plt.subplots_adjust(bottom=0.12)
    
    # ファイル保存
    filename = f'{case_type.lower()}_analysis_{case_number}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {filename}")
    
    return {
        'company': company,
        'stock_name': stock_name,
        'case_type': case_type,
        'best_ncc': best_ncc,
        'best_pattern': best_pattern_idx,
        'total_count': total_matches
    }


def generate_summary_graph(analysis_results, output_dir):
    """全ケースの比較概要グラフを生成"""
    
    n_results = len(analysis_results)
    if n_results == 0:
        print("   ⚠️ 分析結果がありません")
        return
    
    n_cols = min(4, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_results == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fn_count = 0
    fp_count = 0
    
    for i, result in enumerate(analysis_results):
        ax = axes[i]
        
        if result['case_type'] == 'FN':
            fn_count += 1
            color = '#e74c3c'
            title = f"FN-{fn_count}: {result['stock_name']}\n({result['company']})"
        else:
            fp_count += 1
            color = '#e67e22'
            title = f"FP-{fp_count}: {result['stock_name']}\n({result['company']})"
        
        ax.barh(['最大NCC', '閾値'], [result['best_ncc'], 0.7], 
                color=[color, 'gray'], alpha=0.7)
        ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel('NCCスコア', fontsize=9)
        
        status = "検出失敗" if result['case_type'] == 'FN' else "誤検出"
        ax.text(0.5, 0.5, f"マッチング: {result['total_count']}回\n{status}",
                transform=ax.transAxes, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for i in range(len(analysis_results), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'FN/FP分析サマリー（FN: {fn_count}件, FP: {fp_count}件）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'fn_fp_summary.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ fn_fp_summary.png")


def main():
    print("\n" + "="*60)
    print("🔍 FN/FP分析・可視化スクリプト（JSON読み込み版）")
    print("="*60)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # パス設定
    prediction_output_dir = '../prediction/output'
    infected_csv = '../data_source/data/infected/infected_data.csv'
    non_infected_csv = '../data_source/data/non_infected/non-infected.csv'
    output_dir = 'visualizations'
    json_file = 'fn_fp_companies.json'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # FN/FP企業リストをJSONから読み込み
    if not os.path.exists(json_file):
        print(f"\n   ❌ エラー: {json_file} が見つかりません")
        print("   先にmain_detection.pyを実行してください")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        fn_fp_data = json.load(f)
    
    fn_companies = fn_fp_data['fn_companies']
    fp_companies = fn_fp_data['fp_companies']
    result = fn_fp_data['result']
    
    print(f"\n📊 分類結果（JSONから読み込み）:")
    print(f"   TP={result['tp']}, FN={result['fn']}, FP={result['fp']}, TN={result['tn']}")
    print(f"   Precision: {result['precision']:.1%}, Recall: {result['recall']:.1%}, F1: {result['f1']:.3f}")
    print(f"\n   FN企業: {len(fn_companies)}社 - {fn_companies}")
    print(f"   FP企業: {len(fp_companies)}社 - {fp_companies}")
    
    # 銘柄名マップを作成
    stock_name_map = {}
    infected_df = pd.read_csv(infected_csv)
    for _, row in infected_df.iterrows():
        code = str(int(row['stock_code'])) if pd.notna(row['stock_code']) else None
        if code:
            stock_name_map[code] = row['stock_name']
    
    non_infected_df = pd.read_csv(non_infected_csv)
    for _, row in non_infected_df.iterrows():
        try:
            code = str(int(float(row['stock_code'])))
            if code not in stock_name_map:
                stock_name_map[code] = row['stock_name']
        except:
            pass
    
    print(f"\n📚 銘柄名マップ: {len(stock_name_map)}社読み込み")
    
    # パターン生成
    print(f"\n{'='*60}")
    print("🔧 パターン生成")
    print(f"{'='*60}")
    
    patterns = generate_patterns(prediction_output_dir)
    print(f"   生成パターン数: {len(patterns)}")
    
    # 個別グラフ生成
    print(f"\n{'='*60}")
    print("📊 個別グラフ生成")
    print(f"{'='*60}")
    
    analysis_results = []
    
    for i, company in enumerate(fn_companies):
        error_data = load_error_data(prediction_output_dir, company)
        max_ncc_scores, total_matches = calculate_ncc_scores(error_data, patterns)
        result = generate_analysis_graph(
            company, error_data, patterns, max_ncc_scores, total_matches,
            stock_name_map, 'FN', i+1, output_dir
        )
        if result:
            analysis_results.append(result)
    
    for i, company in enumerate(fp_companies):
        error_data = load_error_data(prediction_output_dir, company)
        max_ncc_scores, total_matches = calculate_ncc_scores(error_data, patterns)
        result = generate_analysis_graph(
            company, error_data, patterns, max_ncc_scores, total_matches,
            stock_name_map, 'FP', i+1, output_dir
        )
        if result:
            analysis_results.append(result)
    
    # サマリーグラフ生成
    print(f"\n{'='*60}")
    print("📊 サマリーグラフ生成")
    print(f"{'='*60}")
    
    if analysis_results:
        generate_summary_graph(analysis_results, output_dir)
    
    print("\n" + "="*60)
    print("🎉 FN/FP分析完了!")
    print("="*60)
    print(f"\n📁 出力ファイル:")
    for i in range(len(fn_companies)):
        print(f"   - {output_dir}/fn_analysis_{i+1}.png")
    for i in range(len(fp_companies)):
        print(f"   - {output_dir}/fp_analysis_{i+1}.png")
    print(f"   - {output_dir}/fn_fp_summary.png")
    print()


if __name__ == '__main__':
    main()
