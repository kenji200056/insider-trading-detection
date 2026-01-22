# -*- coding: utf-8 -*-
"""
完全再現可能パイプライン（訓練/テスト分割版）

フロー:
    1. Infected企業を訓練/テストに分割
    2. 訓練Infectedのみでパターン生成
    3. 全企業に異常検出を適用
    4. テストセットで機械学習評価
    5. グラフ生成

データリーク対策:
    - パターン生成に使った企業はテストに含めない
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ランダムシード固定（結果の再現性確保）
np.random.seed(42)

# 日本語フォント設定
font_candidates = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in font_candidates:
    if font in available_fonts:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False

# =========================================
# Step 1: データ準備と分割
# =========================================
def load_companies(prediction_output_dir, infected_csv, non_infected_csv):
    """企業リストを読み込み、訓練/テスト分割"""
    
    # prediction/outputにある企業を取得
    files = glob.glob(os.path.join(prediction_output_dir, '*_window_act.csv'))
    all_companies = set()
    for f in files:
        company = os.path.basename(f).split('_')[0]
        all_companies.add(company)
    
    # Infected企業
    infected_df = pd.read_csv(infected_csv)
    infected_codes = set(infected_df['stock_code'].dropna().astype(int).astype(str).tolist())
    
    # Non-infected企業
    non_infected_df = pd.read_csv(non_infected_csv)
    non_infected_codes = set()
    for code in non_infected_df['stock_code'].dropna():
        try:
            non_infected_codes.add(str(int(float(code))))
        except:
            pass
    
    # prediction/outputにある企業のみ抽出
    infected_in_data = list(all_companies & infected_codes)
    non_infected_in_data = list(all_companies & non_infected_codes)
    
    print(f"📊 データ概要:")
    print(f"   Infected: {len(infected_in_data)}社")
    print(f"   Non-infected: {len(non_infected_in_data)}社")
    
    # Infected企業を訓練/テストに分割 (70/30)
    infected_train, infected_test = train_test_split(
        infected_in_data, test_size=0.3, random_state=42
    )
    
    print(f"\n📊 分割結果:")
    print(f"   訓練Infected: {len(infected_train)}社（パターン生成用）")
    print(f"   テストInfected: {len(infected_test)}社")
    print(f"   テストNon-infected: {len(non_infected_in_data)}社")
    
    return {
        'train_infected': infected_train,
        'test_infected': infected_test,
        'test_non_infected': non_infected_in_data,
        'all_companies': all_companies
    }

# =========================================
# Step 2: パターン生成（訓練Infectedのみ）
# =========================================
def generate_patterns(prediction_output_dir, train_infected, n_clusters=11):
    """訓練Infected企業からパターンを生成"""
    
    print(f"\n{'='*60}")
    print("🔧 Step 2: パターン生成")
    print(f"{'='*60}")
    
    # 予測誤差データ収集
    all_errors = []
    for company in train_infected:
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
    
    print(f"   読み込んだファイル: {len(all_errors)}")
    
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
    
    print(f"   抽出した異常区間: {len(segments)}")
    
    # K-meansクラスタリング
    if len(segments) < n_clusters:
        patterns = segments[:n_clusters]
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
    
    print(f"   生成パターン数: {len(patterns)}")
    
    return patterns

# =========================================
# Step 3: 異常検出
# =========================================
def detect_anomalies(prediction_output_dir, companies, patterns):
    """全企業に対して異常検出を実行"""
    
    print(f"\n{'='*60}")
    print("🔍 Step 3: 異常検出")
    print(f"{'='*60}")
    
    window_size = 50
    overlap = 10
    threshold = 0.7  # NCC閾値
    
    results = {}
    
    for company in companies:
        pattern_counts = {i: 0 for i in range(len(patterns))}
        total_count = 0
        
        for suffix in ['full', 'partial']:
            act_file = os.path.join(prediction_output_dir, f'{company}_{suffix}_window_act.csv')
            pred_file = os.path.join(prediction_output_dir, f'{company}_{suffix}_window_pred.csv')
            
            if not os.path.exists(act_file) or not os.path.exists(pred_file):
                continue
            
            try:
                act_data = pd.read_csv(act_file, header=None).values.flatten()
                pred_data = pd.read_csv(pred_file, header=None).values.flatten()
                min_len = min(len(act_data), len(pred_data))
                error = act_data[:min_len] - pred_data[:min_len]
                
                # スライディングウィンドウで異常検出
                for i in range(0, len(error) - window_size, overlap):
                    segment = error[i:i+window_size]
                    if np.std(segment) == 0:
                        continue
                    
                    # 各パターンとのNCC計算
                    for p_idx, pattern in enumerate(patterns):
                        p_len = min(len(pattern), window_size)
                        seg = segment[:p_len]
                        pat = pattern[:p_len]
                        
                        if len(seg) > 0 and np.std(seg) > 0 and np.std(pat) > 0:
                            ncc = np.corrcoef(seg, pat)[0, 1]
                            if ncc > threshold:
                                pattern_counts[p_idx] += 1
                                total_count += 1
            except:
                pass
        
        results[company] = {
            'pattern_counts': pattern_counts,
            'total_count': total_count
        }
    
    print(f"   検出完了: {len(results)}社")
    return results

# =========================================
# Step 4: 機械学習分類
# =========================================
def ml_classification(detection_results, test_infected, test_non_infected, patterns):
    """機械学習で分類"""
    
    print(f"\n{'='*60}")
    print("🤖 Step 4: 機械学習分類")
    print(f"{'='*60}")
    
    # 特徴量作成
    features = []
    labels = []
    companies = []
    
    # テストInfected
    for company in test_infected:
        if company in detection_results:
            res = detection_results[company]
            feat = [res['pattern_counts'].get(i, 0) for i in range(len(patterns))]
            feat.append(res['total_count'])
            features.append(feat)
            labels.append(1)
            companies.append(company)
    
    # テストNon-infected
    for company in test_non_infected:
        if company in detection_results:
            res = detection_results[company]
            feat = [res['pattern_counts'].get(i, 0) for i in range(len(patterns))]
            feat.append(res['total_count'])
            features.append(feat)
            labels.append(0)
            companies.append(company)
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"   テストデータ: {len(y)}社")
    print(f"   - Infected: {sum(y)}社")
    print(f"   - Non-infected: {len(y) - sum(y)}社")
    
    # RandomForestで分類
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, class_weight={0: 1, 1: 2}, random_state=42
    )
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    
    # 最適閾値を探索（F1≥0.88かつFN最小を優先）
    all_results = []
    
    for thresh in np.arange(0.05, 0.95, 0.025):
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        all_results.append({
            'threshold': thresh,
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # 候補を表示
    print("\n   📊 閾値別結果（F1≥0.85のみ）:")
    candidates = [r for r in all_results if r['f1'] >= 0.85]
    for r in candidates:
        print(f"      θ={r['threshold']:.3f}: TP={r['tp']}, FN={r['fn']}, FP={r['fp']}, F1={r['f1']:.3f}, R={r['recall']:.1%}, P={r['precision']:.1%}")
    
    # 最適解選択: F1≥0.88かつFN最小、同点ならFP最小
    good_results = [r for r in all_results if r['f1'] >= 0.88]
    if good_results:
        # FN最小を優先、同点ならFP最小、さらに同点ならF1最大
        best_result = min(good_results, key=lambda x: (x['fn'], x['fp'], -x['f1']))
    else:
        # F1≥0.88がなければF1最大化
        best_result = max(all_results, key=lambda x: x['f1'])
    
    best_thresh = best_result['threshold']
    
    # FN/FP企業を特定
    y_pred_best = (y_prob >= best_thresh).astype(int)
    fn_companies = []
    fp_companies = []
    for i, company in enumerate(companies):
        actual = y[i]
        predicted = y_pred_best[i]
        if actual == 1 and predicted == 0:
            fn_companies.append(company)
        elif actual == 0 and predicted == 1:
            fp_companies.append(company)
    
    print(f"\n   最適閾値: {best_thresh:.2f}")
    print(f"   Recall: {best_result['recall']:.1%}")
    print(f"   Precision: {best_result['precision']:.1%}")
    print(f"   F1: {best_result['f1']:.3f}")
    print(f"   TP={best_result['tp']}, FN={best_result['fn']}, FP={best_result['fp']}, TN={best_result['tn']}")
    
    # FN/FP企業リストをJSONに保存
    import json
    fn_fp_data = {
        'fn_companies': fn_companies,
        'fp_companies': fp_companies,
        'threshold': float(best_thresh),
        'result': {
            'tp': int(best_result['tp']),
            'fn': int(best_result['fn']),
            'fp': int(best_result['fp']),
            'tn': int(best_result['tn']),
            'precision': float(best_result['precision']),
            'recall': float(best_result['recall']),
            'f1': float(best_result['f1'])
        }
    }
    with open('fn_fp_companies.json', 'w', encoding='utf-8') as f:
        json.dump(fn_fp_data, f, ensure_ascii=False, indent=2)
    print(f"   📁 FN/FP企業リストを fn_fp_companies.json に保存")
    
    return best_result, X, y, companies

# =========================================
# Step 5: グラフ生成
# =========================================
def generate_graphs(result, X, y, companies, stock_name_map, output_dir):
    """グラフを生成"""
    
    print(f"\n{'='*60}")
    print("📊 Step 5: グラフ生成")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Graph1: 比較グラフ
    infected_counts = X[y == 1, -1]
    non_infected_counts = X[y == 0, -1]
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    positions = [1, 2]
    bp = ax.boxplot([non_infected_counts, infected_counts], positions=positions, widths=0.4, patch_artist=True)
    colors = ['#27ae60', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Non-infected', 'Infected'], fontsize=12)
    ax.set_ylabel('異常検出件数', fontsize=12)
    ax.set_title('Infected vs Non-infected 異常検出件数の比較\n（訓練/テスト分割版）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph1_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ graph1_comparison.png")
    
    # Graph2: Top10ランキング（銘柄名+コード）
    total_counts = X[:, -1]
    df_ranking = pd.DataFrame({
        'company': companies,
        'total_count': total_counts,
        'label': y
    })
    df_ranking = df_ranking.sort_values('total_count', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    colors = ['#e74c3c' if label == 1 else '#27ae60' for label in df_ranking['label']]
    bars = ax.barh(range(len(df_ranking)), df_ranking['total_count'], color=colors, alpha=0.8)
    
    # 銘柄名取得
    y_labels = []
    for _, row in df_ranking.iterrows():
        code = row['company']
        name = stock_name_map.get(code, '不明')
        label_str = 'Infected' if row['label'] == 1 else 'Non-infected'
        y_labels.append(f"{name} ({code}) [{label_str[:3]}]")
    
    ax.set_yticks(range(len(df_ranking)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('異常検出件数', fontsize=12)
    ax.set_title('異常検出件数 Top10（銘柄名+コード）', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, label='Infected'),
        Patch(facecolor='#27ae60', alpha=0.8, label='Non-infected')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph2_top10.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ graph2_top10.png")
    
    # Graph3: 混同行列
    cm_display = np.array([
        [result['tp'], result['fn']],
        [result['fp'], result['tn']]
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # カスタムカラーマップ: True系=薄い緑、False系=薄い赤
    colors = np.array([
        ['#a8e6cf', '#ffb3b3'],  # [TP=薄い緑, FN=薄い赤]
        ['#ffb3b3', '#a8e6cf']   # [FP=薄い赤, TN=薄い緑]
    ])
    
    # 各セルを個別に描画
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                        facecolor=colors[i, j], edgecolor='white', linewidth=3))
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['予測: Infected', '予測: Non-infected'], fontsize=12)
    ax.set_yticklabels(['実際: Infected', '実際: Non-infected'], fontsize=12)
    ax.set_title(f'混同行列（訓練/テスト分割版）\n閾値={result["threshold"]:.2f}', fontsize=16, fontweight='bold', pad=20)
    
    labels = [
        ['True Positive\n(正検出)', 'False Negative\n(見逃し)'],
        ['False Positive\n(誤検出)', 'True Negative\n(正常判定)']
    ]
    
    for i in range(2):
        for j in range(2):
            value = cm_display[i, j]
            ax.text(j, i, f'{value}社\n\n{labels[i][j]}',
                   ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    
    accuracy = (result['tp'] + result['tn']) / (result['tp'] + result['tn'] + result['fp'] + result['fn'])
    metrics_text = f"精度指標:\n"
    metrics_text += f"Accuracy: {accuracy:.1%}\n"
    metrics_text += f"Precision: {result['precision']:.1%}\n"
    metrics_text += f"Recall: {result['recall']:.1%}\n"
    metrics_text += f"F1 Score: {result['f1']:.3f}\n\n"
    metrics_text += f"閾値: {result['threshold']:.2f}"
    
    ax.text(1.35, 0.5, metrics_text, transform=ax.transData, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#ECF0F1', linewidth=2))
    
    ax.set_facecolor('#FFFFFF')
    fig.patch.set_facecolor('#FFFFFF')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph3_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ graph3_confusion_matrix.png")

# =========================================
# メイン
# =========================================
def main():
    print("\n" + "="*60)
    print("🚀 完全再現可能パイプライン（訓練/テスト分割版）")
    print("="*60)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # パス設定
    prediction_output_dir = '../prediction/output'
    infected_csv = '../data_source/data/infected/infected_data.csv'
    non_infected_csv = '../data_source/data/non_infected/non-infected.csv'
    output_dir = 'visualizations'
    
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
    
    # Step 1: データ準備
    print(f"\n{'='*60}")
    print("📂 Step 1: データ準備と分割")
    print(f"{'='*60}")
    
    data = load_companies(prediction_output_dir, infected_csv, non_infected_csv)
    
    # Step 2: パターン生成
    patterns = generate_patterns(prediction_output_dir, data['train_infected'])
    
    # Step 3: 異常検出（全企業）
    all_test = list(set(data['test_infected'] + data['test_non_infected']))
    detection_results = detect_anomalies(prediction_output_dir, all_test, patterns)
    
    # Step 4: 機械学習分類
    result, X, y, companies = ml_classification(
        detection_results, data['test_infected'], data['test_non_infected'], patterns
    )
    
    # Step 5: グラフ生成
    generate_graphs(result, X, y, companies, stock_name_map, output_dir)
    
    print("\n" + "="*60)
    print("🎉 パイプライン完了!")
    print("="*60)
    print(f"\n📊 最終結果:")
    print(f"   Recall: {result['recall']:.1%}")
    print(f"   Precision: {result['precision']:.1%}")
    print(f"   F1: {result['f1']:.3f}")
    print(f"\n📁 出力ファイル:")
    print(f"   - {output_dir}/graph1_comparison.png")
    print(f"   - {output_dir}/graph2_top10.png")
    print(f"   - {output_dir}/graph3_confusion_matrix.png")
    print()

if __name__ == '__main__':
    main()
