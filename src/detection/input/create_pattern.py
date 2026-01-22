# -*- coding: utf-8 -*-
"""
日本市場向け異常パターン生成スクリプト

このスクリプトは、日本のInfected企業の予測誤差データから
異常パターンを抽出し、pattern.csv と pattern_sizes.csv を生成します。

使用方法:
    cd detection/input
    python create_pattern.py

入力:
    data/*.csv - Infected企業の予測誤差データ
    
出力:
    pattern_jp.csv - 日本市場向け異常パターン
    pattern_sizes_jp.csv - パターンサイズ
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_prediction_data(data_dir='data'):
    """
    data/ディレクトリからInfected企業の予測データを読み込む
    """
    all_data = []
    files = glob.glob(os.path.join(data_dir, '*_window_act.csv'))
    
    print(f"📂 {len(files)}件のファイルを検出")
    
    for file in files:
        try:
            # 実際値を読み込み
            act_data = pd.read_csv(file, header=None).values.flatten()
            
            # 対応する予測値を読み込み
            pred_file = file.replace('_window_act.csv', '_window_pred.csv')
            if os.path.exists(pred_file):
                pred_data = pd.read_csv(pred_file, header=None).values.flatten()
                
                # 予測誤差を計算
                min_len = min(len(act_data), len(pred_data))
                error = act_data[:min_len] - pred_data[:min_len]
                
                all_data.append({
                    'file': os.path.basename(file),
                    'error': error,
                    'length': len(error)
                })
                print(f"  ✅ {os.path.basename(file)}: {len(error)}日分")
        except Exception as e:
            print(f"  ⚠️ エラー: {file} - {e}")
    
    return all_data

def extract_anomaly_segments(data_list, window_size=50, threshold_percentile=90):
    """
    予測誤差が大きい区間を抽出
    """
    segments = []
    
    for data in data_list:
        error = data['error']
        
        # 絶対誤差の閾値を計算
        threshold = np.percentile(np.abs(error), threshold_percentile)
        
        # 閾値を超える区間を検出
        for i in range(len(error) - window_size):
            segment = error[i:i+window_size]
            
            # 区間内の最大誤差が閾値を超えるか
            if np.max(np.abs(segment)) > threshold:
                # 正規化
                if np.std(segment) > 0:
                    normalized = (segment - np.mean(segment)) / np.std(segment)
                    segments.append(normalized)
    
    print(f"📊 抽出された異常区間: {len(segments)}件")
    return segments

def cluster_patterns(segments, n_clusters=11):
    """
    類似パターンをクラスタリングして代表パターンを抽出
    """
    if len(segments) < n_clusters:
        print(f"⚠️ 区間数が少ないため、全区間を使用 ({len(segments)}件)")
        return segments[:min(len(segments), n_clusters)]
    
    # 同じ長さにパディング
    max_len = max(len(s) for s in segments)
    padded = np.array([np.pad(s, (0, max_len - len(s)), 'constant') for s in segments])
    
    # 標準化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(padded)
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    
    # 各クラスタの代表パターン(中心に最も近いもの)を選択
    patterns = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(scaled[cluster_indices] - center, axis=1)
            best_idx = cluster_indices[np.argmin(distances)]
            patterns.append(segments[best_idx])
    
    print(f"📊 クラスタリング完了: {len(patterns)}パターン")
    return patterns

def save_patterns(patterns, output_prefix='pattern_jp'):
    """
    パターンをCSVファイルに保存
    """
    # パターンサイズを取得
    sizes = [len(p) for p in patterns]
    
    # 最大長にパディング
    max_len = max(sizes)
    padded_patterns = []
    for p in patterns:
        padded = np.pad(p, (0, max_len - len(p)), 'constant', constant_values=0)
        padded_patterns.append(padded)
    
    # pattern.csv を保存
    pattern_df = pd.DataFrame(padded_patterns)
    pattern_df.to_csv(f'{output_prefix}.csv', index=False, header=False)
    print(f"✅ 保存: {output_prefix}.csv ({len(patterns)}パターン × {max_len}日)")
    
    # pattern_sizes.csv を保存
    sizes_df = pd.DataFrame([sizes])
    sizes_df.to_csv(f'{output_prefix}_sizes.csv', index=False, header=False)
    print(f"✅ 保存: {output_prefix}_sizes.csv")
    
    return f'{output_prefix}.csv', f'{output_prefix}_sizes.csv'

def main():
    print("\n" + "="*60)
    print("🔧 日本市場向け異常パターン生成")
    print("="*60 + "\n")
    
    # スクリプトのディレクトリに移動
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # データを読み込み
    print("📂 データ読み込み中...")
    data_list = load_prediction_data('data')
    
    if len(data_list) == 0:
        print("\n⚠️ データが見つかりません!")
        print("detection/input/data/ にInfected企業の予測データをコピーしてください。")
        print("\n例:")
        print("  cp ../../../prediction/output/1619_full_window_*.csv data/")
        return
    
    # 異常区間を抽出
    print("\n📊 異常区間抽出中...")
    segments = extract_anomaly_segments(data_list)
    
    if len(segments) == 0:
        print("⚠️ 異常区間が検出されませんでした。閾値を調整してください。")
        return
    
    # クラスタリング
    print("\n🔄 パターンクラスタリング中...")
    n_clusters = min(11, len(segments))
    patterns = cluster_patterns(segments, n_clusters=n_clusters)
    
    # 保存
    print("\n💾 パターン保存中...")
    save_patterns(patterns)
    
    print("\n" + "="*60)
    print("🎉 パターン生成完了!")
    print("="*60)
    print("\n次のステップ:")
    print("1. pattern_jp.csv を pattern.csv として使用")
    print("2. pattern_sizes_jp.csv を pattern_sizes.csv として使用")
    print("3. detection を再実行して精度を比較")
    print()

if __name__ == '__main__':
    main()
