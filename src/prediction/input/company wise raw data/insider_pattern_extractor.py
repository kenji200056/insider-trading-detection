#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
insider_pattern_extractor.py
- infected.csv からインサイダー取引期間 (period) と銘柄コード (stock_code) を取得
- その期間の予測誤差データからパターンを抽出
- 可変サイズ (ENと同様) でパターンを生成

使用方法:
    cd jp/prediction/input/company\ wise\ raw\ data
    python insider_pattern_extractor.py

出力:
    ../../detection/input/pattern_jp_v6.csv
    ../../detection/input/pattern_jp_v6_sizes.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INFECTED_CSV = os.path.join(BASE_DIR, 'data_source/data/infected/infected.csv')
PREDICTION_OUTPUT = os.path.join(BASE_DIR, 'prediction/output')
DETECTION_INPUT = os.path.join(BASE_DIR, 'detection/input')

def parse_period(period_str):
    """
    period文字列を解析して開始日・終了日・日数を返す
    
    例:
      "2005/10/04-2005/10/06" → (2005-10-04, 2005-10-06, 3日間)
      "2007/03/19" → (2007-03-19, 2007-03-19, 1日)
    """
    if pd.isna(period_str) or str(period_str).strip() == "":
        return None, None, 0
    
    s = str(period_str).strip()
    try:
        if '-' in s and s.count('/') == 4:
            start_s, end_s = s.split('-')
            start_date = datetime.strptime(start_s.strip(), '%Y/%m/%d')
            end_date = datetime.strptime(end_s.strip(), '%Y/%m/%d')
            days = (end_date - start_date).days + 1
            return start_date, end_date, days
        else:
            dt = datetime.strptime(s, '%Y/%m/%d')
            return dt, dt, 1
    except Exception as e:
        print(f"  日付パースエラー ({s}): {e}")
        return None, None, 0

def calculate_pattern_size(period_days):
    """
    インサイダー期間に基づいてパターンサイズを決定
    
    ENパターンサイズ: 21,50,36,50,21,50,41,46,36,42,50
    → 最小21日、最大50日
    
    方針: 期間 + 前後マージン(各10日)
    """
    pattern_size = period_days + 20
    return max(21, min(50, pattern_size))

def load_prediction_data(stock_code):
    """
    予測データを読み込み、差分を計算
    """
    act_file = os.path.join(PREDICTION_OUTPUT, f'{stock_code}_partial_window_act.csv')
    pred_file = os.path.join(PREDICTION_OUTPUT, f'{stock_code}_partial_window_pred.csv')
    
    if not os.path.exists(act_file) or not os.path.exists(pred_file):
        return None
    
    try:
        act = pd.read_csv(act_file, header=None).values.flatten()
        pred = pd.read_csv(pred_file, header=None).values.flatten()
        min_len = min(len(act), len(pred))
        error = act[:min_len] - pred[:min_len]
        return error
    except Exception as e:
        print(f"  読み込みエラー ({stock_code}): {e}")
        return None

def extract_patterns_from_infected():
    """
    全Infected企業からインサイダー期間ベースのパターンを抽出
    """
    print("\n" + "="*60)
    print("🔧 インサイダー期間ベース パターン抽出 (v6)")
    print("="*60)
    
    # infected.csv読み込み
    print(f"\n📂 読み込み: {INFECTED_CSV}")
    df = pd.read_csv(INFECTED_CSV)
    
    # class=1 のみ (確認済みInfected)
    infected_df = df[df['class'] == 1]
    print(f"   Infected企業: {len(infected_df)}社")
    
    patterns = []
    sizes = []
    pattern_info = []
    
    print("\n📊 パターン抽出中...")
    for _, row in infected_df.iterrows():
        stock_code = str(row['stock_code'])
        period_str = row['period']
        
        # 期間を解析
        start_dt, end_dt, period_days = parse_period(period_str)
        if period_days == 0:
            continue
        
        # パターンサイズ決定
        pattern_size = calculate_pattern_size(period_days)
        
        # 予測誤差データ読み込み
        error = load_prediction_data(stock_code)
        if error is None:
            continue
        
        # partialデータの末尾からパターン抽出
        # (partialはインサイダー期間±100日なので、末尾が期間後)
        if len(error) >= pattern_size:
            # 末尾からpattern_size分を抽出
            pattern = error[-pattern_size:]
            patterns.append(pattern)
            sizes.append(pattern_size)
            
            max_err = np.max(np.abs(pattern))
            pattern_info.append({
                'code': stock_code,
                'period_days': period_days,
                'size': pattern_size,
                'max_error': max_err
            })
            
            print(f"   ✅ {stock_code}: 期間{period_days}日 → サイズ{pattern_size}日, 最大誤差{max_err:.2f}")
    
    print(f"\n📊 抽出パターン: {len(patterns)}件")
    return patterns, sizes, pattern_info

def cluster_and_select_patterns(patterns, sizes, n_clusters=11):
    """
    クラスタリングして代表パターンを選定
    可変サイズのまま保持
    """
    if len(patterns) < n_clusters:
        print(f"⚠️ パターン数が少ないため、全{len(patterns)}件を使用")
        return patterns, sizes
    
    # パディングしてクラスタリング用配列を作成
    max_size = max(len(p) for p in patterns)
    padded = np.array([np.pad(p, (0, max_size - len(p)), 'constant') for p in patterns])
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(padded)
    
    # 各クラスタから最大誤差のパターンを選定
    selected_patterns = []
    selected_sizes = []
    
    print(f"\n📊 クラスタリング: {n_clusters}クラスタ")
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        
        # クラスタ内で最大誤差を持つものを選択
        best_idx = max(cluster_indices, 
                      key=lambda idx: np.max(np.abs(patterns[idx])))
        
        selected_patterns.append(patterns[best_idx])
        selected_sizes.append(sizes[best_idx])
        
        print(f"   Cluster {i}: サイズ{sizes[best_idx]}日, 最大誤差{np.max(np.abs(patterns[best_idx])):.2f}")
    
    return selected_patterns, selected_sizes

def save_patterns(patterns, sizes, output_prefix='pattern_jp_v6'):
    """
    可変サイズパターンを保存 (ENと同形式)
    """
    if len(patterns) == 0:
        print("⚠️ パターンがありません")
        return
    
    # 最大サイズでパディング
    max_size = max(len(p) for p in patterns)
    padded = []
    for p in patterns:
        padded_p = np.pad(p, (0, max_size - len(p)), 'constant', constant_values=0)
        padded.append(padded_p)
    
    # パターン保存
    pattern_file = os.path.join(DETECTION_INPUT, f'{output_prefix}.csv')
    pd.DataFrame(padded).to_csv(pattern_file, index=False, header=False)
    print(f"\n✅ 保存: {pattern_file}")
    print(f"   {len(patterns)}パターン × 最大{max_size}日")
    
    # サイズ保存
    sizes_file = os.path.join(DETECTION_INPUT, f'{output_prefix}_sizes.csv')
    pd.DataFrame([sizes]).to_csv(sizes_file, index=False, header=False)
    print(f"✅ 保存: {sizes_file}")
    print(f"   サイズ: {sizes}")
    
    # 統計表示
    print(f"\n📊 パターン統計:")
    for i, (p, s) in enumerate(zip(patterns, sizes)):
        print(f"   Pattern {i+1}: サイズ{s}日, 最大{np.max(p):.2f}, 最小{np.min(p):.2f}")

def main():
    # パターン抽出
    patterns, sizes, info = extract_patterns_from_infected()
    
    if len(patterns) == 0:
        print("\n❌ パターンが抽出できませんでした")
        return
    
    # クラスタリングして代表選定
    selected_patterns, selected_sizes = cluster_and_select_patterns(patterns, sizes)
    
    # 保存
    save_patterns(selected_patterns, selected_sizes)
    
    print("\n" + "="*60)
    print("🎉 v6パターン生成完了!")
    print("="*60)
    print("\n次のステップ:")
    print("1. cd ../../detection/input")
    print("2. cp pattern_jp_v6.csv pattern.csv")
    print("3. cp pattern_jp_v6_sizes.csv pattern_sizes.csv")
    print("4. cd .. && python run_batch.py")
    print()

if __name__ == '__main__':
    main()
