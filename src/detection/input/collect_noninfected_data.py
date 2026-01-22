# -*- coding: utf-8 -*-
"""
Non-infected企業データ収集スクリプト

このスクリプトは、確実にNon-infected(class=0)と判定された企業の
Partialデータ(正常期間)をランダムに選択してコピーします。

目的: Non-infectedの「正常パターン」を学習し、
      Infected企業がこのパターンから「逸脱」しているかを検出

使用方法:
    cd detection/input
    python collect_noninfected_data.py

オプション:
    --count N   : コピーする企業数 (デフォルト: 10)
"""

import os
import glob
import shutil
import random
import argparse
import pandas as pd

def get_verified_noninfected_companies(litigation_dir):
    """
    litigation/dataからclass=0のNon-infected企業を取得
    """
    non_infected_file = os.path.join(litigation_dir, 'non_infected', 'non-infected.csv')
    
    if not os.path.exists(non_infected_file):
        print(f"❌ ファイルが見つかりません: {non_infected_file}")
        return []
    
    df = pd.read_csv(non_infected_file)
    
    # class列(列7)が0の企業のみ抽出
    non_infected_companies = []
    for _, row in df.iterrows():
        stock_code = str(row.iloc[3])  # 列3: stock_code
        class_value = int(row.iloc[7])  # 列7: class
        
        if class_value == 0:
            if stock_code not in non_infected_companies:
                non_infected_companies.append(stock_code)
    
    return non_infected_companies

def get_available_companies(prediction_dir, non_infected_companies):
    """
    prediction/outputに存在するNon-infected企業を確認
    """
    available = []
    
    for code in non_infected_companies:
        # Partialデータの確認
        partial_act = os.path.join(prediction_dir, f'{code}_partial_window_act.csv')
        partial_pred = os.path.join(prediction_dir, f'{code}_partial_window_pred.csv')
        
        if os.path.exists(partial_act) and os.path.exists(partial_pred):
            files = [partial_act, partial_pred]
            available.append({
                'code': code,
                'files': files
            })
    
    return available

def copy_random_selection(available, dst_dir, count=10):
    """
    ランダムに選択してコピー
    """
    if len(available) < count:
        count = len(available)
        print(f"⚠️ 利用可能な企業数が少ないため、{count}社を使用")
    
    # ランダム選択
    selected = random.sample(available, count)
    
    # コピー
    copied = 0
    for company in selected:
        code = company['code']
        print(f"\n📂 {code} (Non-infected確認済み)")
        
        for src in company['files']:
            filename = os.path.basename(src)
            dst = os.path.join(dst_dir, filename)
            shutil.copy(src, dst)
            print(f"   ✅ {filename}")
            copied += 1
    
    return copied, [c['code'] for c in selected]

def main():
    parser = argparse.ArgumentParser(description='Non-infected企業データ収集')
    parser.add_argument('--count', type=int, default=10,
                       help='コピーする企業数 (デフォルト: 10)')
    args = parser.parse_args()
    
    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))  # jp/
    
    litigation_dir = os.path.join(base_dir, 'data_source', 'data')
    prediction_dir = os.path.join(base_dir, 'prediction', 'output')
    dst_dir = os.path.join(script_dir, 'data')
    
    print("\n" + "="*60)
    print("🔍 Non-infected企業データ収集 (正常パターン用)")
    print("="*60)
    
    print(f"\n設定:")
    print(f"  - コピー企業数: {args.count}")
    print(f"  - データ種別: Partialのみ")
    
    # 既存データをクリア
    os.makedirs(dst_dir, exist_ok=True)
    for f in glob.glob(os.path.join(dst_dir, '*.csv')):
        os.remove(f)
    print(f"\n📂 出力先: {dst_dir} (クリア済み)")
    
    # Non-infected企業を取得
    print("\n🔍 Non-infected企業を確認中...")
    non_infected_companies = get_verified_noninfected_companies(litigation_dir)
    print(f"   確認済みNon-infected企業: {len(non_infected_companies)}社")
    
    # 利用可能な企業を確認
    print("\n🔍 prediction/outputを確認中...")
    available = get_available_companies(prediction_dir, non_infected_companies)
    print(f"   利用可能なNon-infected企業: {len(available)}社")
    
    if len(available) == 0:
        print("\n❌ 利用可能なデータがありません!")
        return
    
    # ランダムに選択してコピー
    print(f"\n📋 ランダムに{args.count}社を選択してコピー...")
    copied, selected_codes = copy_random_selection(available, dst_dir, args.count)
    
    # サマリー
    print("\n" + "="*60)
    print("🎉 データ収集完了!")
    print("="*60)
    print(f"\n📊 結果:")
    print(f"   コピーした企業: {len(selected_codes)}社")
    print(f"   コピーしたファイル: {copied}件")
    print(f"   選択された企業コード: {', '.join(selected_codes)}")
    
    print(f"\n次のステップ:")
    print(f"   python create_pattern.py")
    print()

if __name__ == '__main__':
    main()
