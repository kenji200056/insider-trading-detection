# -*- coding: utf-8 -*-
"""
Infected企業データ収集スクリプト

このスクリプトは、確実にInfected(class=1)と判定された企業の
Partialデータ(インサイダー取引期間前後)のみをランダムに選択してコピーします。

使用方法:
    cd detection/input
    python collect_infected_data.py

オプション:
    --count N   : コピーする企業数 (デフォルト: 10)
    --include_full : Fullデータも含める
"""

import os
import glob
import shutil
import random
import argparse
import pandas as pd

def get_verified_infected_companies(litigation_dir):
    """
    litigation/dataからclass=1のInfected企業を取得
    """
    infected_file = os.path.join(litigation_dir, 'infected', 'infected.csv')
    
    if not os.path.exists(infected_file):
        print(f"❌ ファイルが見つかりません: {infected_file}")
        return []
    
    df = pd.read_csv(infected_file)
    
    # class列(列7)が1の企業のみ抽出
    infected_companies = []
    for _, row in df.iterrows():
        stock_code = str(row.iloc[3])  # 列3: stock_code
        class_value = int(row.iloc[7])  # 列7: class
        
        if class_value == 1:
            if stock_code not in infected_companies:
                infected_companies.append(stock_code)
    
    return infected_companies

def get_available_companies(prediction_dir, infected_companies, include_full=False):
    """
    prediction/outputに存在するInfected企業を確認
    """
    available = []
    
    for code in infected_companies:
        # Partialデータの確認
        partial_act = os.path.join(prediction_dir, f'{code}_partial_window_act.csv')
        partial_pred = os.path.join(prediction_dir, f'{code}_partial_window_pred.csv')
        
        if os.path.exists(partial_act) and os.path.exists(partial_pred):
            files = [partial_act, partial_pred]
            
            # Fullデータも含める場合
            if include_full:
                full_act = os.path.join(prediction_dir, f'{code}_full_window_act.csv')
                full_pred = os.path.join(prediction_dir, f'{code}_full_window_pred.csv')
                if os.path.exists(full_act) and os.path.exists(full_pred):
                    files.extend([full_act, full_pred])
            
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
        print(f"\n📂 {code} (Infected確認済み)")
        
        for src in company['files']:
            filename = os.path.basename(src)
            dst = os.path.join(dst_dir, filename)
            shutil.copy(src, dst)
            print(f"   ✅ {filename}")
            copied += 1
    
    return copied, [c['code'] for c in selected]

def main():
    parser = argparse.ArgumentParser(description='Infected企業データ収集')
    parser.add_argument('--count', type=int, default=10,
                       help='コピーする企業数 (デフォルト: 10)')
    parser.add_argument('--include_full', action='store_true',
                       help='Fullデータも含める')
    args = parser.parse_args()
    
    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))  # jp/
    
    litigation_dir = os.path.join(base_dir, 'data_source', 'data')
    prediction_dir = os.path.join(base_dir, 'prediction', 'output')
    dst_dir = os.path.join(script_dir, 'data')
    
    print("\n" + "="*60)
    print("🔍 Infected企業データ収集")
    print("="*60)
    
    print(f"\n設定:")
    print(f"  - コピー企業数: {args.count}")
    print(f"  - データ種別: {'Partial + Full' if args.include_full else 'Partialのみ'}")
    
    # 既存データをクリア
    os.makedirs(dst_dir, exist_ok=True)
    for f in glob.glob(os.path.join(dst_dir, '*.csv')):
        os.remove(f)
    print(f"\n📂 出力先: {dst_dir} (クリア済み)")
    
    # Infected企業を取得
    print("\n🔍 Infected企業を確認中...")
    infected_companies = get_verified_infected_companies(litigation_dir)
    print(f"   確認済みInfected企業: {len(infected_companies)}社")
    
    # 利用可能な企業を確認
    print("\n🔍 prediction/outputを確認中...")
    available = get_available_companies(prediction_dir, infected_companies, args.include_full)
    print(f"   利用可能なInfected企業: {len(available)}社")
    
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
