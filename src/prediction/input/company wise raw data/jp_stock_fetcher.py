#!/usr/bin/env python3
"""
jp_stock_fetcher.py
- infected_data.csv と non-infected_data.csv を読み込む
- stock_code を使用して Yahoo Finance (yfinance) から株価データを取得
- period を中心に前後3年分のデータで [symbol]_full.csv を作成
- period 期間（前後マージン含む）で [symbol]_partial.csv を作成
- 列名なし、出来高（Volume）のみの1列形式で出力
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import re

def parse_period(period_str):
    """
    period 文字列から開始日と終了日を算出
    例: "2015/06/01-2015/06/30" -> (2015-06-01, 2015-06-30)
    例: "2015/06/01" -> (2015-06-01, 2015-06-01)
    """
    if pd.isna(period_str) or str(period_str).strip() == "":
        return None, None
    
    s = str(period_str).strip()
    try:
        if '-' in s:
            start_s, end_s = s.split('-')
            start_date = datetime.strptime(start_s.strip(), '%Y/%m/%d')
            end_date = datetime.strptime(end_s.strip(), '%Y/%m/%d')
            return start_date, end_date
        else:
            dt = datetime.strptime(s, '%Y/%m/%d')
            return dt, dt
    except Exception as e:
        print(f"  日付パースエラー ({s}): {e}")
        return None, None

def fetch_and_save(stock_code, period_str, output_dir, symbol_prefix=""):
    """
    株価データを取得して保存する
    """
    # 日本株のティッカー形式に変換 (例: 7203 -> 7203.T)
    ticker_symbol = f"{str(stock_code).strip()}.T"
    
    start_dt, end_dt = parse_period(period_str)
    if not start_dt:
        return False

    print(f"  株価取得中: {ticker_symbol} (Period: {period_str})")

    # Fullデータ用（前後3年）
    full_start = (start_dt - timedelta(days=365*3)).strftime('%Y-%m-%d')
    full_end = (end_dt + timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    # Partialデータ用（事件期間の前100日、後100日のマージンを持たせる）
    partial_start = (start_dt - timedelta(days=100)).strftime('%Y-%m-%d')
    partial_end = (end_dt + timedelta(days=100)).strftime('%Y-%m-%d')

    try:
        # yfinanceを使用してダウンロード
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=full_start, end=full_end)
        
        if df.empty:
            print(f"    警告: データが空です ({ticker_symbol})")
            return False

        # 出来高（Volume）のみを抽出
        volume_data = df['Volume']

        # --- Full CSV の作成 ---
        full_path = os.path.join(output_dir, f"{stock_code}_full.csv")
        volume_data.to_csv(full_path, index=False, header=False)

        # --- Partial CSV の作成 ---
        # 日付インデックスを使って事件期間付近をスライス
        mask = (df.index >= partial_start) & (df.index <= partial_end)
        partial_data = volume_data.loc[mask]
        
        if not partial_data.empty:
            partial_path = os.path.join(output_dir, f"{stock_code}_partial.csv")
            partial_data.to_csv(partial_path, index=False, header=False)
            print(f"    保存完了: {stock_code}_full.csv / partial.csv")
            return True
        else:
            print(f"    警告: Partial用の期間データが不足しています")
            return False

    except Exception as e:
        print(f"    取得エラー: {e}")
        return False

def main():
    # フォルダパスの設定（相対パス）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))  # jp/
    data_source_dir = os.path.join(base_dir, "data_source/data")
    output_dir = script_dir  # 現在のスクリプトと同じディレクトリに出力
    
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {
            "path": os.path.join(data_source_dir, "infected/infected.csv"),
            "prefix": "inf_"
        },
        {
            "path": os.path.join(data_source_dir, "non_infected/non-infected.csv"),
            "prefix": "noninf_"
        }
    ]

    for config in configs:
        file_path = config["path"]
        prefix = config["prefix"]
        
        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue

        print(f"\n--- {os.path.basename(file_path)} の処理開始 ---")
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        if 'stock_code' not in df.columns or 'period' not in df.columns:
            print(f"エラー: カラム 'stock_code' または 'period' が見つかりません。")
            continue

        success_count = 0
        for _, row in df.iterrows():
            code = row['stock_code']
            period = row['period']
            
            if pd.isna(code) or str(code).strip() == "" or code == "nan":
                continue
                
            if fetch_and_save(code, period, output_dir, prefix):
                success_count += 1

        print(f"--- 完了: {success_count} 件のデータを取得・生成しました ---")

if __name__ == "__main__":
    main()
