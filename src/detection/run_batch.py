# -*- coding: utf-8 -*-
"""
作成日: 2025年12月
目的: 全企業の異常検出をバッチ処理で実行

このスクリプトは、prediction/outputディレクトリ内の全ての企業データに対して
異常検出を自動実行します。
"""

import os
import glob
from detect_anomaly import detect_anomaly

def run_batch_detection(prediction_output_dir='../prediction/output', 
                       pattern_dir='input',
                       output_dir='output',
                       save_figures=False):
    """
    全企業の異常検出をバッチ処理で実行
    
    引数:
        prediction_output_dir: str - prediction/outputディレクトリへのパス
        pattern_dir: str - 異常パターンファイルのディレクトリ
        output_dir: str - 出力ディレクトリ
        save_figures: bool - 可視化図を保存するかどうか
    """
    # スクリプトのあるディレクトリにカレントディレクトリを移動
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # prediction/outputディレクトリ内の全CSVファイルを取得
    csv_files = glob.glob(os.path.join(prediction_output_dir, '*_window_act.csv'))
    
    # 企業とデータセットの組み合わせを抽出
    companies_datasets = set()
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        # 例: 1301_full_window_act.csv -> 1301_full
        parts = basename.replace('_window_act.csv', '').split('_')
        if len(parts) >= 2:
            company = parts[0]
            dataset = parts[1]
            companies_datasets.add((company, dataset))
    
    total = len(companies_datasets)
    print(f"\n{'='*60}")
    print(f"検出された企業・データセット数: {total}")
    print(f"{'='*60}\n")
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, (company, dataset) in enumerate(sorted(companies_datasets), 1):
        # 出力ファイルが既に存在するかチェック
        output_file = os.path.join(output_dir, f'{company}_output.csv')
        
        if os.path.exists(output_file):
            print(f"[{idx}/{total}] ✅ スキップ（処理済み）: {company} ({dataset})")
            skipped += 1
            continue
        
        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] 🔄 処理中: {company} ({dataset})")
        print(f"{'='*60}")
        
        try:
            # 異常検出を実行
            result = detect_anomaly(
                company=company,
                dataset=dataset,
                prediction_output_dir=prediction_output_dir,
                pattern_dir=pattern_dir,
                output_dir=output_dir,
                save_figures=save_figures
            )
            
            processed += 1
            print(f"✅ 完了: {company} ({dataset}) - {len(result)}件の異常を検出")
            
        except FileNotFoundError as e:
            print(f"⚠️  エラー: ファイルが見つかりません - {e}")
            errors += 1
        except Exception as e:
            print(f"❌ エラー: {company} ({dataset}) - {e}")
            errors += 1
    
    # 最終サマリー
    print(f"\n{'='*60}")
    print(f"🎉 バッチ処理完了!")
    print(f"📊 処理済み: {processed}/{total}")
    print(f"⏭️  スキップ: {skipped}/{total}")
    print(f"❌ エラー: {errors}/{total}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='全企業の異常検出をバッチ処理で実行')
    parser.add_argument('--prediction_output_dir', type=str, default='../prediction/output',
                       help='prediction/outputディレクトリパス')
    parser.add_argument('--pattern_dir', type=str, default='input',
                       help='異常パターンファイルのディレクトリパス')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='出力ディレクトリパス')
    parser.add_argument('--save_figures', action='store_true',
                       help='可視化図を保存')
    
    args = parser.parse_args()
    
    run_batch_detection(
        prediction_output_dir=args.prediction_output_dir,
        pattern_dir=args.pattern_dir,
        output_dir=args.output_dir,
        save_figures=args.save_figures
    )
