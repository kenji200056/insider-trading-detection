#!/usr/bin/env python3
"""
通常エポックと50エポックの予測精度を比較するスクリプト
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import glob

def calculate_metrics(pred_file, act_file):
    """予測ファイルと実際値ファイルを比較してメトリクスを計算"""
    try:
        pred_df = pd.read_csv(pred_file, header=None)
        act_df = pd.read_csv(act_file, header=None)
        
        # サイズが合わない場合は小さい方に合わせる
        min_len = min(len(pred_df), len(act_df))
        if min_len == 0:
            return None
            
        pred_values = pred_df.iloc[:min_len, 0].values
        act_values = act_df.iloc[:min_len, 0].values
        
        # NaNを除外
        mask = ~(np.isnan(pred_values) | np.isnan(act_values))
        if mask.sum() == 0:
            return None
            
        pred_values = pred_values[mask]
        act_values = act_values[mask]
        
        mse = mean_squared_error(act_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(act_values, pred_values)
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae}
    except Exception as e:
        return None

def compare_folders(output_dir, output50_dir):
    """2つの出力フォルダを比較"""
    
    # 両方に存在するwindow_pred.csvファイルを見つける
    results = {'default': [], 'epoch50': []}
    
    # デフォルト出力のwindow_predファイル
    default_files = glob.glob(os.path.join(output_dir, '*_window_pred.csv'))
    
    for pred_file in default_files:
        basename = os.path.basename(pred_file)
        # 対応するactファイル
        act_file = pred_file.replace('_window_pred.csv', '_window_act.csv')
        
        if os.path.exists(act_file):
            metrics = calculate_metrics(pred_file, act_file)
            if metrics:
                results['default'].append(metrics)
    
    # 50エポック出力のwindow_predファイル
    epoch50_files = glob.glob(os.path.join(output50_dir, '*_window_pred.csv'))
    
    for pred_file in epoch50_files:
        basename = os.path.basename(pred_file)
        # 対応するactファイル
        act_file = pred_file.replace('_window_pred.csv', '_window_act.csv')
        
        if os.path.exists(act_file):
            metrics = calculate_metrics(pred_file, act_file)
            if metrics:
                results['epoch50'].append(metrics)
    
    return results

def main():
    # 相対パスで設定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    output50_dir = os.path.join(base_dir, 'output50')
    
    print("=" * 60)
    print("予測精度比較: デフォルトエポック vs 50エポック")
    print("=" * 60)
    
    results = compare_folders(output_dir, output50_dir)
    
    # デフォルトの統計
    if results['default']:
        default_rmse = np.mean([r['rmse'] for r in results['default']])
        default_mae = np.mean([r['mae'] for r in results['default']])
        print(f"\n【デフォルトエポック】")
        print(f"  ファイル数: {len(results['default'])}")
        print(f"  平均RMSE: {default_rmse:.6f}")
        print(f"  平均MAE: {default_mae:.6f}")
    else:
        print("\nデフォルトエポック: データなし")
    
    # 50エポックの統計
    if results['epoch50']:
        epoch50_rmse = np.mean([r['rmse'] for r in results['epoch50']])
        epoch50_mae = np.mean([r['mae'] for r in results['epoch50']])
        print(f"\n【50エポック】")
        print(f"  ファイル数: {len(results['epoch50'])}")
        print(f"  平均RMSE: {epoch50_rmse:.6f}")
        print(f"  平均MAE: {epoch50_mae:.6f}")
    else:
        print("\n50エポック: データなし")
    
    # 比較
    if results['default'] and results['epoch50']:
        print("\n" + "=" * 60)
        print("【比較結果】")
        rmse_diff = default_rmse - epoch50_rmse
        mae_diff = default_mae - epoch50_mae
        
        rmse_pct = (rmse_diff / default_rmse) * 100 if default_rmse > 0 else 0
        mae_pct = (mae_diff / default_mae) * 100 if default_mae > 0 else 0
        
        print(f"  RMSE改善: {rmse_diff:.6f} ({rmse_pct:+.2f}%)")
        print(f"  MAE改善: {mae_diff:.6f} ({mae_pct:+.2f}%)")
        
        if rmse_diff > 0:
            print("\n  ✓ 50エポックの方が精度が良い（RMSEが低い）")
        elif rmse_diff < 0:
            print("\n  ✗ デフォルトの方が精度が良い（RMSEが低い）")
        else:
            print("\n  = 同等の精度")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
