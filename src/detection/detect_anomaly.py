# -*- coding: utf-8 -*-
"""
著者: Sheikh Rabiul Islam（Pythonへ変換）
作成日: 2017年11月20日（オリジナル）、2025年変換
目的: 離散信号（時系列）類似度測定による異常検出

このスクリプトは、株価時系列データにおける異常パターンを検出します。
正規化相互相関（Normalized Cross-Correlation）を使用して、
予測データと実データ、または既知の異常パターンとの類似度を計算します。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import pandas as pd
import os

# グローバル変数
window_size = 50      # ウィンドウサイズ
overlap = 10          # オーバーラップ量
num_method = 3        # 予測手法の数
num_pattern = 11      # 異常パターンの数

# 結果格納用
result = []


def load_csv(filepath):
    """
    CSVファイルをNumPy配列として読み込む（MATLABのcsvreadと同等）
    
    引数:
        filepath: CSVファイルのパス
    
    戻り値:
        numpy.ndarray: 読み込んだデータ
    """
    try:
        # まず単純な読み込みを試みる
        return np.loadtxt(filepath, delimiter=',')
    except ValueError:
        # 行ごとに列数が異なるファイル（pattern.csvなど）を処理
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 最大列数を求める
        max_cols = 0
        rows = []
        for line in lines:
            line = line.strip()
            if line:
                values = line.split(',')
                values = [float(v) for v in values if v]
                rows.append(values)
                max_cols = max(max_cols, len(values))
        
        # パディングされた配列を作成（MATLABと同様に0でパディング）
        result = np.zeros((len(rows), max_cols))
        for i, row in enumerate(rows):
            result[i, :len(row)] = row
        
        return result


def load_pattern_sizes(filepath):
    """
    パターンサイズを1次元配列として読み込む
    
    引数:
        filepath: pattern_sizes.csvファイルのパス
    
    戻り値:
        numpy.ndarray: 各パターンのサイズ
    """
    with open(filepath, 'r') as f:
        line = f.read().strip()
    return np.array([int(x) for x in line.split(',') if x])


def normalized_cross_correlation(method, window, pattern, day,
                                  actual_data, predicted_data_list,
                                  pattern_mat, pattern_desc_arr,
                                  window_size, overlap, save_figures=False, output_dir='output'):
    """
    信号間の正規化相互相関を計算する
    
    正規化相互相関（NCC）は、2つの信号の類似度を-1から1の範囲で測定します。
    1に近いほど強い正の相関（類似）、-1に近いほど強い負の相関を示します。
    
    引数:
        method: int (1, 2, または 3 - 異なる予測手法を表す)
            1: ウィンドウベース予測
            2: ポイントベース予測（1日先予測）
            3: 履歴ベース予測（全シーケンス予測）
        window: int (ウィンドウインデックス、1始まり)
        pattern: int (パターンインデックス、0は実vs予測、1以上はパターンマッチング)
        day: int (ウィンドウ内の日インデックス、1始まり)
        actual_data: 実際のデータのNumPy配列
        predicted_data_list: [ウィンドウベース, 日ベース, 履歴ベース]予測配列のリスト
        pattern_mat: 異常パターンのNumPy配列
        pattern_desc_arr: パターンサイズのNumPy配列
        window_size: int
        overlap: int
        save_figures: bool - 可視化図を保存するかどうか
        output_dir: str - 出力ディレクトリ
    
    戻り値:
        result_row: タプル (method, window, pattern, day) または None
                    相関係数が0.80を超えた場合に返す
    """
    predicted_data = predicted_data_list[method - 1]
    
    # グラフのタイトルを構築
    method_names = ['ウィンドウベース', 'ポイントベース', '履歴ベース']
    graph_title1 = f'実測値 vs 予測値 ({method_names[method-1]}).(w={window},p={pattern},d={day})'
    graph_title2 = f'日ラグ vs 正規化相互相関.(w={window},p={pattern},d={day})'
    
    if pattern == 0:
        # 実測値 vs 予測値のチェック（異常パターンとの比較なし）
        signal1_start = (window - 1) * window_size  # 0始まりインデックス
        signal1_end = signal1_start + window_size
        if len(predicted_data) < signal1_end:
            signal1_end = len(predicted_data)
        signal2_start = signal1_start
        signal2_end = signal1_end
        
        signal1 = predicted_data[signal1_start:signal1_end]
        signal2 = actual_data[signal2_start:signal2_end]
    else:  # pattern > 0
        # 異常パターンと予測/実測データの比較
        pattern_size = int(pattern_desc_arr[pattern - 1])  # 0始まりインデックス
        signal1_start = (window - 1) * window_size + (day - 1)  # 0始まりインデックス
        signal1_end = signal1_start + pattern_size
        if len(predicted_data) < signal1_end:
            signal1_end = len(predicted_data)
        
        signal1 = predicted_data[signal1_start:signal1_end]
        signal2 = pattern_mat[pattern - 1, :pattern_size]  # 0始まりインデックス
    
    if len(signal1) == len(signal2) and len(signal1) > 0:
        # scipyを使用した正規化相互相関
        # MATLABのxcorr(..., 'coeff')は自己相関で正規化する
        # これをscipy.signal.correlateで再現
        
        # 相関係数計算のため信号を正規化
        signal1_norm = signal1 - np.mean(signal1)
        signal2_norm = signal2 - np.mean(signal2)
        
        # 相関を計算
        cor_sequence = correlate(signal1_norm, signal2_norm, mode='full')
        
        # 相関係数（-1から1）を得るため正規化
        norm_factor = np.sqrt(np.sum(signal1_norm**2) * np.sum(signal2_norm**2))
        if norm_factor > 0:
            cor_sequence = cor_sequence / norm_factor
        
        # ラグ配列を作成（MATLABのxcorrのlag出力と同等）
        n = len(signal1)
        lag = np.arange(-(n-1), n)
        
        max_cor = np.max(cor_sequence)
        
        if max_cor > 0.80:
            # 閾値0.80（80%以上の類似度）で異常と判定
            max_index = np.argmax(cor_sequence)
            
            if save_figures:
                # 図1: 実測値 vs 予測値
                x1 = np.arange(1, len(signal1) + 1)
                plt.figure()
                plt.plot(x1, signal2, label='実測値')
                plt.plot(x1, signal1, label='予測値')
                plt.legend()
                plt.xlabel('日')
                plt.ylabel('出来高')
                plt.title(graph_title1)
                plt.savefig(os.path.join(output_dir, f'fig_m{method}_w{window}_p{pattern}_d{day}_signals.png'))
                plt.close()
                
                # 図2: 日ラグ vs 正規化相互相関
                plt.figure()
                plt.plot(lag, cor_sequence)
                plt.xlabel(f'日ラグ\n最大NCR = {max_cor:.4f} 日ラグ = {lag[max_index]}')
                plt.ylabel('正規化相互相関')
                plt.title(graph_title2)
                plt.savefig(os.path.join(output_dir, f'fig_m{method}_w{window}_p{pattern}_d{day}_correlation.png'))
                plt.close()
            
            return (method, window, pattern, day)
    
    return None


def detect_anomaly(company='bp', dataset='full', prediction_output_dir='../prediction/output', 
                   pattern_dir='input', output_dir='output', save_figures=False):
    """
    時系列データにおける異常を検出するメイン関数
    
    この関数は3つの予測手法それぞれについて、スライディングウィンドウ方式で
    異常パターンとの類似度を計算し、閾値を超えるケースを異常として検出します。
    
    引数:
        company: str - 企業ティッカー（例: 'bp', 'amsc', 'wfc'）または日本企業コード（例: '1301', '7203'）
        dataset: str - 'full' または 'partial'
        prediction_output_dir: str - prediction/outputディレクトリへのパス
        pattern_dir: str - 異常パターンファイル(pattern.csv, pattern_sizes.csv)のディレクトリ
        output_dir: str - 出力ディレクトリへのパス
        save_figures: bool - 可視化図を保存するかどうか
    
    戻り値:
        result: 検出された異常のNumPy配列 [method, window, pattern, day]
    """
    global window_size, overlap, num_method, num_pattern
    
    # パラメータ設定
    window_size = 50    # ウィンドウサイズ（50日間）
    overlap = 10        # オーバーラップ（10日間）
    num_method = 3      # 予測手法数
    num_pattern = 11    # 異常パターン数
    
    print(f"{company} ({dataset}) のデータを読み込み中...")
    print(f"予測結果の読み込み元: {prediction_output_dir}")
    print(f"異常パターンの読み込み元: {pattern_dir}")
    
    # データ読み込み（predictionの出力から）
    actual_data = load_csv(os.path.join(prediction_output_dir, f'{company}_{dataset}_window_act.csv'))
    predicted_data_window_based = load_csv(os.path.join(prediction_output_dir, f'{company}_{dataset}_window_pred.csv'))
    predicted_data_day_based = load_csv(os.path.join(prediction_output_dir, f'{company}_{dataset}_point_pred.csv'))
    predicted_data_historical_based = load_csv(os.path.join(prediction_output_dir, f'{company}_{dataset}_sequence_pred.csv'))
    
    # 異常パターンの読み込み（detection/inputから）
    pattern_mat = load_csv(os.path.join(pattern_dir, 'pattern.csv'))
    pattern_desc_arr = load_pattern_sizes(os.path.join(pattern_dir, 'pattern_sizes.csv'))
    
    # pattern_desc_arrが1次元であることを確認
    if pattern_desc_arr.ndim == 0:
        pattern_desc_arr = np.array([pattern_desc_arr])
    pattern_desc_arr = pattern_desc_arr.flatten()
    
    # pattern_matが2次元であることを確認
    if pattern_mat.ndim == 1:
        pattern_mat = pattern_mat.reshape(1, -1)
    
    predicted_data_list = [
        predicted_data_window_based,
        predicted_data_day_based,
        predicted_data_historical_based
    ]
    
    # 各手法のウィンドウ数を計算
    methods = [
        (len(predicted_data_window_based), int(np.ceil(len(predicted_data_window_based) / window_size))),
        (len(predicted_data_day_based), int(np.ceil(len(predicted_data_day_based) / window_size))),
        (len(predicted_data_historical_based), int(np.ceil(len(predicted_data_historical_based) / window_size)))
    ]
    
    result = []
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    print("異常検出を実行中...")
    
    for method in range(1, 4):  # 1から3
        num_window = methods[method - 1][1]
        
        for window in range(1, num_window + 1):  # 1からnum_window
            # まず: 実測値 vs 予測値のチェック (pattern = 0)
            res = normalized_cross_correlation(
                method, window, 0, 0,
                actual_data, predicted_data_list,
                pattern_mat, pattern_desc_arr,
                window_size, overlap, save_figures, output_dir
            )
            if res is not None:
                result.append(res)
            
            # 次に: パターンマッチング
            for pattern in range(1, num_pattern + 1):  # 1からnum_pattern
                current_overlap = overlap
                if window == num_window:
                    current_overlap = 0
                
                pattern_size = int(pattern_desc_arr[pattern - 1]) if pattern - 1 < len(pattern_desc_arr) else window_size
                
                for day in range(1, window_size - pattern_size + current_overlap + 1):
                    res = normalized_cross_correlation(
                        method, window, pattern, day,
                        actual_data, predicted_data_list,
                        pattern_mat, pattern_desc_arr,
                        window_size, current_overlap, save_figures, output_dir
                    )
                    if res is not None:
                        result.append(res)
    
    # 結果をNumPy配列に変換
    if len(result) > 0:
        result = np.array(result)
    else:
        result = np.array([]).reshape(0, 4)
    
    # 結果をCSVに書き出し
    output_file = os.path.join(output_dir, f'{company}_output.csv')
    np.savetxt(output_file, result, delimiter=',', fmt='%d')
    
    print(f"\n結果ヘッダー: 手法, ウィンドウ, パターン, 日")
    print(f"検出された異常の総数: {len(result)}")
    print(f"結果の保存先: {output_file}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='時系列データにおける異常を検出')
    parser.add_argument('--company', type=str, default='bp', help='企業ティッカー（例: bp, amsc, wfc）または日本企業コード（例: 1301, 7203）')
    parser.add_argument('--dataset', type=str, default='full', help='データセットタイプ: full または partial')
    parser.add_argument('--prediction_output_dir', type=str, default='../prediction/output', help='prediction/outputディレクトリパス')
    parser.add_argument('--pattern_dir', type=str, default='input', help='異常パターンファイルのディレクトリパス')
    parser.add_argument('--output_dir', type=str, default='output', help='出力ディレクトリパス')
    parser.add_argument('--save_figures', action='store_true', help='可視化図を保存')
    
    args = parser.parse_args()
    
    result = detect_anomaly(
        company=args.company,
        dataset=args.dataset,
        prediction_output_dir=args.prediction_output_dir,
        pattern_dir=args.pattern_dir,
        output_dir=args.output_dir,
        save_figures=args.save_figures
    )
    
    if len(result) > 0:
        print("\n検出された異常:")
        print(result)
