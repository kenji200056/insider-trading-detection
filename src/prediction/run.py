# Jakob Aungiers によるコードテンプレート 
# 修正者: Sheikh Rabiul Islam
# 日付: 2017/11/10 
# 目的: 株式市場のボラティリティ予測（実行用プログラム）
import lstm
import time
import sys # 並列処理のために追加
import matplotlib
import japanize_matplotlib # 日本語表示用
if '--visual' not in sys.argv:
    matplotlib.use('Agg') # 画面表示しない設定（バックグラウンド実行用）
import matplotlib.pyplot as plt
import pandas as pd
import csv
import itertools
import os
import glob

def plot_results(predicted_data, true_data, method, dataset, company):
    """
    単一予測結果のプロット（グラフ作成）と保存
    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='実際のデータ')
    plt.plot(predicted_data, label='予測値')
    plt.legend()
    
    # 予測結果をCSVに保存
    df_pr = pd.DataFrame(predicted_data)
    fname = 'output/' + company + '_' + dataset + '_' + method + '_pred.csv' 
    df_pr.to_csv(fname, index=False, header=False)
    
    # 実際のデータをCSVに保存
    # plt.show() # 自動化のため非表示
    if '--visual' in sys.argv:
        plt.show() # ビジュアルモードの場合に表示

    plt.close() # メモリ解放

def plot_results_multiple(predicted_data, true_data, prediction_len, method, dataset, company):
    """
    複数（スライディング窓）予測結果のプロットと保存
    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='実際のデータ')
    
    # 予測結果のリストをグラフ用に加工してプロット
    pr_list=[]
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='予測値')
        pr_list.append(data)
        plt.legend()
    
    # 予測結果を統合してCSV保存
    pr_list_flat = list(itertools.chain.from_iterable(pr_list))
    df_pr = pd.DataFrame(pr_list_flat)
    fname = 'output/' + company + '_' + dataset + '_' + method + '_pred.csv'
    df_pr.to_csv(fname, index=False, header=False)
    
    df_tr = pd.DataFrame(true_data)
    fname = 'output/' + company + '_' + dataset + '_' + method + '_act.csv'
    df_tr.to_csv(fname, index=False, header=False)
    
    # グラフ画像をscreensフォルダに保存
    if not os.path.exists('output/screens'):
        os.makedirs('output/screens')
    img_fname = 'output/screens/' + company + '_' + dataset + '_' + method + '.png'
    plt.savefig(img_fname)

    # plt.show() # 自動化のため非表示
    if '--visual' in sys.argv:
        plt.show() # ビジュアルモードの場合に表示

    plt.close() # メモリ解放
    

# メイン実行スレッド
if __name__=='__main__':
    # スクリプトのあるディレクトリにカレントディレクトリを移動する（どこから実行してもパスが合うようにする）
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    global_start_time = time.time()
    epochs  = 1   # エポック（学習回数）。大きくすると予測精度が向上の可能性あり。
    seq_len = 50  # 入力シークエンスの長さ（過去50日間を見る）

    # --visual フラグをチェックして、ファイルリストから除外
    if '--visual' in sys.argv:
        print("ビジュアルモードで実行します。グラフがリアルタイムで表示されます。")
        sys.argv.remove('--visual')

    # 並列処理の引数があるか確認し、処理対象のファイルリストを決定する
    all_files = sorted(glob.glob('input/*.csv'))
    input_files = all_files  # デフォルトでは全ファイルを対象とする

    try:
        # --parallel-job <index> <total> の形式で引数を探す
        p_idx = sys.argv.index('--parallel-job')
        job_index = int(sys.argv[p_idx + 1])
        total_jobs = int(sys.argv[p_idx + 2])

        num_files = len(all_files)
        # 各ジョブのファイル数を計算（切り上げ）
        files_per_job = (num_files + total_jobs - 1) // total_jobs
        start = job_index * files_per_job
        end = min(start + files_per_job, num_files) # 配列の範囲を超えないようにする
        
        input_files = all_files[start:end]
        print(f"並列ジョブ {job_index+1}/{total_jobs} として、{len(input_files)}個のファイルを処理します。")

        # 処理済みの引数をリストから削除
        del sys.argv[p_idx:p_idx+3]

    except ValueError:
        # --parallel-job が見つからなかった場合
        if len(sys.argv) > 1:
            # 他の引数がもしあれば、それをファイルリストと見なす（従来の挙動）
            input_files = sys.argv[1:]
            print(f"コマンドライン引数で指定された {len(input_files)} 個のファイルを処理します。")
        else:
            # 引数が何もなければ全ファイルを対象
            print(f"単一プロセスで全 {len(input_files)} 個のファイルを処理します。")
    except IndexError:
        print("エラー: --parallel-job 引数の後に <ジョブ番号> と <総ジョブ数> が必要です。")
        exit()
    
    # 全手法のリスト
    methods = ['window', 'sequence', 'point']

    print(f"\n{'='*60}")
    print(f"検出された入力ファイル数: {len(input_files)}")
    print(f"処理する手法: {', '.join(methods)}")
    
    # outputディレクトリが存在しない場合は作成
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 実際に処理が必要なファイル数を事前にカウント
    valid_files = []
    for input_file in input_files:
        basename = os.path.basename(input_file)
        name_parts = basename.split('_')
        
        # ファイル名形式チェック
        if len(name_parts) < 2:
            continue
        
        company = name_parts[0]
        dataset = name_parts[1].replace('.csv', '')
        
        # dataset種別チェック
        if dataset not in ['full', 'partial']:
            continue
        
        # 全手法が処理済みかチェック
        all_done = True
        for method in methods:
            pred_file = f'output/{company}_{dataset}_{method}_pred.csv'
            if not os.path.exists(pred_file):
                all_done = False
                break
        
        if not all_done:
            valid_files.append(input_file)
    
    total_files = len(valid_files)
    total_methods_to_process = total_files * len(methods)
    
    print(f"処理が必要なファイル数: {total_files}")
    print(f"処理が必要な手法数: {total_methods_to_process}")
    print(f"{'='*60}\n")
    
    # 進捗追跡用の変数
    processed_files = 0
    total_methods_done = 0
    method_start_times = []  # 各手法の処理時間を記録

    for file_idx, input_file in enumerate(valid_files, 1):
        # ファイル名から company と dataset を抽出
        # input/1301_full.csv -> basename: 1301_full.csv -> split: ['1301', 'full.csv']
        basename = os.path.basename(input_file)
        name_parts = basename.split('_')
        
        # ファイル名が予期した形式（例: 1301_full.csv）かチェック
        if len(name_parts) < 2:
            print(f"スキップ（ファイル名形式対象外）: {basename}")
            continue

        company = name_parts[0]
        dataset = name_parts[1].replace('.csv', '')

        # dataset が full か partial でない場合はスキップ（念のため）
        if dataset not in ['full', 'partial']:
            print(f"スキップ（データセット種別対象外）: {dataset}")
            continue

        print(f"\n{'='*60}")
        print(f"📁 ファイル進捗: [{file_idx}/{total_files}] ({file_idx/total_files*100:.1f}%)")
        print(f"📊 処理中: 企業={company}, データセット={dataset}")
        print(f"{'='*60}")

        # この組み合わせに必要な出力ファイルが全て揃っているかチェック
        # 全ての手法について出力済みなら、データ読み込み自体をスキップして効率化
        all_done = True
        for method in methods:
            pred_file = f'output/{company}_{dataset}_{method}_pred.csv'
            if not os.path.exists(pred_file):
                all_done = False
                break
        
        if all_done:
            print(f"✅ スキップ（全手法処理済み）: 企業={company}, データセット={dataset}")
            processed_files += 1
            total_methods_done += len(methods)
            continue

        # データを読み込む（このファイルに対する全手法で共通）
        print('> データを読み込み中... ')
        # lstm.load_dataはエラー時に空を返す可能性があるためチェックが必要かも
        try:
            X_train, y_train, X_test, y_test = lstm.load_data(input_file, seq_len, True)
            if len(X_train) == 0:
                print(f"警告: データが空、または読み込みに失敗しました: {input_file}")
                continue
        except Exception as e:
            print(f"エラー: データ読み込み中にエラーが発生しました: {e}")
            continue
            
        print('> データ読み込み完了。')
        
        # このファイルで処理する手法の数をカウント
        methods_to_process = [m for m in methods if not os.path.exists(f'output/{company}_{dataset}_{m}_pred.csv')]
        methods_done_for_this_file = 0

        for method_idx, method in enumerate(methods, 1):
            # 出力ファイルが既に存在するかチェック
            pred_file = f'output/{company}_{dataset}_{method}_pred.csv'
            
            if os.path.exists(pred_file):
                print(f"✅ スキップ（処理済み）: {method} ({pred_file} が存在します)")
                total_methods_done += 1
                methods_done_for_this_file += 1
                continue
            
            method_start = time.time()
            
            print(f"\n{'─'*60}")
            print(f"🔄 実行中: {method}")
            print(f"📈 全体進捗: {total_methods_done}/{total_methods_to_process} 手法完了")
            
            # 残り時間の推定
            if method_start_times:
                avg_time_per_method = sum(method_start_times) / len(method_start_times)
                remaining_methods = total_methods_to_process - total_methods_done
                estimated_remaining = avg_time_per_method * remaining_methods
                hours = int(estimated_remaining // 3600)
                minutes = int((estimated_remaining % 3600) // 60)
                seconds = int(estimated_remaining % 60)
                print(f"⏱️  推定残り時間: {hours}時間 {minutes}分 {seconds}秒")
            print(f"{'─'*60}")
            
            # モデル構築とコンパイル
            # ※ 各手法・各ループごとにモデルをリセットして再学習させる（元のロジックに従う）
            print('> モデル構築・コンパイル中...')
            model = lstm.build_model([1, 50, 100, 1])

            # 学習開始
            print('> 学習開始...')
            model.fit(
                X_train,
                y_train,
                batch_size=512,
                epochs=epochs,
                validation_split=0.05,
                verbose=1) # 進捗を表示
            
            fig_name = company + '_' + dataset + '_' + method	
            print(f"\n📊 現在処理中の設定: {fig_name}")	

            if method == 'window':
                # 50日分の予測を繰り返し行う
                predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
                plot_results_multiple(predictions, y_test, 50, method, dataset, company)

            if method == 'sequence':
                # 全期間を一気に予測する
                predictions = lstm.predict_sequence_full(model, X_test, seq_len)
                plot_results(predictions, y_test, method, dataset, company)

            if method == 'point':
                # 1日ずつ予測する
                predictions = lstm.predict_point_by_point(model, X_test) 
                plot_results(predictions, y_test, method, dataset, company)
            
            # この手法の処理時間を記録
            method_elapsed = time.time() - method_start
            method_start_times.append(method_elapsed)
            # 最新10件の平均を使用（より正確な推定のため）
            if len(method_start_times) > 10:
                method_start_times.pop(0)
            
            total_methods_done += 1
            methods_done_for_this_file += 1
            
            print(f"\n✅ 完了: {method} (処理時間: {method_elapsed:.1f}秒)")
            print(f"累計処理時間: {time.time() - global_start_time:.1f}秒")
            
            # Kerasのセッションをクリアしてメモリリークを防ぐ（念のため）
            from keras import backend as K
            K.clear_session()
        
        # このファイルの処理完了
        processed_files += 1
        print(f"\n✅ ファイル完了: {company}_{dataset} ({methods_done_for_this_file}/{len(methods)} 手法処理)")

    total_time = time.time() - global_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"🎉 全処理完了!")
    print(f"📁 処理ファイル数: {processed_files}/{total_files}")
    print(f"📈 処理手法数: {total_methods_done}/{total_methods_to_process}")
    print(f"⏱️  総処理時間: {hours}時間 {minutes}分 {seconds}秒")
    print(f"{'='*60}\n")
