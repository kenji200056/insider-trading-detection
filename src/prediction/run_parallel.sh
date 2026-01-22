#!/bin/bash

# スクリプトのあるディレクトリに移動
cd "$(dirname "$0")"

# 処理対象のファイルリストを取得
FILES=(input/*.csv)
NUM_FILES=${#FILES[@]}
NUM_TERMINALS=4

# ターミナルあたりのファイル数を計算 (切り上げ)
FILES_PER_TERMINAL=$(( (NUM_FILES + NUM_TERMINALS - 1) / NUM_TERMINALS ))

echo "合計ファイル数: $NUM_FILES"
echo "ターミナル数: $NUM_TERMINALS"
echo "ターミナルあたりのファイル数: $FILES_PER_TERMINAL"
echo "-----------------------------------------------"

# ログディレクトリを作成
mkdir -p logs

# ファイルリストを分割して、各ターミナル（プロセス）で実行
for (( i=0; i<$NUM_TERMINALS; i++ )); do
    # 各プロセスが処理するファイルの範囲を計算
    start=$(( i * FILES_PER_TERMINAL ))
    
    # 配列のスライスで対象ファイルを取得
    # bash 3.x (macOSデフォルト) でも動くように工夫
    chunk_files=()
    for (( j=0; j<FILES_PER_TERMINAL; j++ )); do
        index=$(( start + j ))
        if [ $index -lt $NUM_FILES ]; then
            chunk_files+=("${FILES[$index]}")
        fi
    done

    # 処理するファイルがなければループを抜ける
    if [ ${#chunk_files[@]} -eq 0 ]; then
        continue
    fi
    
    echo "プロセス $i: ${#chunk_files[@]} 個のファイルを処理します -> logs/process_$i.log"
    
    # nohupでバックグラウンド実行し、ログをファイルに出力
    # `python`の代わりに`python3`が必要な環境もある
    nohup python run.py "${chunk_files[@]}" > "logs/process_$i.log" 2>&1 &
done

echo "-----------------------------------------------"
echo "4つのプロセスをバックグラウンドで起動しました。"
echo "進捗は logs/process_*.log で確認できます。"
echo "すべての処理が完了するのを待っています..."

# バックグラウンドジョブの完了を待つ
wait

echo "-----------------------------------------------"
echo "🎉 すべてのプロセスが完了しました。"
