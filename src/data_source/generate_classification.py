# -*- coding: utf-8 -*-
"""
決定木アルゴリズムによる客観的分類システム（メインスクリプト）
data_source: 訴訟文書分類・教師データ構築モジュール
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def parse_and_create_period(date_value):
    """
    日付文字列からperiodを算出する
    - 単一日付 → そのまま
    - カンマ区切り複数日付 → 最小日-最大日
    - 期間範囲が複数ある場合 → すべての日付から最小日-最大日を算出
    """
    if pd.isna(date_value) or str(date_value).strip() == "":
        return ""
    
    date_str = str(date_value).strip()
    
    # すべての日付を抽出するリスト
    all_dates = []
    
    # カンマで分割（期間範囲が複数ある場合に対応）
    parts = [p.strip() for p in date_str.split(',') if p.strip()]
    
    for part in parts:
        # ハイフンで範囲指定されている場合（例: 2010/07/21-2010/07/30）
        if '-' in part:
            range_dates = [d.strip() for d in part.split('-') if d.strip()]
            for d in range_dates:
                try:
                    if '/' in d:
                        parsed = datetime.strptime(d, '%Y/%m/%d')
                        all_dates.append((parsed, d))
                except:
                    continue
        else:
            # 単一日付
            try:
                if '/' in part:
                    parsed = datetime.strptime(part, '%Y/%m/%d')
                    all_dates.append((parsed, part))
            except:
                continue
    
    # 日付が1つもパースできなかった場合
    if not all_dates:
        return date_str
    
    # 日付が1つだけの場合
    if len(all_dates) == 1:
        return all_dates[0][1]
    
    # 複数の日付がある場合、最小日-最大日を返す
    all_dates.sort(key=lambda x: x[0])
    min_date = all_dates[0][1]
    max_date = all_dates[-1][1]
    
    # 最小日と最大日が同じ場合は単一日付として返す
    if min_date == max_date:
        return min_date
    
    return f"{min_date}-{max_date}"

print("=" * 80)
print("訴訟案件の自動分類システム（決定木アルゴリズム）")
print("=" * 80)

# ==================== 設定 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
result_dir = os.path.join(current_dir, "result")
input_file = os.path.join(data_dir, "complete_dataset_jp.csv")
feature_vector_file = os.path.join(result_dir, "feature_vector_unlabeled.csv")
feature_map_file = os.path.join(result_dir, "feature_vector_index_map.csv")
output_dir_infected = os.path.join(data_dir, "infected")
output_dir_non_infected = os.path.join(data_dir, "non_infected")

# ディレクトリ作成
os.makedirs(output_dir_infected, exist_ok=True)
os.makedirs(output_dir_non_infected, exist_ok=True)

# ==================== ステップ1: データ読み込み ====================
print(f"\n【ステップ1】データ読み込み")
print(f"  入力ファイル: {input_file}")

if not os.path.exists(input_file):
    print(f"  ❌ エラー: {input_file} が見つかりません")
    sys.exit(1)

df = pd.read_csv(input_file, encoding='utf-8-sig')
print(f"  ✅ 読み込み完了: {len(df)}件")

# ==================== ステップ2: 特徴量ベクトル読み込み ====================
print(f"\n【ステップ2】特徴量ベクトル読み込み")
print(f"  特徴量ファイル: {feature_vector_file}")

if not os.path.exists(feature_vector_file):
    print(f"  ❌ エラー: {feature_vector_file} が見つかりません")
    print(f"  ヒント: test_withoutLabel/preprocess.py を実行してください")
    sys.exit(1)

X = pd.read_csv(feature_vector_file, index_col=0).values
features_df = pd.read_csv(feature_map_file)
features = features_df['feature'].tolist()

print(f"  ✅ 特徴量数: {X.shape[1]}")

# ==================== ステップ3: 初期ラベル作成 ====================
print(f"\n【ステップ3】決定木訓練用の初期ラベル作成")

# 既存の infected.csv と non-infected.csv からラベルを取得
infected_path = os.path.join(output_dir_infected, "infected.csv")
non_infected_path = os.path.join(output_dir_non_infected, "non-infected.csv")

if os.path.exists(infected_path) and os.path.exists(non_infected_path):
    print(f"  既存のラベル付きデータから初期ラベルを作成")
    infected_df = pd.read_csv(infected_path, encoding='utf-8-sig')
    non_infected_df = pd.read_csv(non_infected_path, encoding='utf-8-sig')

    # 既存データを結合
    labeled_df = pd.concat([infected_df, non_infected_df], ignore_index=True)

    # complete_dataset_jp.csv と照合してラベルを割り当て
    df['class'] = 0  # デフォルトは0
    for idx, row in df.iterrows():
        # title と link で照合
        match = labeled_df[(labeled_df['title'] == row['title']) & (labeled_df['link'] == row['link'])]
        if not match.empty:
            df.loc[idx, 'class'] = match.iloc[0]['class']

    y_initial = df['class'].astype(int).values
    print(f"  ✅ 既存ラベルから初期ラベル作成完了")
else:
    print(f"  既存ラベルが見つからないため、デフォルトラベルを使用")
    # フォールバック: 「内部者取引」キーワードで初期ラベル
    title_has = df['title'].str.contains('内部者取引', na=False)
    content_has = df['content'].str.contains('内部者取引', na=False)
    y_initial = (title_has | content_has).astype(int).values
    print(f"  ⚠️  フォールバック: '内部者取引'キーワードで初期ラベル作成")

print(f"  初期ラベル: class=1が{y_initial.sum()}件、class=0が{len(y_initial)-y_initial.sum()}件")

# ==================== ステップ4: 決定木で最重要特徴量を自動抽出 ====================
print(f"\n【ステップ4】決定木による最重要特徴量の自動抽出")

X_train, X_test, y_train, y_test = train_test_split(X, y_initial, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 最重要特徴量を取得
feature_importances = clf.feature_importances_
top_feature_idx = np.argmax(feature_importances)
top_feature_name = features[top_feature_idx] if pd.notna(features[top_feature_idx]) else f'feature_{top_feature_idx}'

print(f"  🎯 最重要特徴量: '{top_feature_name}'")
print(f"  📊 重要度: {feature_importances[top_feature_idx]:.4f}")
print(f"  📈 訓練精度: {clf.score(X_train, y_train):.4f}")
print(f"  📈 テスト精度: {clf.score(X_test, y_test):.4f}")

# 最重要特徴量を保存（他のスクリプトで使用するため）
top_feature_file = os.path.join(result_dir, "top_feature.txt")
with open(top_feature_file, 'w', encoding='utf-8') as f:
    f.write(top_feature_name)
print(f"  💾 最重要特徴量を保存: {top_feature_file}")

# ==================== ステップ5: 最重要特徴量で最終分類 ====================
print(f"\n【ステップ5】最重要特徴量'{top_feature_name}'で最終分類")

title_has_feature = df['title'].str.contains(top_feature_name, na=False)
content_has_feature = df['content'].str.contains(top_feature_name, na=False)
y_final = (title_has_feature | content_has_feature).astype(int).values

class_1_count = (y_final == 1).sum()
class_0_count = (y_final == 0).sum()

print(f"  分類基準: title OR content に '{top_feature_name}' を含む")
print(f"  ✅ class=1 (内部者取引あり): {class_1_count}件 ({class_1_count/len(y_final)*100:.1f}%)")
print(f"  ✅ class=0 (非内部者取引): {class_0_count}件 ({class_0_count/len(y_final)*100:.1f}%)")

# ==================== ステップ6: データ整形と保存（期間情報マージ） ====================
print(f"\n【ステップ6】データ整形と期間情報のマージ")

def add_period_data(df_main, df_type, data_dir):
    """
    stock_codeをキーとして、_data.csvから期間情報をマージする
    """
    print(f"\n  マージ処理: {df_type}")
    
    # 対応する _data.csv ファイルのパスを構築
    # non_infected のみファイル名がハイフンなので個別対応
    if df_type == 'non_infected':
        file_name = 'non-infected_data.csv'
    else:
        file_name = f"{df_type}_data.csv"
    data_file = os.path.join(data_dir, df_type, file_name)

    if not os.path.exists(data_file):
        print(f"  ⚠️  警告: {data_file} が見つかりません。period列は空になります。")
        df_main['period'] = ''
        return df_main
        
    # _data.csv を読み込み
    df_data = pd.read_csv(data_file, encoding='utf-8-sig')
    print(f"    データファイル読み込み: {os.path.basename(data_file)} ({len(df_data)}行)")
    
    # non-infected_data.csv の日本語列名 '番号' を 'no' に統一
    if '番号' in df_data.columns:
        df_data = df_data.rename(columns={'番号': 'no'})

    # stock_code と period のマッピングを作成
    # period_merger.py のロジックを参考に、重複するstock_codeに対応
    code_period_map = {}
    
    # まずは全データを1周して、各stock_codeの出現回数を数える
    code_total_counts = df_data['stock_code'].astype(str).str.strip().value_counts().to_dict()
    
    # マッピング辞書を作成
    # { "8601_1": "2005/10/04-2005/10/06", "8601_2": "...", ... }
    code_current_counts = {}
    for _, row in df_data.iterrows():
        code = str(row.get('stock_code', '')).strip()
        if code and code != 'nan':
            # このstock_codeの現在の出現回数をインクリメント
            count = code_current_counts.get(code, 0) + 1
            code_current_counts[code] = count
            
            # ユニークキーを作成
            key = f"{code}_{count}"
            
            # period を算出
            date_val = row.get('date', '')
            period = parse_and_create_period(date_val)
            code_period_map[key] = period

    # メインのDataFrameにperiodをマッピング
    periods = []
    main_code_counts = {}
    for _, row in df_main.iterrows():
        code = str(row.get('stock_code', '')).strip()
        if code and code != 'nan':
            # このstock_codeの現在の出現回数をインクリメント
            count = main_code_counts.get(code, 0) + 1
            main_code_counts[code] = count
            
            # マップからperiodを取得
            key = f"{code}_{count}"
            period = code_period_map.get(key, '')
            periods.append(period)
        else:
            periods.append('')
            
    df_main['period'] = periods
    
    filled_count = sum(1 for p in periods if p)
    print(f"    ✅ マージ完了: {len(df_main)}件中 {filled_count}件に期間を設定")
    
    return df_main

# --- ステップ6 メインロジック ---
df['class'] = y_final
infected_df = df[df['class'] == 1].copy()
non_infected_df = df[df['class'] == 0].copy()

# 期間情報をマージ
infected_df = add_period_data(infected_df, 'infected', data_dir)
non_infected_df = add_period_data(non_infected_df, 'non_infected', data_dir)

# 列の整理
infected_df['lt'] = infected_df['content']
non_infected_df['lt'] = non_infected_df['content']

# 必要な列のみ選択
columns_to_keep = ['date', 'period', 'stock_name', 'stock_code', 'title', 'link', 'lt', 'class']
infected_df = infected_df[columns_to_keep]
non_infected_df = non_infected_df[columns_to_keep]

# stock_code がないレコードをフィルタリング（株価データ取得に必要）
print(f"\n  🔍 stock_code フィルタリング:")
infected_before = len(infected_df)
non_infected_before = len(non_infected_df)

infected_df = infected_df[infected_df['stock_code'].notna() & (infected_df['stock_code'] != '')].copy()
non_infected_df = non_infected_df[non_infected_df['stock_code'].notna() & (non_infected_df['stock_code'] != '')].copy()

infected_removed = infected_before - len(infected_df)
non_infected_removed = non_infected_before - len(non_infected_df)

print(f"    - infected: {infected_before}件 → {len(infected_df)}件 (削除: {infected_removed}件)")
print(f"    - non-infected: {non_infected_before}件 → {len(non_infected_df)}件 (削除: {non_infected_removed}件)")

# 保存
infected_path = os.path.join(output_dir_infected, "infected.csv")
non_infected_path = os.path.join(output_dir_non_infected, "non-infected.csv")

infected_df.to_csv(infected_path, index=False, encoding='utf-8-sig')
non_infected_df.to_csv(non_infected_path, index=False, encoding='utf-8-sig')

print(f"\n  💾 保存先:")
print(f"    - infected.csv: {infected_path} ({len(infected_df)}件)")
print(f"    - non-infected.csv: {non_infected_path} ({len(non_infected_df)}件)")

# ==================== 完了サマリー ====================
print("\n" + "=" * 80)
print("✅ 分類完了！")
print("=" * 80)
print("\n📝 分類サマリー:")
print(f"  手法: 決定木アルゴリズム（max_depth=1）による特徴量重要度分析")
print(f"  最重要特徴量: '{top_feature_name}' (重要度: {feature_importances[top_feature_idx]:.4f})")
print(f"  分類基準: 最重要特徴量の有無（客観的・アルゴリズム的）")
print(f"  内部者取引案件 (class=1): {class_1_count}件 ({class_1_count/len(y_final)*100:.1f}%)")
print(f"  非内部者取引案件 (class=0): {class_0_count}件 ({class_0_count/len(y_final)*100:.1f}%)")
print(f"  総件数: {len(df)}件")
print("=" * 80)
