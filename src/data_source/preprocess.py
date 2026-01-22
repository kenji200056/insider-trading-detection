# -*- coding: utf-8 -*-
"""
作成日: 2018年6月13日
修正日: 7月18日
作成者: Sheikh Rabiul Islam
目的: 機械学習アルゴリズムに入力できるようにデータを前処理する
入力: data/infected/ フォルダ内の json形式（実際はcsv）の感染ログファイル
      data/non_infected/ フォルダ内の json形式（実際はcsv）の非感染ログファイル
出力: result/ フォルダ内の特徴量ベクトル
      result/ フォルダ内の特徴量ベクトルインデックスマップ
      中間結果は result/interim/ フォルダに保存される（例: フィルタリング後のログ）
"""
# フォルダパスの設定 (絶対パス化)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path_infected = os.path.join(current_dir, "data/infected/")
folder_path_non_infected = os.path.join(current_dir, "data/non_infected/")
folder_path_result = os.path.join(current_dir, "result/")
folder_path_result_interim = os.path.join(current_dir, "result/interim/")


# MeCabのインポート
import MeCab

# モジュールのインポート
import os
import pandas as pd
import numpy as np
# from generalFunctions import *  <-- 削除: 使われていないため

# グローバルコンテナ
features = []
features_infected = []
features_non_infected = []

# 特徴量として考慮したくない単語のリスト（ストップワード）
stopwords_path = os.path.join(current_dir, 'stopwords.txt')
wildcards_path = os.path.join(current_dir, 'wildcards.txt')
user_dic_path = os.path.join(current_dir, 'mecab_userdic/user.dic')

stopwords = set(w.rstrip() for w in open(stopwords_path, encoding='utf-8'))
wildcards = set(w.rstrip() for w in open(wildcards_path, encoding='utf-8') if os.path.exists(wildcards_path))

# MeCabトークナイザの初期化（ユーザー辞書を使用）
mecab_args = f'-d /usr/local/lib/mecab/dic/ipadic'
if os.path.exists(user_dic_path):
    mecab_args += f' -u {user_dic_path}'
    print(f"MeCab: ユーザー辞書を読み込みました: {user_dic_path}")
else:
    print("警告: ユーザー辞書が見つかりません。標準辞書のみを使用します。")

tagger = MeCab.Tagger(mecab_args)

# 個々のログをループし、さらにフィルタリングし、特徴量データベースを構築し、未見の単語なら追加する
def build_features(path, file_name):
    list_row = []   
    file_path = os.path.join(path, file_name)
    #print(file_path)
    data_df = pd.read_csv(file_path)
    #print(data_df)
    for i, r in data_df.iterrows():
        #print("i",i)
        #print(r)
        
        title = r['title']
        litigation = r['lt']      
        #print(litigation)
        litigation_combined = str(title) + str(litigation)
        
        # MeCabでトークナイズ（分かち書き）
        # parseToNode()を使って単語リストを取得
        node = tagger.parseToNode(litigation_combined)
        litigation_combined = []
        while node:
            # 表層形（surface）を取得
            if node.surface:
                litigation_combined.append(node.surface)
            node = node.next
        
        #print(litigation_combined)
        
        row =[]
        for i in range(0,len(litigation_combined)):
            token = litigation_combined[i]
            # 文字列長チェックとストップワード除外
            if len(token) > 1 and len(token) < 20 and token.lower() not in stopwords:
                val = str(token)
                #val ="%r"%val #converting into raw string
                if val.isdigit():
                    val = str(len(val)) + "digit"
                row.append(val)
                #print("row")
        list_row.append(row)   
    # リストのリストを返す。リストの各要素は個々のログの特徴量を含み、
    # 内部リストはそのログの個々の特徴量を含む。
    #print(list_row)
    return list_row




print("*******************************************************************")
print ("infectedログ（インサイダー関連）から特徴量を抽出中（最初の5件を表示）... ")
print("*******************************************************************")

print("*******************************************************************")
print ("non-infectedログ（インサイダー関連）から特徴量を抽出中... ")
print("*******************************************************************")
# infected.csv を直接指定
target_file_infected = "infected.csv"
path_infected = folder_path_infected
print ("ファイル処理開始: %s ... " % target_file_infected)
try:
    features_infected = features_infected + build_features(path_infected, target_file_infected)
    print("\n")
except Exception as e:
    print(f"エラー発生: {e}")
    pass
        

print("*******************************************************************")
print ("non-infectedログから特徴量を抽出中... ")
print("*******************************************************************")
# non-infected.csv を直接指定（ハイフン注意）
target_file_non_infected = "non-infected.csv"
path_non_infected = folder_path_non_infected
print ("ファイル処理開始: %s ... " % target_file_non_infected)
try:
    features_non_infected = features_non_infected + build_features(path_non_infected, target_file_non_infected)
    print("\n")
except Exception as e:
    print(f"エラー発生: {e}")
    pass


print("*******************************************************************")
print("最初の5つの感染特徴量:")
print("*******************************************************************")
print(features_infected[:5])

print("*******************************************************************")
print("最初の5つの非感染特徴量:")
print("*******************************************************************")
print(features_non_infected[:5])


# 特徴量からインデックスへのマッピング
feature_index_map = {}
counter = 0

for feature_row in features_infected:
    for feature in feature_row:
        if feature not in feature_index_map:
            feature_index_map[feature] = counter
            counter = counter + 1
            
for feature_row in features_non_infected:
    for feature in feature_row:
        if feature not in feature_index_map:
            feature_index_map[feature] = counter
            counter = counter + 1
            

# 特徴量ベクトル化
def features_to_vector(features, cls):
    row = np.zeros(len(feature_index_map) +1 ) # 最後の列はクラス属性用（1=感染、0=非感染）
    for f in features:
        index = feature_index_map[f]
        row[index] = 1
        #@todo 正規化、もし役立つなら
        row[-1] = cls
    return row
        
data_feature_vector = np.zeros(((len(features_infected) + len(features_non_infected)), (len(feature_index_map) + 1)))


row_count =0
for feature_row in features_infected:
    vector = features_to_vector(feature_row, 1) # 1 = 感染（Insider）
    data_feature_vector[row_count, :] = vector
    row_count = row_count + 1
    
for feature_row in features_non_infected:
    vector = features_to_vector(feature_row, 0) # 0 = 非感染（Non-Insider）
    data_feature_vector[row_count, :] = vector
    row_count = row_count + 1


print("*******************************************************************")
print("特徴量ベクトル:")
print("*******************************************************************")


data_feature_vector_df = pd.DataFrame(data_feature_vector)
path = os.path.join(folder_path_result, "feature_vector.csv")
data_feature_vector_df.to_csv(path, sep=',')
print(data_feature_vector_df[:5])

feature_index_map_list = []
for key, val in feature_index_map.items():
    temp = [val,key]
    feature_index_map_list.append(temp)

# 最後の列はクラス
temp = [len(feature_index_map_list), "class"]
feature_index_map_list.append(temp)

feature_index_map_df = pd.DataFrame(feature_index_map_list, columns = ['idx', 'feature'])
path = os.path.join(folder_path_result, "feature_vector_index_map.csv")
feature_index_map_df.to_csv(path, sep=',', index = False)

print("*******************************************************************")
print("特徴量インデックスマップ（最初の50件を表示）:")
print("*******************************************************************")
print(feature_index_map_list[:50])

print("\n\n前処理が完了しました。これで個々の機械学習アルゴリズムのコードを実行できます。")
