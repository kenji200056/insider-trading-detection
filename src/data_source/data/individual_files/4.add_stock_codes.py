#!/usr/bin/env python3
"""
J-Quants APIを使用して会社名(stock_name)から銘柄コード(stock_code)を取得し、CSVに追加するスクリプト
"""

import jquantsapi
import pandas as pd
import unicodedata
import os
import glob
import re
import time

# ============================================
# J-Quants API 設定
# ============================================
# ここにリフレッシュトークンを直接入力するか、
# ホームディレクトリに .jquants-api.toml を作成してください。
REFRESH_TOKEN = "eyJjdHkiOiJKV1QiLCJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiUlNBLU9BRVAifQ.oTtX7untMwNYq3mT4Cm6Ux91ndgZWYopRRy4iHgdqMdbOMu-e7eiDMJz68aU4mCMIVwlsRR5YTxuThgrv01Fgg9ht5rS35Gbv4TXsLws7yBczuQKnDNoIkdMuH9Yl2h27zA_Qtam0XP3018F4QnBXWIpvpwukK4x0z2CHMXp-e6r1k2s_U5OsiiwOPUxmAtBZFpaMLzT9nQFC03b95BUo8iwozjGHO0o6bwqCDLEeqy6Cf4D5sHinFLf_HUzUEbxRDeLcJBPJe0TBgdfDsQUvXDDhnRGpDr0W4wcBMgXbKw5RKxBd4TbxBbY-4kKV2QEeDftPBZi2X_Zyq1vrDWKtQ.0p8rc0Xk4jnisoUr.Uts5UzcwW2aM9s9dnj5k_6LNLIgYvmSKCJFeUO75Lh6XhQmisxwGKmlI6Bmsw5CfkAHoR1tTcSuVochiPMRfL1o_6AWs_V0bFdpOIjfNNN8Kitv9gWukuUK7pmcECi9_uxtn4Jfal3RWatUCep2wajx91_J2JNfXc3wJNc-wyP9OaX1t-p4taRxZUO9C54uGbBPIKo51m7d87B_0iQO9NiYZzIjkl2NTVrw4aWXAt3vS3w8Z6m-DLHI_AGhVtZP84Vr6XAYoZn2ma5SxHNkHyeutQEeIeBcUBlQt4c7Is9S7phqAG-v62g7dL9QWUb28MmAk72N8cfFGonOGpkwHIb92PcLeSqpvphNuCA0n6fGM4AcECh1D7y8JBfpuJVTHGdZJYmOfMmkYEzcyRuC8OxT1gzaThFh3faSMEGUKR-8g8tAFuKU0qjpYn8x6gP-VYL07HZ7mnmT9vuECdyVgGZOU2imDMmOHLQSUYr5qxrK7wmkAiacebVHp86aiCW3BoC9YOV7IZzFuaMfLRKB8PY0FGPudZb9JB2slxw2L9yBsclzOhLEn4aI_FHHxsFIQg9VXoVvsiVeUSOKFljsFTwv7C3WOYiG6wS2l65YBFVOfY4Ks6hpyUh5LCrVVOY6qJVbAGrlKQET2liP9tilp4QrflZjHlFQjLoHbZ-DJzlO9EeeANT0tgnC0gQnidozPgNORW8DELAg3B77s4ADQsmQHOYsSW7fa1k6T8Zm2c_HdcUyZJw2MF8QMXujs9_WV93YeL8jHznv270iYNJsKb0avPnBWVJGzT0fwdZ2khLEaZ1H6_CDzbfA1h8pjnMw62Y6R6yrf6TusOIih80-aIX7YTiJA7HuJhYZtSdF5CTvtF-aGT8eYUVtChK2zuZ6iXRLVeBOfeQyabsrb7BVo77hTJYoF-HTG02XzZPydFcrN4kFosEvOQLqbvSlCLq9iuRgGWQBFefDSM7hUMWraWIMQCI-nkgBd9sKss1nJLdtOXvOOPDmrviuiQmYM7XFsU15b6ipe8DJYa1eJwHeo2ierfN_vwYYQdP0EPfOozINnhB3gBneEg9-8zIRVcaWifqL1woD74px-MF2SJBhGNnF0sQPNx10nAk7m3ZSeIiBB8f-_3H7-5l6Qpt-YHpiOW7LUXMf8aIrnHsV0-_B33NBx3Sut24spWhdmRBER0pLHCD3WDY0V2sztcjRxJqJoKGVLWu9xVbkDLPjYw1OKTEMns2DwH7PWjBezOqogaEC02cjdSAkdWrn_0u0P-VtdXGMMialxawjQ8sX3wt4IKDdvsnCKATQWaen5phuLuPiy9F76x5T3YqjTODbih_u5DjbqqsRKqA1ZIw.f0FiL47EYyY35rz71anLMQ" 
# ============================================

class JQuantsMasterManager:
    def __init__(self, refresh_token: str = None):
        self.cli = jquantsapi.Client(refresh_token=refresh_token)
        self.df_master = None

    def fetch_and_prepare(self):
        """
        最新の全上場銘柄一覧を取得し、検索用にデータ型を整理する
        """
        print("--- J-Quants APIからマスターデータを取得中 ---")
        try:
            data = self.cli.get_list()
            self.df_master = pd.DataFrame(data)
            
            # --- ここでカラムの存在をチェック ---
            # 基本となる銘柄名カラム
            name_col = 'CompanyName'
            # フルネームカラム（存在しない場合があるため安全に処理）
            full_name_col = 'CompanyNameFull' if 'CompanyNameFull' in self.df_master.columns else None

            # 検索用カラムの作成
            self.df_master['NormalizedName'] = self.df_master[name_col].apply(
                lambda x: unicodedata.normalize('NFKC', str(x)) if pd.notnull(x) else ""
            )

            if full_name_col:
                self.df_master['NormalizedNameFull'] = self.df_master[full_name_col].apply(
                    lambda x: unicodedata.normalize('NFKC', str(x)) if pd.notnull(x) else ""
                )
            else:
                # 存在しない場合は、通常の銘柄名で代用しておく
                self.df_master['NormalizedNameFull'] = self.df_master['NormalizedName']

            print(f"取得完了: {len(self.df_master)} 銘柄")
        except Exception as e:
            # e 自体を print するのではなく、詳細なトレースを確認可能にする
            print(f"マスターデータの取得中にエラーが発生しました: {e}")
            raise

    def search_code(self, query: str):
        if not query or not isinstance(query, str) or query == "nan": # pandasのNaN対策
            return ""

        if self.df_master is None:
            self.fetch_and_prepare()

        norm_query = unicodedata.normalize('NFKC', query)
        clean_query = re.sub(r'株式会社|有限会社|合同会社|合資会社|合名会社|\(株\)', '', norm_query).strip()
        
        # 1. 完全一致（正規化済み名称）
        results = self.df_master[self.df_master['NormalizedName'] == clean_query]
        
        if results.empty:
            # 2. 部分一致（NormalizedName と NormalizedNameFull の両方から）
            results = self.df_master[
                self.df_master['NormalizedName'].str.contains(clean_query, case=False, na=False) |
                self.df_master['NormalizedNameFull'].str.contains(norm_query, case=False, na=False)
            ]

        if not results.empty:
            code = results.iloc[0]['Code']
            # J-Quantsの5桁コード（末尾0）を4桁に変換
            code_str = str(code)
            if len(code_str) == 5 and code_str.endswith('0'):
                return code_str[:4]
            return code_str
        
        return ""
        
def process_csv_files():
    # J-Quantsマネージャーの初期化
    try:
        # トークンが空でない場合はそれを使用、空なら設定ファイル(.toml)を探す
        token = REFRESH_TOKEN if REFRESH_TOKEN else None
        manager = JQuantsMasterManager(refresh_token=token)
        manager.fetch_and_prepare()
    except Exception as e:
        print(f"J-Quants APIの初期化に失敗しました。認証設定を確認してください: {e}")
        return

    # 実行ディレクトリの取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(script_dir, "*.csv"))
    
    # 統合ファイルはスキップするか最後に処理する
    if os.path.join(script_dir, "complete_dataset_jp.csv") in csv_files:
        csv_files.remove(os.path.join(script_dir, "complete_dataset_jp.csv"))
        csv_files.append(os.path.join(script_dir, "complete_dataset_jp.csv"))

    for file_path in csv_files:
        print(f"\n処理中: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            if 'stock_name' not in df.columns:
                print(f"  スキップ: 'stock_name' カラムが見つかりません。")
                continue
            
            # 銘柄コードを取得
            stock_codes = []
            unique_names = {} # キャッシュ用
            
            for name in df['stock_name']:
                if name not in unique_names:
                    code = manager.search_code(str(name))
                    unique_names[name] = code
                stock_codes.append(unique_names[name])
            
            # stock_codeカラムを追加
            df['stock_code'] = stock_codes
            
            # カラム順序の整理: date, stock_name, stock_code, title, ...
            cols = list(df.columns)
            if 'stock_name' in cols and 'stock_code' in cols:
                cols.remove('stock_code')
                name_idx = cols.index('stock_name')
                cols.insert(name_idx + 1, 'stock_code')
                df = df[cols]
            
            # 保存
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            found_count = sum(1 for c in stock_codes if c)
            print(f"  完了: {found_count}/{len(df)} 件の銘柄コードを特定しました。")
            
        except Exception as e:
            print(f"  エラー発生: {e}")

if __name__ == "__main__":
    process_csv_files()
