#!/usr/bin/env python3
"""
stockchecker.py
- pre-infected.csv と pre-non_infected.csv を読み込む
- stock_name から銘柄コードを再確認 (J-Quants API)
- 銘柄コードが存在する行のみを残して、infected.csv と non-infected.csv に保存する
"""

import jquantsapi
import pandas as pd
import unicodedata
import os
import re

# ============================================
# J-Quants API 設定
# ============================================
# add_stock_codes.py からトークンを継承
REFRESH_TOKEN = "eyJjdHkiOiJKV1QiLCJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiUlNBLU9BRVAifQ.oTtX7untMwNYq3mT4Cm6Ux91ndgZWYopRRy4iHgdqMdbOMu-e7eiDMJz68aU4mCMIVwlsRR5YTxuThgrv01Fgg9ht5rS35Gbv4TXsLws7yBczuQKnDNoIkdMuH9Yl2h27zA_Qtam0XP3018F4QnBXWIpvpwukK4x0z2CHMXp-e6r1k2s_U5OsiiwOPUxmAtBZFpaMLzT9nQFC03b95BUo8iwozjGHO0o6bwqCDLEeqy6Cf4D5sHinFLf_HUzUEbxRDeLcJBPJe0TBgdfDsQUvXDDhnRGpDr0W4wcBMgXbKw5RKxBd4TbxBbY-4kKV2QEeDftPBZi2X_Zyq1vrDWKtQ.0p8rc0Xk4jnisoUr.Uts5UzcwW2aM9s9dnj5k_6LNLIgYvmSKCJFeUO75Lh6XhQmisxwGKmlI6Bmsw5CfkAHoR1tTcSuVochiPMRfL1o_6AWs_V0bFdpOIjfNNN8Kitv9gWukuUK7pmcECi9_uxtn4Jfal3RWatUCep2wajx91_J2JNfXc3wJNc-wyP9OaX1t-p4taRxZUO9C54uGbBPIKo51m7d87B_0iQO9NiYZzIjkl2NTVrw4aWXAt3vS3w8Z6m-DLHI_AGhVtZP84Vr6XAYoZn2ma5SxHNkHyeutQEeIeBcUBlQt4c7Is9S7phqAG-v62g7dL9QWUb28MmAk72N8cfFGonOGpkwHIb92PcLeSqpvphNuCA0n6fGM4AcECh1D7y8JBfpuJVTHGdZJYmOfMmkYEzcyRuC8OxT1gzaThFh3faSMEGUKR-8g8tAFuKU0qjpYn8x6gP-VYL07HZ7mnmT9vuECdyVgGZOU2imDMmOHLQSUYr5qxrK7wmkAiacebVHp86aiCW3BoC9YOV7IZzFuaMfLRKB8PY0FGPudZb9JB2slxw2L9yBsclzOhLEn4aI_FHHxsFIQg9VXoVvsiVeUSOKFljsFTwv7C3WOYiG6wS2l65YBFVOfY4Ks6hpyUh5LCrVVOY6qJVbAGrlKQET2liP9tilp4QrflZjHlFQjLoHbZ-DJzlO9EeeANT0tgnC0gQnidozPgNORW8DELAg3B77s4ADQsmQHOYsSW7fa1k6T8Zm2c_HdcUyZJw2MF8QMXujs9_WV93YeL8jHznv270iYNJsKb0avPnBWVJGzT0fwdZ2khLEaZ1H6_CDzbfA1h8pjnMw62Y6R6yrf6TusOIih80-aIX7YTiJA7HuJhYZtSdF5CTvtF-aGT8eYUVtChK2zuZ6iXRLVeBOfeQyabsrb7BVo77hTJYoF-HTG02XzZPydFcrN4kFosEvOQLqbvSlCLq9iuRgGWQBFefDSM7hUMWraWIMQCI-nkgBd9sKss1nJLdtOXvOOPDmrviuiQmYM7XFsU15b6ipe8DJYa1eJwHeo2ierfN_vwYYQdP0EPfOozINnhB3gBneEg9-8zIRVcaWifqL1woD74px-MF2SJBhGNnF0sQPNx10nAk7m3ZSeIiBB8f-_3H7-5l6Qpt-YHpiOW7LUXMf8aIrnHsV0-_B33NBx3Sut24spWhdmRBER0pLHCD3WDY0V2sztcjRxJqJoKGVLWu9xVbkDLPjYw1OKTEMns2DwH7PWjBezOqogaEC02cjdSAkdWrn_0u0P-VtdXGMMialxawjQ8sX3wt4IKDdvsnCKATQWaen5phuLuPiy9F76x5T3YqjTODbih_u5DjbqqsRKqA1ZIw.f0FiL47EYyY35rz71anLMQ"
class JQuantsMasterManager:
    def __init__(self, refresh_token: str = None):
        self.cli = jquantsapi.Client(refresh_token=refresh_token)
        self.df_master = None

    def fetch_and_prepare(self):
        print("--- J-Quants APIからマスターデータを取得中 ---")
        try:
            data = self.cli.get_list()
            self.df_master = pd.DataFrame(data)
            
            name_col = 'CompanyName'
            full_name_col = 'CompanyNameFull' if 'CompanyNameFull' in self.df_master.columns else None

            self.df_master['NormalizedName'] = self.df_master[name_col].apply(
                lambda x: unicodedata.normalize('NFKC', str(x)) if pd.notnull(x) else ""
            )

            if full_name_col:
                self.df_master['NormalizedNameFull'] = self.df_master[full_name_col].apply(
                    lambda x: unicodedata.normalize('NFKC', str(x)) if pd.notnull(x) else ""
                )
            else:
                self.df_master['NormalizedNameFull'] = self.df_master['NormalizedName']

            print(f"取得完了: {len(self.df_master)} 銘柄")
        except Exception as e:
            print(f"マスターデータの取得中にエラーが発生しました: {e}")
            raise

    def search_code(self, query: str):
        if not query or not isinstance(query, str) or query == "nan":
            return ""

        if self.df_master is None:
            self.fetch_and_prepare()

        norm_query = unicodedata.normalize('NFKC', query)
        clean_query = re.sub(r'株式会社|有限会社|合同会社|合資会社|合名会社|\(株\)', '', norm_query).strip()
        
        results = self.df_master[self.df_master['NormalizedName'] == clean_query]
        
        if results.empty:
            results = self.df_master[
                self.df_master['NormalizedName'].str.contains(clean_query, case=False, na=False) |
                self.df_master['NormalizedNameFull'].str.contains(norm_query, case=False, na=False)
            ]

        if not results.empty:
            code = results.iloc[0]['Code']
            code_str = str(code)
            if len(code_str) == 5 and code_str.endswith('0'):
                return code_str[:4]
            return code_str
        
        return ""

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # ファイル設定
    files_to_process = [
        {
            "input": os.path.join(data_dir, "infected", "pre-infected.csv"),
            "output": os.path.join(data_dir, "infected", "infected.csv")
        },
        {
            "input": os.path.join(data_dir, "non_infected", "pre-non_infected.csv"),
            "output": os.path.join(data_dir, "non_infected", "non-infected.csv")
        }
    ]

    # J-Quantsマネージャーの初期化
    try:
        manager = JQuantsMasterManager(refresh_token=REFRESH_TOKEN)
        manager.fetch_and_prepare()
    except Exception as e:
        print(f"J-Quants APIの初期化に失敗しました: {e}")
        return

    for f_info in files_to_process:
        input_path = f_info["input"]
        output_path = f_info["output"]
        
        if not os.path.exists(input_path):
            print(f"スキップ: {input_path} が見つかりません。")
            continue
            
        print(f"\n処理中: {os.path.basename(input_path)}")
        try:
            df = pd.read_csv(input_path, encoding='utf-8-sig')
            
            if 'stock_name' not in df.columns:
                print(f"  エラー: 'stock_name' カラムが見つかりません。")
                continue
            
            # 銘柄コードを再精査して、あれば保持する
            valid_rows = []
            unique_names = {}
            
            for _, row in df.iterrows():
                name = str(row['stock_name'])
                if name not in unique_names:
                    code = manager.search_code(name)
                    unique_names[name] = code
                
                # 銘柄コードが見つかった場合のみ保持
                if unique_names[name]:
                    row['stock_code'] = unique_names[name]
                    valid_rows.append(row)
            
            if valid_rows:
                new_df = pd.DataFrame(valid_rows)
                
                # カラム順序の整理: date, stock_name, stock_code, title, ...
                cols = list(new_df.columns)
                if 'stock_name' in cols and 'stock_code' in cols:
                    cols.remove('stock_code')
                    name_idx = cols.index('stock_name')
                    cols.insert(name_idx + 1, 'stock_code')
                    new_df = new_df[cols]
                
                new_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"  完了: {len(new_df)}/{len(df)} 件の有効なデータを保存しました -> {os.path.basename(output_path)}")
            else:
                print(f"  警告: 有効な銘柄コードを持つデータがありませんでした。")
                
        except Exception as e:
            print(f"  エラー発生: {e}")

if __name__ == "__main__":
    main()
