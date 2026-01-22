#!/usr/bin/env python3
"""
会社名抽出スクリプト（機械学習ベース）
=====================================
CSVファイルのtitleとcontentカラムから訴えられた会社名を抽出し、
新しいカラム「company_name」として追加します。

使用技術:
- Transformers (Hugging Face)
- 日本語BERT NERモデル (lxyuan/span-marker-bert-base-multilingual-uncased-multinerd)
または stockmark/ner-jp モデル

このモデルは機械学習ベースで、様々な会社形式を認識できます：
- 株式会社（前株・後株両対応）
- 有限会社、合同会社、合資会社、合名会社
- 海外企業（.co, Inc., Corp., Ltd.など）
- 証券会社、銀行など
"""

import os
import re
import glob
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================
# NERモデル設定
# ============================================
NER_MODEL_NAME = "tsmatz/xlm-roberta-ner-japanese"  # 日本語NERモデル

# グローバル変数でパイプラインをキャッシュ
_ner_pipeline = None


def get_ner_pipeline():
    """NERパイプラインをロード（初回のみ）"""
    global _ner_pipeline
    if _ner_pipeline is None:
        print("NERモデルをロード中...")
        from transformers import pipeline
        try:
            # 日本語NERモデルをロード
            _ner_pipeline = pipeline(
                "ner",
                model=NER_MODEL_NAME,
                aggregation_strategy="simple"
            )
            print(f"モデル '{NER_MODEL_NAME}' をロードしました。")
        except Exception as e:
            print(f"モデルのロードに失敗しました: {e}")
            print("フォールバック: 正規表現ベースの抽出を使用します。")
            _ner_pipeline = "fallback"
    return _ner_pipeline


def extract_organizations_with_ner(text, max_length=512):
    """
    NERモデルを使用して組織名（会社名）を抽出します。
    """
    if not text or not isinstance(text, str):
        return []
    
    pipeline_obj = get_ner_pipeline()
    
    if pipeline_obj == "fallback":
        return extract_with_regex(text)
    
    # テキストを適切な長さに制限
    text_to_process = text[:max_length]
    
    try:
        results = pipeline_obj(text_to_process)
        organizations = []
        for entity in results:
            # ORG（組織）ラベルのエンティティを抽出
            label = entity.get("entity_group", entity.get("entity", ""))
            if "ORG" in label.upper() or "ORGANIZATION" in label.upper():
                org_name = entity.get("word", "").strip()
                # 「##」などのサブワードトークンを除去
                org_name = org_name.replace("##", "")
                if org_name and len(org_name) > 1:
                    organizations.append(org_name)
        return organizations
    except Exception as e:
        print(f"NER処理エラー: {e}")
        return extract_with_regex(text)


def extract_with_regex(text):
    """
    正規表現を使用して様々な形式の会社名を抽出します。
    接続詞や助詞で分割し、会社形式を含む最小の会社名のみを抽出します。
    """
    if not text or not isinstance(text, str):
        return []
    
    # 会社形式のリスト
    company_suffixes = ['株式会社', '有限会社', '合同会社', '合資会社', '合名会社', '証券会社', '証券株式会社', '銀行']
    company_prefixes = ['株式会社', '有限会社', '合同会社', '合資会社', '合名会社']
    
    companies = []
    
    # パターン1: 後株パターン（〇〇株式会社）
    # 接続詞・助詞の後ろから会社形式まで抽出
    for suffix in company_suffixes:
        # より広いパターンで検索し、接続詞・助詞で分割
        pattern = r'([一-龯ぁ-んァ-ンA-Za-zＡ-Ｚａ-ｚ0-9０-９\s・\-ー＆&]+?)' + re.escape(suffix)
        matches = re.findall(pattern, text)
        
        for match in matches:
            # 接続詞・助詞で分割して最後の部分のみ取得
            # 「が」「による」「の」「から」「との」「への」などの後ろの部分
            particles = ['が', 'による', 'から', 'との', 'への', 'について', 'における', 'に対する', 'に係る', 'と', 'を', 'に', 'で']
            cleaned = match
            for particle in particles:
                if particle in cleaned:
                    # 最後に出現した接続詞・助詞の後ろを取得
                    parts = cleaned.rsplit(particle, 1)
                    if len(parts) > 1:
                        cleaned = parts[-1]
            
            cleaned = cleaned.strip()
            # 無効な抽出を除外
            if cleaned and len(cleaned) > 1 and len(cleaned) < 100:
                if re.search(r'[一-龯ぁ-んァ-ンA-Za-z]', cleaned):
                    companies.append(cleaned + suffix)
    
    # パターン2: 前株パターン（株式会社〇〇）
    for prefix in company_prefixes:
        pattern = re.escape(prefix) + r'([一-龯ぁ-んァ-ンA-Za-zＡ-Ｚａ-ｚ0-9０-９\s・\-ー]+?)(?:における|に対する|に係る|の|は|が|を|、)'
        matches = re.findall(pattern, text)
        for match in matches:
            company = (prefix + match).strip()
            if company and len(company) > 3 and len(company) < 50:
                if re.search(r'[一-龯ぁ-んァ-ンA-Za-z]', company):
                    companies.append(company)
    
    # パターン3: 海外企業形式
    foreign_pattern = r'([A-Z][A-Za-z0-9\s\.\-&]+?)(?:\s+(?:Inc\.|Corp\.|Corporation|Ltd\.|LLC|Co\.,?\s*Ltd\.|Company|Holdings|Group))'
    matches = re.findall(foreign_pattern, text)
    for match in matches:
        company = match.strip()
        if company and len(company) > 1 and len(company) < 50:
            companies.append(company)
    
    return companies


def clean_company_name(name):
    """
    会社名をクリーンアップします。
    """
    if not name:
        return ""
    
    # 不要な前後の文字を削除
    name = re.sub(r'^[\s　、。・]+', '', name)
    name = re.sub(r'[\s　、。・]+$', '', name)
    
    # 重複した会社形式を整理
    name = re.sub(r'(株式会社|有限会社|合同会社)(\1)+', r'\1', name)
    
    # 会社名の後に「の従業員」「の役員」などが続く場合は削除
    name = re.sub(r'の(従業員|役員|代表者|取締役|社員|関係者|契約|公開|情報).*$', '', name)
    
    # 会社名の後に続く不要なテキストを削除
    # 「役員」「従業員」「から」「と」「に」などで終わる場合は削除
    trailing_patterns = [
        r'(役員|従業員|代表者|取締役|社員|関係者)(から|らの|の)?$',
        r'(と|に|を|が|から|との|への|への)$',
        r'(における|に対する|に係る|について)$',
    ]
    for pattern in trailing_patterns:
        name = re.sub(pattern, '', name)
    
    # 先頭の不要なテキストを削除
    leading_patterns = [
        # 日付パターン（「月15日」「年12月」など）
        r'^.*?[0-9]{1,2}月[0-9]{1,2}日\s*',
        r'^.*?年[0-9]{1,2}月\s*',
        # 政府機関名
        r'^.*?(証券取引等監視委員会|金融庁|財務局|財務支局)\s*',
        # 「長が」などの残余
        r'^.*?長が\s*',
        # その他
        r'^(令和|平成|昭和)\S+\s*',
        r'^(会社員|個人|元|前|現)\s*',
        r'^(による|に対する|における|関する)\s*',
        r'^(海外に居住する個人による)\s*',
        r'^(関東財務局長が|東海財務局長が|近畿財務局長が|北海道財務局長が|九州財務局長が|福岡財務支局長が|沖縄総合事務局長が)\s*',
    ]
    for pattern in leading_patterns:
        name = re.sub(pattern, '', name)
    
    return name.strip()


def extract_main_company(title, content):
    """
    タイトルとコンテンツから主要な会社名を抽出します。
    優先順位:
    1. タイトルからの直接抽出（タイトルに会社名が含まれることが多い）
    2. コンテンツからの抽出
    """
    # 会社形式のリスト
    company_suffixes = ['株式会社', '有限会社', '合同会社', '合資会社', '合名会社', '証券会社', '証券株式会社', '銀行']
    company_prefixes = ['株式会社', '有限会社', '合同会社', '合資会社', '合名会社']
    
    # まずタイトルから試行
    if title and isinstance(title, str):
        # 後株パターン（〇〇株式会社）から抽出
        for suffix in company_suffixes:
            pattern = r'([一-龯ぁ-んァ-ンA-Za-zＡ-Ｚａ-ｚ0-9０-９\s・\-ー＆&]+?)' + re.escape(suffix)
            match = re.search(pattern, title)
            if match:
                company_part = match.group(1)
                
                # 接続詞・助詞で分割して最後の部分のみ取得
                particles = ['が', 'による', 'から', 'との', 'への', 'について', 'における', 'に対する', 'に係る', 'と', 'を', 'に', 'で']
                cleaned = company_part
                for particle in particles:
                    if particle in cleaned:
                        parts = cleaned.rsplit(particle, 1)
                        if len(parts) > 1:
                            cleaned = parts[-1]
                
                cleaned = cleaned.strip()
                if cleaned and len(cleaned) > 1:
                    full_company = cleaned + suffix
                    full_company = clean_company_name(full_company)
                    if full_company and len(full_company) > 2:
                        return full_company
        
        # 前株パターン（株式会社〇〇）
        for prefix in company_prefixes:
            pattern = re.escape(prefix) + r'([一-龯ぁ-んァ-ンA-Za-zＡ-Ｚａ-ｚ0-9０-９\s・\-ー]+?)(?:における|に対する|に係る|の|は|が|を|、)'
            match = re.search(pattern, title)
            if match:
                company = clean_company_name(prefix + match.group(1))
                if company and len(company) > 3:
                    return company
        
        # NERでタイトルから抽出
        orgs = extract_organizations_with_ner(title)
        if orgs:
            # 最も長い組織名を選択（通常は会社名が最も詳細）
            return clean_company_name(max(orgs, key=len))
    
    # コンテンツから抽出
    if content and isinstance(content, str):
        # コンテンツの先頭部分（会社名は通常文頭に出演）
        content_head = content[:1000]
        
        # 「当社」の前に出現する会社名を探す
        # 例: "〇〇株式会社（以下「当社」）"
        for suffix in company_suffixes:
            tousha_pattern = r'([一-龯ぁ-んァ-ンA-Za-zＡ-Ｚａ-ｚ0-9０-９\s・\-ー＆&]+?' + re.escape(suffix) + r')(?:（|[\(].*?(?:以下「?当社|以下「?会社))'
            match = re.search(tousha_pattern, content_head)
            if match:
                company = clean_company_name(match.group(1))
                if company and len(company) > 2:
                    return company
        
        # 正規表現で抽出
        orgs_regex = extract_with_regex(content_head)
        if orgs_regex:
            # 最も長い会社名を選択
            longest = max(orgs_regex, key=len)
            cleaned = clean_company_name(longest)
            if cleaned and len(cleaned) > 2:
                return cleaned
        
        # NERでコンテンツから抽出
        orgs = extract_organizations_with_ner(content_head)
        if orgs:
            return clean_company_name(max(orgs, key=len))
    
    return ""


def process_csv_file(file_path):
    """
    CSVファイルを処理し、company_nameカラムを追加します。
    """
    print(f"\n処理中: {file_path}")
    
    try:
        # CSVを読み込み
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 必要なカラムがあるか確認
        if 'title' not in df.columns and 'content' not in df.columns:
            print(f"  スキップ: titleとcontentカラムが見つかりません")
            return False
        
        # 会社名を抽出
        company_names = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            title = row.get('title', '')
            content = row.get('content', '')
            
            company = extract_main_company(title, content)
            company_names.append(company)
            
            if (idx + 1) % 10 == 0:
                print(f"  進捗: {idx + 1}/{total_rows} 行処理完了")
        
        # stock_nameカラムを追加または更新（dateの次、titleの前に配置）
        if 'stock_name' in df.columns:
            df['stock_name'] = company_names
        elif 'company_name' in df.columns:
            # 既存のcompany_nameがあれば削除（stock_nameとして作成し直す）
            df['stock_name'] = company_names
            df = df.drop(columns=['company_name'])
        else:
            df['stock_name'] = company_names
        
        # カラムの順序を修正: date, stock_name, title, ...
        cols = list(df.columns)
        if 'date' in cols and 'stock_name' in cols:
            cols.remove('stock_name')
            date_idx = cols.index('date')
            cols.insert(date_idx + 1, 'stock_name')
            df = df[cols]
        
        # CSVを保存
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        # 抽出結果のサマリー
        extracted_count = sum(1 for c in company_names if c)
        print(f"  完了: {extracted_count}/{total_rows} 行で会社名を抽出")
        
        # サンプル表示
        sample_extracted = [(row['title'][:30] + '...', company) 
                           for (_, row), company in zip(df.head(5).iterrows(), company_names[:5]) 
                           if company]
        if sample_extracted:
            print("  サンプル抽出結果:")
            for title, company in sample_extracted[:3]:
                print(f"    - '{title}' → '{company}'")
        
        return True
        
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    メイン処理: ディレクトリ内のすべてのCSVファイルを処理します。
    """
    # 現在のディレクトリを取得
    script_dir = Path(__file__).parent
    
    # CSVファイルを検索（年別ファイルとcomplete_dataset_jpを対象）
    csv_files = sorted(glob.glob(str(script_dir / "*.csv")))
    
    if not csv_files:
        print("CSVファイルが見つかりません。")
        return
    
    print(f"処理対象: {len(csv_files)} ファイル")
    print("=" * 60)
    
    # complete_dataset_jp.csv は後で処理（他のファイルを先に）
    regular_files = [f for f in csv_files if "complete_dataset" not in f]
    complete_files = [f for f in csv_files if "complete_dataset" in f]
    
    success_count = 0
    fail_count = 0
    
    # 年別ファイルを処理
    for csv_file in regular_files:
        if process_csv_file(csv_file):
            success_count += 1
        else:
            fail_count += 1
    
    # complete_datasetを処理
    for csv_file in complete_files:
        print("\n" + "=" * 60)
        print("complete_dataset_jp.csvを処理中（時間がかかる場合があります）...")
        if process_csv_file(csv_file):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"処理完了: 成功 {success_count} / 失敗 {fail_count}")


if __name__ == "__main__":
    main()
