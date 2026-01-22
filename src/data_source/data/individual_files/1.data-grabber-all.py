import requests
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
from urllib.parse import urljoin
import os
import time
import re
import subprocess
import sys

# Base URL
base_url = "https://www.fsa.go.jp"

def wareki_to_seireki(wareki_text):
    """
    和暦（令和・平成・昭和）を西暦yyyy-mm-dd形式に変換します。
    
    Args:
        wareki_text: 「令和5年3月29日」のような和暦文字列、またはcontentテキスト
    
    Returns:
        yyyy-mm-dd形式の日付文字列、見つからない場合は空文字列
    """
    if not wareki_text or not isinstance(wareki_text, str):
        return ""
    
    # 和暦のパターンを探す
    # パターン: 令和5年3月29日、平成12年12月12日、昭和63年1月8日
    pattern = r'(令和|平成|昭和)(\d{1,2})年(\d{1,2})月(\d{1,2})日'
    match = re.search(pattern, wareki_text)
    
    if not match:
        return ""
    
    era = match.group(1)
    era_year = int(match.group(2))
    month = int(match.group(3))
    day = int(match.group(4))
    
    # 和暦を西暦に変換
    if era == '令和':
        # 令和元年 = 2019年
        western_year = 2019 + era_year - 1
    elif era == '平成':
        # 平成元年 = 1989年
        western_year = 1989 + era_year - 1
    elif era == '昭和':
        # 昭和元年 = 1926年
        western_year = 1926 + era_year - 1
    else:
        return ""
    
    # yyyy-mm-dd形式に変換
    try:
        date_str = f"{western_year:04d}-{month:02d}-{day:02d}"
        return date_str
    except:
        return ""

def get_content(url):
    """Fetches text content from a detail page."""
    try:
        resp = requests.get(url, timeout=10)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, 'lxml')
        
        # Try finding the main content div
        content_div = soup.find("div", {"id": "main"})
        if not content_div:
            # Fallback for older pages: sometimes simple div structure or body
            content_div = soup.body
        
        if content_div:
            # Get text, strip whitespace
            content_text = content_div.get_text(separator=" ", strip=True)
            # Remove newlines and carriage returns
            content_text = content_text.replace("\n", "").replace("\r", "")
            
            # Remove machine translation note
            translation_note = "Note: This page is machine translated. Translated pages are not necessarily correct. "
            content_text = content_text.replace(translation_note, "")
            
            return content_text.strip()
        return ""
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

def scrape_year(year, output_dir):
    """Scrapes data for a specific year and saves to CSV."""
    target_url = f"https://www.fsa.go.jp/sesc/news/c_{year}/c_{year}.html"
    print(f"[{year}] Fetching list from {target_url}...")
    
    try:
        resp = requests.get(target_url, timeout=10)
        if resp.status_code == 404:
            print(f"[{year}] Page not found (404). Skipping.")
            return None
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, 'lxml')
    except Exception as e:
        print(f"[{year}] Failed to fetch list page: {e}")
        return None

    data_list = []
    
    # Locate list items
    main_div = soup.find("div", {"id": "main"})
    if not main_div:
        print(f"[{year}] Could not find main div.")
        return None

    ul_list = main_div.find_all("ul")
    target_lis = []
    for ul in ul_list:
        lis = ul.find_all("li")
        if lis:
            target_lis.extend(lis)

    print(f"[{year}] Found {len(target_lis)} list items.")

    if not target_lis:
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_item = {}
        
        for li in target_lis:
            # Extract date from list item as initial fallback
            date_text = li.find(string=True, recursive=False)
            if not date_text:
                date_text = li.get_text().split(' ')[0]
            
            date_col = date_text.strip() if date_text else ""
            
            # Extract Title and Link
            a_tag = li.find("a")
            if a_tag:
                title_col = a_tag.get_text(strip=True)
                link_href = a_tag.get("href")
                full_link = urljoin(base_url, link_href)
                
                future = executor.submit(get_content, full_link)
                future_to_item[future] = {
                    "date": date_col,
                    "title": title_col,
                    "link": full_link
                }
            else:
                continue

        for future in concurrent.futures.as_completed(future_to_item):
            item_data = future_to_item[future]
            try:
                content = future.result()
                item_data["content"] = content
                
                # contentから和暦日付を抽出し、西暦yyyy-mm-dd形式に変換
                seireki_date = wareki_to_seireki(content)
                if seireki_date:
                    item_data["date"] = seireki_date
                
                data_list.append(item_data)
            except Exception as e:
                print(f"[{year}] Error processing item {item_data['link']}: {e}")

    # Create DataFrame and Save
    if data_list:
        df = pd.DataFrame(data_list, columns=["date", "title", "link", "content"])
        output_path = os.path.join(output_dir, f"{year}.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[{year}] Saved {len(df)} records to {output_path}")
        return output_path
    else:
        print(f"[{year}] No data extracted.")
        return None

# ============================================
# 取得期間の設定
# ============================================
year_start = 1998  # 1998年が一番古いです
year_last = 2025   # 取得できる最新のものは実行していただいている年度です（執筆時点では2025年まで可能）
# ============================================

def main():
    # 保存先をスクリプトと同じディレクトリに設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    merged_list = []
    
    print(f"Scraping from {year_start} to {year_last}...")
    print(f"Output directory: {script_dir}")
    
    # 指定された期間（year_startからyear_lastまで）をループ
    for year in range(year_start, year_last + 1):
        csv_file = scrape_year(year, script_dir)
        if csv_file:
            try:
                # 読み込みの際はエンコーディングに注意
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                merged_list.append(df)
            except Exception as e:
                print(f"Error reading generated CSV {csv_file}: {e}")
        
        time.sleep(1)

    # 全てのデータを統合
    if merged_list:
        complete_df = pd.concat(merged_list, ignore_index=True)
        complete_path = os.path.join(script_dir, "complete_dataset_jp.csv")
        complete_df.to_csv(complete_path, index=False, encoding='utf-8-sig')
        print(f"Completed! Merged dataset saved to {complete_path}. Total records: {len(complete_df)}")
        
        # 会社名抽出スクリプトを自動実行
        print("\n" + "=" * 60)
        print("会社名抽出スクリプト（extract_company_names.py）を開始します...")
        print("=" * 60)
        
        extractor_script = os.path.join(script_dir, "extract_company_names.py")
        if os.path.exists(extractor_script):
            try:
                subprocess.run([sys.executable, extractor_script], check=True)
                print("\n会社名抽出が完了しました。")
            except subprocess.CalledProcessError as e:
                print(f"\n会社名抽出スクリプトの実行中にエラーが発生しました: {e}")
        else:
            print(f"\nエラー: 抽出スクリプトが見つかりませんでした: {extractor_script}")
    else:
        print("No data was merged.")

if __name__ == "__main__":
    main()
