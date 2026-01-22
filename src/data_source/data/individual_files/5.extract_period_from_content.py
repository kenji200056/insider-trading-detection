#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.extract_period_from_content.py
文章内容（content）から事件期間を自動抽出してperiodカラムに追加

抽出パターン:
- 「平成X年X月から平成Y年Y月まで」
- 「平成X年X月X日から平成Y年Y月Y日まで」
- 「平成X年X月から同Y年Y月まで」（同 = 同じ元号）
- 「令和X年X月から令和Y年Y月まで」
- 西暦パターンも対応
"""

import pandas as pd
import re
import os
from datetime import datetime

def wareki_to_seireki(era, year):
    """
    和暦を西暦に変換

    Args:
        era: 元号（令和、平成、昭和）
        year: 和暦の年

    Returns:
        西暦の年
    """
    year = int(year)
    if era == '令和':
        return 2019 + year - 1
    elif era == '平成':
        return 1989 + year - 1
    elif era == '昭和':
        return 1926 + year - 1
    return None

def extract_period_from_content(content):
    """
    文章内容から期間を抽出

    Args:
        content: 文章内容

    Returns:
        period文字列（YYYY/MM/DD-YYYY/MM/DD形式）、見つからない場合は空文字列
    """
    if pd.isna(content) or not isinstance(content, str):
        return ""

    # 全角数字を半角に変換
    content = content.translate(str.maketrans('０１２３４５６７８９', '0123456789'))

    periods = []

    # パターン1: 「平成X年X月X日から平成Y年Y月Y日まで」「平成X年X月X日から同Y年Y月Y日まで」
    pattern1 = r'(令和|平成|昭和)(\d{1,2})年(\d{1,2})月(?:(\d{1,2})日)?から(?:(令和|平成|昭和)|同|同年)(\d{1,2})年(\d{1,2})月(?:(\d{1,2})日)?まで'
    matches1 = re.finditer(pattern1, content)

    for match in matches1:
        era1 = match.group(1)
        year1 = match.group(2)
        month1 = match.group(3)
        day1 = match.group(4) or '01'

        era2_match = match.group(5)
        if era2_match and era2_match in ['令和', '平成', '昭和']:
            era2 = era2_match
        else:
            era2 = era1  # 「同」または「同年」の場合

        year2 = match.group(6)
        month2 = match.group(7)
        day2 = match.group(8) or '01'

        seireki1 = wareki_to_seireki(era1, year1)
        seireki2 = wareki_to_seireki(era2, year2)

        if seireki1 and seireki2:
            date1 = f"{seireki1}/{int(month1):02d}/{int(day1):02d}"
            date2 = f"{seireki2}/{int(month2):02d}/{int(day2):02d}"
            periods.append((date1, date2))

    # パターン2: 「にかけて」形式 - 「平成X年X月から同Y年Y月にかけて」
    pattern2 = r'(令和|平成|昭和)(\d{1,2})年(\d{1,2})月(?:(\d{1,2})日)?から(?:(令和|平成|昭和)|同|同年)?(\d{1,2})年(\d{1,2})月(?:(\d{1,2})日)?にかけて'
    matches2 = re.finditer(pattern2, content)

    for match in matches2:
        era1 = match.group(1)
        year1 = match.group(2)
        month1 = match.group(3)
        day1 = match.group(4) or '01'

        era2_match = match.group(5)
        if era2_match and era2_match in ['令和', '平成', '昭和']:
            era2 = era2_match
        else:
            era2 = era1

        year2 = match.group(6)
        month2 = match.group(7)
        day2 = match.group(8) or '01'

        seireki1 = wareki_to_seireki(era1, year1)
        seireki2 = wareki_to_seireki(era2, year2)

        if seireki1 and seireki2:
            date1 = f"{seireki1}/{int(month1):02d}/{int(day1):02d}"
            date2 = f"{seireki2}/{int(month2):02d}/{int(day2):02d}"
            periods.append((date1, date2))

    # パターン3: 元号省略パターン - 「平成9年7月1日から10年6月30日までの期間」
    pattern3 = r'(令和|平成|昭和)(\d{1,2})年(\d{1,2})月(\d{1,2})日から(\d{1,2})年(\d{1,2})月(\d{1,2})日まで'
    matches3 = re.finditer(pattern3, content)

    for match in matches3:
        era1 = match.group(1)
        year1 = match.group(2)
        month1 = match.group(3)
        day1 = match.group(4)
        year2 = match.group(5)  # 元号省略
        month2 = match.group(6)
        day2 = match.group(7)

        seireki1 = wareki_to_seireki(era1, year1)
        seireki2 = wareki_to_seireki(era1, year2)  # 同じ元号を使用

        if seireki1 and seireki2:
            date1 = f"{seireki1}/{int(month1):02d}/{int(day1):02d}"
            date2 = f"{seireki2}/{int(month2):02d}/{int(day2):02d}"
            periods.append((date1, date2))

    # パターン4: 「までの間」形式 - 「平成X年X月から同年Y月までの間」
    pattern4 = r'(令和|平成|昭和)(\d{1,2})年(\d{1,2})月(?:(\d{1,2})日)?から(?:(令和|平成|昭和)|同|同年)(\d{1,2})年(\d{1,2})月(?:(\d{1,2})日)?までの間'
    matches4 = re.finditer(pattern4, content)

    for match in matches4:
        era1 = match.group(1)
        year1 = match.group(2)
        month1 = match.group(3)
        day1 = match.group(4) or '01'

        era2_match = match.group(5)
        if era2_match and era2_match in ['令和', '平成', '昭和']:
            era2 = era2_match
        else:
            era2 = era1

        year2 = match.group(6)
        month2 = match.group(7)
        day2 = match.group(8) or '01'

        seireki1 = wareki_to_seireki(era1, year1)
        seireki2 = wareki_to_seireki(era2, year2)

        if seireki1 and seireki2:
            date1 = f"{seireki1}/{int(month1):02d}/{int(day1):02d}"
            date2 = f"{seireki2}/{int(month2):02d}/{int(day2):02d}"
            periods.append((date1, date2))

    # パターン5: 西暦パターン - 「YYYY年MM月DD日からYYYY年MM月DD日まで」
    pattern5 = r'(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?から(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?まで'
    matches5 = re.finditer(pattern5, content)

    for match in matches5:
        year1 = match.group(1)
        month1 = match.group(2)
        day1 = match.group(3) or '01'
        year2 = match.group(4)
        month2 = match.group(5)
        day2 = match.group(6) or '01'

        date1 = f"{year1}/{int(month1):02d}/{int(day1):02d}"
        date2 = f"{year2}/{int(month2):02d}/{int(day2):02d}"
        periods.append((date1, date2))

    if not periods:
        return ""

    # 複数の期間がある場合、最も古い開始日と最も新しい終了日を使用
    if len(periods) == 1:
        return f"{periods[0][0]}-{periods[0][1]}"

    # すべての日付をパースして最小・最大を見つける
    all_dates = []
    for start, end in periods:
        try:
            all_dates.append(datetime.strptime(start, '%Y/%m/%d'))
            all_dates.append(datetime.strptime(end, '%Y/%m/%d'))
        except:
            continue

    if not all_dates:
        return f"{periods[0][0]}-{periods[0][1]}"

    min_date = min(all_dates).strftime('%Y/%m/%d')
    max_date = max(all_dates).strftime('%Y/%m/%d')

    return f"{min_date}-{max_date}"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    input_file = os.path.join(parent_dir, "complete_dataset_jp.csv")
    output_file = os.path.join(parent_dir, "complete_dataset_jp_with_period.csv")

    print("=" * 80)
    print("文章内容から期間を自動抽出")
    print("=" * 80)

    if not os.path.exists(input_file):
        print(f"❌ エラー: {input_file} が見つかりません")
        return

    # データ読み込み
    print(f"\n📖 データ読み込み: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"   総件数: {len(df)}件")

    # 期間抽出
    print(f"\n🔍 文章内容から期間を抽出中...")
    df['period'] = df['content'].apply(extract_period_from_content)

    # 統計
    period_found = df['period'].notna() & (df['period'] != '')
    print(f"\n📊 抽出結果:")
    print(f"   期間が見つかった: {period_found.sum()}件 ({period_found.sum()/len(df)*100:.1f}%)")
    print(f"   期間が見つからない: {(~period_found).sum()}件 ({(~period_found).sum()/len(df)*100:.1f}%)")

    # サンプル表示
    if period_found.sum() > 0:
        print(f"\n📝 抽出サンプル（最初の5件）:")
        sample = df[period_found][['title', 'period']].head(5)
        for idx, row in sample.iterrows():
            print(f"   - {row['title'][:50]}...")
            print(f"     期間: {row['period']}")

    # 保存
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 保存完了: {output_file}")
    print(f"   総件数: {len(df)}件")
    print("=" * 80)

if __name__ == "__main__":
    main()
