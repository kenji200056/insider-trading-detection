# 代替手法の混同行列 生成方法ドキュメント

本ドキュメントでは、比較実験で使用した3つの代替手法について、混同行列がどのように生成されたかを説明する。

---

## 1. 統計的閾値法

### 概要
最もシンプルなベースライン手法。機械学習を使用せず、予測誤差が統計的閾値を超えた場合に異常と判定する。

### 生成手順

```python
# 1. 各企業の予測誤差データを読み込む
errors = 実測値 - 予測値  # LSTM出力との差分

# 2. 企業ごとに閾値を計算
threshold = mean(|errors|) + 2.0 × std(|errors|)

# 3. 閾値を超える異常回数をカウント
anomaly_count = sum(|errors| > threshold)

# 4. 異常率で最終判定
if (anomaly_count / len(errors)) > 0.10:
    判定 = "Infected"
else:
    判定 = "Non-infected"
```

### パラメータ
| パラメータ | 値 | 説明 |
|-----------|------|------|
| sigma | 2.0 | 標準偏差の倍率 |
| 異常率閾値 | 10% | この割合を超えたら異常判定 |

### 結果
```
TP=0, FN=33, FP=0, TN=125
Recall: 0.0%, Precision: 0.0%, F1: 0.000
```

### 考察
- 全てのInfected企業を見逃している（FN=33）
- 企業ごとに閾値を計算するため、各企業が自身の「正常範囲」内に収まる
- インサイダー取引のパターンは単純な統計的逸脱では捉えられない

---

## 2. 単純ML分類（パターン抽出なし）

### 概要
予測誤差から統計的特徴量を抽出し、Random Forestで直接分類する。本研究のようなパターンクラスタリングは行わない。

### 生成手順

```python
# 1. 各企業の予測誤差から統計的特徴量を抽出
def extract_simple_features(errors):
    return [
        mean(errors),           # 平均
        std(errors),            # 標準偏差
        max(|errors|),          # 最大絶対値
        min(errors),            # 最小値
        max(errors),            # 最大値
        percentile(|errors|, 90),  # 90パーセンタイル
        percentile(|errors|, 95),  # 95パーセンタイル
        sum(|errors| > 2σ),     # 2σ超え回数
        sum(|errors| > 3σ),     # 3σ超え回数
        len(errors)             # データ長
    ]

# 2. 訓練データでモデル学習
#    - 訓練Infected企業（70%）
#    - Non-infected企業の70%
X_train = [extract_simple_features(e) for e in 訓練企業]
clf = RandomForestClassifier(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)

# 3. テストデータで予測
y_pred = clf.predict(X_test)
```

### パラメータ
| パラメータ | 値 | 説明 |
|-----------|------|------|
| n_estimators | 100 | 決定木の数 |
| max_depth | 5 | 木の深さ |
| 特徴量数 | 10 | 統計的特徴量 |

### 結果
```
TP=24, FN=9, FP=21, TN=17
Recall: 72.7%, Precision: 53.3%, F1: 0.615
```

### 考察
- 統計的閾値法より大幅に改善（F1: 0.000 → 0.615）
- ただし誤検出（FP=21）が多い
- 時系列パターンを考慮していないため、「インサイダー特有の動き」を捕捉できない

---

## 3. LSTMのみ（予測誤差閾値法）

### 概要
LSTMの予測誤差を使用するが、パターンマッチングは行わない。単純に予測誤差の大きさだけで異常判定する。

### 生成手順

```python
# 1. 訓練Infected企業の予測誤差から閾値を決定
all_train_errors = []
for company in train_infected:
    errors = |実測値 - 予測値|
    all_train_errors.extend(errors)

# 2. 90パーセンタイルを閾値として設定
threshold = percentile(all_train_errors, 90)
# 実測値: 4.4467

# 3. テスト企業ごとに判定
for company in test_companies:
    errors = |実測値 - 予測値|
    exceed_ratio = sum(errors > threshold) / len(errors)
    
    if exceed_ratio > 0.05:  # 5%以上が閾値を超えたら異常
        判定 = "Infected"
    else:
        判定 = "Non-infected"
```

### パラメータ
| パラメータ | 値 | 説明 |
|-----------|------|------|
| percentile | 90 | 閾値決定用パーセンタイル |
| exceed_ratio閾値 | 5% | この割合を超えたら異常判定 |
| 計算された閾値 | 4.4467 | 予測誤差の絶対値 |

### 結果
```
TP=20, FN=13, FP=83, TN=42
Recall: 60.6%, Precision: 19.4%, F1: 0.294
```

### 考察
- Recallは60.6%とそこそこだが、Precisionが19.4%と極めて低い
- 偽陽性（FP=83）が非常に多い
- 「予測誤差が大きい」だけでは、市場全体の変動や決算発表なども拾ってしまう
- パターンマッチングなしでは「インサイダー的な動き」と「正常な大変動」を区別できない

---

## 手法比較サマリ

| 手法 | TP | FN | FP | TN | Recall | Precision | F1 |
|------|---:|---:|---:|---:|-------:|----------:|---:|
| 統計的閾値法 | 0 | 33 | 0 | 125 | 0.0% | 0.0% | 0.000 |
| 単純ML分類 | 24 | 9 | 21 | 17 | 72.7% | 53.3% | 0.615 |
| LSTMのみ（閾値） | 20 | 13 | 83 | 42 | 60.6% | 19.4% | 0.294 |

### 結論

- **統計的閾値法**: 文脈を無視した単純な逸脱検出では、インサイダー取引は検出不可能
- **単純ML分類**: 機械学習を導入することで改善するが、時系列パターンの欠如が精度の上限となる
- **LSTMのみ**: LSTMは正常パターンを学習できるが、「異常の種類」を区別できず偽陽性が多発

これらの限界を克服するために、本研究では**パターンクラスタリング**と**正規化相互相関による類似度特徴量**を導入した。
