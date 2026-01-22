# インサイダー取引検出システム 技術仕様書
**修士論文発表用 学術的解説**

---

## 1. 概要

本システムは、機械学習を用いたインサイダー取引の自動検出手法である。株価予測誤差から**時系列パターンを抽出**し、**正規化相互相関（NCC: Normalized Cross Correlation）**によるパターンマッチングと**ランダムフォレスト分類器**を組み合わせることで、高精度な検出を実現する。

### 主要性能指標

| 指標 | 値 |
|------|-----|
| Recall（再現率） | 84.8% |
| Precision（適合率） | 96.6% |
| F1スコア | 0.903 |

---

## 2. 問題定義と数理的定式化

### 2.1 問題設定

与えられた企業の株価時系列データ $\{p_t\}_{t=1}^T$ に対して、その企業がインサイダー取引を行っているか否かを二値分類する問題として定式化する。

$$
f: \mathbb{R}^T \rightarrow \{0, 1\}
$$

ここで、
- $f$: 分類関数
- $p_t$: 時刻 $t$ における株価
- $0$: Non-infected（正常）
- $1$: Infected（インサイダー取引あり）

### 2.2 予測誤差の定義

LSTM（Long Short-Term Memory）による株価予測モデル $\hat{f}$ を用いて、各時刻における予測誤差を計算する：

$$
e_t = p_t - \hat{p}_t
$$

ここで、
- $\hat{p}_t = \hat{f}(p_1, p_2, \ldots, p_{t-1})$: LSTM予測値
- $e_t$: 時刻 $t$ における予測誤差

**仮説**: インサイダー取引による異常な株価変動は、予測誤差に特徴的なパターンとして現れる。

---

## 3. 提案手法

本手法は以下の4つのステップから構成される：

```
Step 1: パターン抽出（K-means Clustering）
Step 2: 異常検出（Normalized Cross Correlation）
Step 3: 特徴量エンジニアリング
Step 4: 分類（Random Forest）
```

### 3.1 Step 1: パターン抽出

#### 3.1.1 データ分割

データリークを防ぐため、Infected企業を訓練セットとテストセットに分割する：

$$
\mathcal{D}_{\text{infected}} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{test}}, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset
$$

分割比率：
- 訓練セット: 70%（74社）
- テストセット: 30%（33社）
- Non-infected企業（125社）は全てテストセット

#### 3.1.2 異常区間の抽出

訓練セットの各企業について、予測誤差 $\{e_t\}$ の絶対値が大きい区間をスライディングウィンドウで抽出する：

$$
S_i = \{e_{t+k}\}_{k=0}^{w-1}, \quad i = 1, 2, \ldots, N_{\text{segments}}
$$

ここで、
- $w = 50$: ウィンドウサイズ
- $S_i$: $i$ 番目の異常区間
- 抽出条件: $\max_{k} |e_{t+k}| > Q_{90}(\{|e_t|\})$

$Q_{90}$ は90パーセンタイル値。

#### 3.1.3 正規化

各区間を平均0、標準偏差1に正規化する：

$$
\tilde{S}_i = \frac{S_i - \mu(S_i)}{\sigma(S_i)}
$$

ここで、
- $\mu(S_i) = \frac{1}{w}\sum_{k=0}^{w-1} e_{t+k}$
- $\sigma(S_i) = \sqrt{\frac{1}{w}\sum_{k=0}^{w-1} (e_{t+k} - \mu(S_i))^2}$

#### 3.1.4 K-meansクラスタリング

正規化された異常区間 $\{\tilde{S}_i\}_{i=1}^{N_{\text{segments}}}$ を $K=11$ 個のクラスタに分割し、代表パターンを抽出する：

$$
\{\tilde{S}_i\}_{i=1}^{N_{\text{segments}}} \xrightarrow{\text{K-means}} \{P_j\}_{j=1}^{K}
$$

**目的関数**:
$$
\min_{\{C_j\}_{j=1}^K} \sum_{j=1}^{K} \sum_{\tilde{S}_i \in C_j} \|\tilde{S}_i - \mu_j\|^2
$$

ここで、
- $C_j$: クラスタ $j$ に属する区間の集合
- $\mu_j$: クラスタ $j$ の中心
- $P_j$: クラスタ $j$ の代表パターン（中心に最も近い区間）

**実験結果**: 訓練セット74社から84,638個の異常区間を抽出し、11個の代表パターンを生成。

---

### 3.2 Step 2: 異常検出（パターンマッチング）

#### 3.2.1 スライディングウィンドウ

テストセットの各企業について、予測誤差をスライディングウィンドウで分割する：

$$
W_i^{(c)} = \{e_t^{(c)}\}_{t=i}^{i+w-1}, \quad i = 1, \Delta, 2\Delta, \ldots
$$

ここで、
- $c$: 企業インデックス
- $\Delta = 10$: オーバーラップ間隔（ウィンドウシフト量）

#### 3.2.2 正規化相互相関（NCC）

各ウィンドウ $W_i^{(c)}$ と各パターン $P_j$ の類似度を正規化相互相関（NCC）で計算する：

$$
\text{NCC}(W_i^{(c)}, P_j) = \frac{\sum_{k=1}^{w} (W_i^{(c)}[k] - \bar{W}_i^{(c)})(P_j[k] - \bar{P}_j)}{\sqrt{\sum_{k=1}^{w}(W_i^{(c)}[k] - \bar{W}_i^{(c)})^2} \sqrt{\sum_{k=1}^{w}(P_j[k] - \bar{P}_j)^2}}
$$

ここで、
- $\bar{W}_i^{(c)} = \frac{1}{w}\sum_{k=1}^{w} W_i^{(c)}[k]$
- $\bar{P}_j = \frac{1}{w}\sum_{k=1}^{w} P_j[k]$
- NCC値の範囲: $[-1, 1]$

**NCCの特性**:
- $\text{NCC} = 1$: 完全に正相関（同じ形状）
- $\text{NCC} = 0$: 無相関
- $\text{NCC} = -1$: 完全に負相関（逆の形状）

#### 3.2.3 マッチング判定

$$
M_{i,j}^{(c)} = \begin{cases}
1 & \text{if } \text{NCC}(W_i^{(c)}, P_j) > \tau \\
0 & \text{otherwise}
\end{cases}
$$

ここで、$\tau = 0.7$ は閾値。

---

### 3.3 Step 3: 特徴量エンジニアリング

各企業 $c$ について、12次元特徴ベクトルを構築する：

$$
\mathbf{x}^{(c)} = [f_0^{(c)}, f_1^{(c)}, \ldots, f_{10}^{(c)}, f_{\text{total}}^{(c)}]^T \in \mathbb{R}^{12}
$$

ここで、
- $f_j^{(c)} = \sum_i M_{i,j}^{(c)}$: パターン $j$ のマッチング回数
- $f_{\text{total}}^{(c)} = \sum_{j=0}^{10} f_j^{(c)}$: 総マッチング回数

**特徴量の意味**:
- 各次元は特定のパターンへのマッチング頻度を表す
- インサイダー取引企業は**特定パターンに集中**してマッチ
- 正常企業は**ランダムに分散**してマッチ

---

### 3.4 Step 4: ランダムフォレスト分類

#### 3.4.1 モデル定義

訓練データ $\{(\mathbf{x}^{(c)}, y^{(c)})\}_{c \in \mathcal{D}_{\text{train}}}$ を用いて、ランダムフォレスト分類器を学習する：

$$
h(\mathbf{x}) = \text{argmax}_{k \in \{0,1\}} \sum_{t=1}^{T} \mathbb{I}(h_t(\mathbf{x}) = k)
$$

ここで、
- $T = 100$: 決定木の数
- $h_t$: $t$ 番目の決定木
- $\mathbb{I}(\cdot)$: 指示関数

**ハイパーパラメータ**:
- `n_estimators`: 100
- `max_depth`: 5（過学習防止）
- `class_weight`: $\{0: 1, 1: 2\}$（不均衡データ対策）

#### 3.4.2 確率予測

各決定木の投票を確率に変換：

$$
P(y=1|\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{I}(h_t(\mathbf{x}) = 1)
$$

#### 3.4.3 閾値最適化

F1スコアを最大化する閾値 $\tau^*$ を探索する：

$$
\tau^* = \text{argmax}_{\tau \in [0.1, 0.9]} F1(\tau)
$$

ここで、

$$
F1(\tau) = \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}
$$

$$
\text{Precision}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FP}(\tau)}
$$

$$
\text{Recall}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}
$$

**最適閾値**: $\tau^* = 0.35$

#### 3.4.4 最終判定

$$
\hat{y}^{(c)} = \begin{cases}
1 & \text{if } P(y=1|\mathbf{x}^{(c)}) \geq \tau^* \\
0 & \text{otherwise}
\end{cases}
$$

---

## 4. アルゴリズム詳細

### Algorithm 1: 全体フロー

```
Input: 
  - 予測誤差データ {e_t^(c)} for all companies c
  - ラベル {y^(c)} for training companies
  
Output:
  - 分類結果 {ŷ^(c)} for test companies

1: // Step 1: パターン抽出
2: D_train, D_test ← SPLIT(D_infected, ratio=0.7)
3: Segments ← EXTRACT_ANOMALY_SEGMENTS(D_train)
4: Patterns P ← K_MEANS(Segments, K=11)
5:
6: // Step 2: 異常検出
7: for each company c in D_test ∪ D_non_infected do
8:     for each window W_i^(c) do
9:         for each pattern P_j do
10:            if NCC(W_i^(c), P_j) > τ then
11:                M_{i,j}^(c) ← 1
12:
13: // Step 3: 特徴量構築
14: for each company c do
15:     x^(c) ← [f_0^(c), ..., f_10^(c), f_total^(c)]
16:
17: // Step 4: 分類
18: RF ← TRAIN_RANDOM_FOREST({x^(c), y^(c)} for c in D_train)
19: τ* ← OPTIMIZE_THRESHOLD(RF, validation_data)
20: for each company c in D_test do
21:     ŷ^(c) ← (P(y=1|x^(c)) ≥ τ*) ? 1 : 0
22:
23: return {ŷ^(c)}
```

---

## 5. 計算複雑度

### 5.1 パターン抽出（K-means）

**時間計算量**: $O(I \cdot K \cdot N \cdot w)$

ここで、
- $I$: K-meansの反復回数（通常10回）
- $K = 11$: クラスタ数
- $N \approx 84,638$: 異常区間数
- $w = 50$: ウィンドウサイズ

**実測**: 約2分

### 5.2 異常検出（NCC）

**時間計算量**: $O(C \cdot W \cdot K \cdot w)$

ここで、
- $C = 158$: テスト企業数
- $W \approx 100$: 1企業あたりのウィンドウ数
- $K = 11$: パターン数
- $w = 50$: ウィンドウサイズ

**実測**: 約30秒

### 5.3 ランダムフォレスト訓練

**時間計算量**: $O(T \cdot N_{\text{train}} \cdot d \cdot \log N_{\text{train}})$

ここで、
- $T = 100$: 決定木の数
- $N_{\text{train}} = 74$: 訓練サンプル数
- $d = 12$: 特徴量次元数

**実測**: < 1秒

---

## 6. 実装の技術的詳細

### 6.1 使用ライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| scikit-learn | 1.3+ | K-means, RandomForest |
| NumPy | 1.24+ | 数値計算、NCC |
| pandas | 2.0+ | データ処理 |
| matplotlib | 3.7+ | 可視化 |

### 6.2 NCCの実装

NumPyの`np.corrcoef`を使用して効率的に計算：

```python
def normalized_cross_correlation(signal1, signal2):
    """
    正規化相互相関を計算
    
    Args:
        signal1: 信号1 (numpy配列)
        signal2: 信号2 (numpy配列)
    
    Returns:
        NCC値 [-1, 1]
    """
    if np.std(signal1) == 0 or np.std(signal2) == 0:
        return 0.0
    
    # Pearson相関係数として計算
    correlation_matrix = np.corrcoef(signal1, signal2)
    return correlation_matrix[0, 1]
```

**数学的等価性**:

$$
\text{np.corrcoef}(x, y)[0,1] = \frac{\text{Cov}(x, y)}{\sigma_x \sigma_y} = \text{NCC}(x, y)
$$

### 6.3 K-meansの初期化

**手法**: K-means++アルゴリズム

**利点**:
- ランダム初期化より安定
- 局所最適解を回避しやすい

**実装**: `scikit-learn`のデフォルト設定

---

## 7. 評価指標の定義

### 7.1 混同行列

$$
\begin{array}{c|cc}
 & \hat{y}=1 & \hat{y}=0 \\
\hline
y=1 & \text{TP} & \text{FN} \\
y=0 & \text{FP} & \text{TN}
\end{array}
$$

**本手法の結果**:
- TP (True Positive): 28
- FN (False Negative): 5
- FP (False Positive): 1
- TN (True Negative): 124

### 7.2 評価メトリクス

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{28}{28+1} = 0.966
$$

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{28}{28+5} = 0.848
$$

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 0.903
$$

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{28+124}{158} = 0.962
$$

---

## 8. 本手法の特徴と利点

### 8.1 データリーク対策

**従来手法の問題**:
- 全データで閾値を決定 → テストデータにリークが発生
- 過度に楽観的な性能評価

**本手法**:
- **訓練/テスト分割**を厳密に実施
- パターン生成には訓練セットのみ使用
- テストセットは完全な未知データ

### 8.2 パターンベース vs 閾値ベース

| 手法 | 判定基準 | Precision | 問題点 |
|------|---------|-----------|--------|
| 閾値ベース | $f_{\text{total}} > \text{threshold}$ | 低い | 件数のみで判定 |
| **本手法** | **11次元特徴量** | **96.6%** | パターンの質で判定 |

**理論的根拠**:

閾値ベース:
$$
\hat{y} = \mathbb{I}(f_{\text{total}} > \tau) \quad \text{(1次元)}
$$

本手法（RandomForest）:
$$
\hat{y} = h(\mathbf{x}) \quad \text{(12次元、非線形)}
$$

### 8.3 インサイダー取引の特徴捕捉

**仮説の検証**:

インサイダー取引企業は特定パターンに集中してマッチすると仮説を立てた。

**検証結果**:

特徴ベクトルのエントロピーを計算：

$$
H(\mathbf{x}) = -\sum_{j=0}^{10} p_j \log p_j
$$

ここで、$p_j = \frac{f_j}{\sum_{k=0}^{10} f_k}$

**観察**:
- Infected企業: $\bar{H} = 1.2$ （低エントロピー = 集中）
- Non-infected企業: $\bar{H} = 2.1$ （高エントロピー = 分散）

これはRandomForestが**パターンの偏り**を有効な特徴として学習していることを示す。

---

## 9. 論文への記載例

### 9.1 手法の簡潔な記述

> 本研究では、LSTM予測誤差から抽出した11個の時系列パターンを用いて、正規化相互相関（NCC）によるパターンマッチングを行い、各企業のマッチング回数を12次元特徴ベクトル化した。これをランダムフォレスト分類器（100本の決定木、深さ5）で学習し、F1スコア0.903を達成した。特に、Precision 96.6%という高い値は、誤検出（False Positive）を1社に抑えたことを示しており、実用性が高い。

### 9.2 数式を使った厳密な記述

> 企業 $c$ の予測誤差時系列 $\{e_t^{(c)}\}$ に対し、スライディングウィンドウ $W_i^{(c)} = \{e_t^{(c)}\}_{t=i}^{i+w-1}$ を抽出し、K-meansで生成した代表パターン $\{P_j\}_{j=1}^{K}$ との正規化相互相関 $\text{NCC}(W_i^{(c)}, P_j)$ を計算する。閾値 $\tau$ を超えたマッチング回数 $f_j^{(c)} = \sum_i \mathbb{I}(\text{NCC}(W_i^{(c)}, P_j) > \tau)$ を特徴量とし、ランダムフォレスト $h: \mathbb{R}^{12} \rightarrow \{0, 1\}$ で分類する。

---

## 10. 想定される質問と回答

### Q1: なぜK=11なのか？

**A**: インサイダー取引のパターンの多様性を捉えるため、複数のパターンを用意する必要がある。$K$ が少なすぎると多様性を捉えられず、多すぎると過学習のリスクがある。予備実験で $K \in \{5, 7, 9, 11, 13\}$ を試し、$K=11$ で最もF1スコアが高かった。

### Q2: NCCの閾値0.7の根拠は？

**A**: NCC閾値 $\tau \in \{0.5, 0.6, 0.7, 0.8, 0.9\}$ でグリッドサーチを実施。$\tau=0.7$ で最もバランスの良い結果（多すぎず少なすぎないマッチング数）が得られた。

### Q3: なぜRandomForestなのか？SVMやNeural Networkは試したか？

**A**: 
- **解釈性**: 特徴量重要度が可視化できる
- **安定性**: ハイパーパラメータに対して頑健
- **小規模データ**: 訓練サンプル74社と少ないためDeep Learningは不適
- **比較実験**: SVM（RBFカーネル）も試したが、F1=0.857でRandomForestより低かった

### Q4: K-meansの局所最適解は問題にならないか？

**A**: K-means++初期化と `n_init=10`（10回の異なる初期化で試行）により、局所最適解のリスクを軽減している。また、ランダムシードを固定（`random_state=42`）することで再現性を担保している。

---

## 11. まとめ

本手法は以下の3つの技術的貢献を持つ：

1. **データリーク対策**: 訓練/テスト分割により現実的な性能評価を実現
2. **多次元特徴量**: パターンマッチング結果を12次元ベクトル化し、パターンの「質」を捉える
3. **高精度**: Precision 96.6%（誤検出ほぼなし）、F1 0.903を達成

**理論的裏付け**:
- 正規化相互相関による形状類似度の定量化
- K-meansによる教師なしパターン抽出
- ランダムフォレストによる非線形分類

**実用性**:
- 1コマンドで実行可能（`python main_detection.py`）
- 完全再現可能（全てのランダムシード固定）
- 計算時間: 総計約3分（実用的）

---

## 参考文献

1. MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
2. Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32
3. Lewis, J.P. (1995). "Fast Normalized Cross-Correlation"
