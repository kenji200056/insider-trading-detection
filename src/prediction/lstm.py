# Jakob Aungiers によるコードテンプレート
# 修正者: Sheikh Rabiul Islam
# 日付: 2017/11/10
# 目的: 株式市場のボラティリティ予測（LSTM）
import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # TensorFlowの警告を非表示にする
warnings.filterwarnings("ignore") # Numpyの警告を非表示にする

def load_data(filename, seq_len, normalise_window):
    """
    ファイルを読み込み、LSTM用の学習/テストデータを作成する
    """
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    # 空行や不正な値を除外（大量データ処理時の安定性向上）
    cleaned_data = []
    for x in data:
        x = x.strip()
        if x:
            try:
                float(x)
                cleaned_data.append(x)
            except ValueError:
                # 数値に変換できない行はスキップ
                continue
    data = cleaned_data

    sequence_length = seq_len + 1
    
    result = []
    # スライディングウィンドウ（1日ずつずらしてデータを塊にする）の作成
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    # 正規化（割合への変換）が有効な場合
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    # データを分割（この実装では全データを学習とテストの両方に使用する設定になっている）
    row = round(1 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]  # 入力（過去50日分）
    y_train = train[:, -1]   # 正解（51日目の値）
    x_test = result[:, :-1]
    y_test = result[:, -1]
    
    # LSTMの入力形式 [サンプル数, タイムステップ, 特徴量数] に整形
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    """
    データの正規化。各ウィンドウの最初の値を基準とした変化率に変換する
    式: (当日 / ウィンドウ初日の値) - 1
    """
    normalised_data = []
    for window in window_data:
        try:
            p_0 = float(window[0])
            if p_0 == 0:
                # 基準が0の場合は全て0にする（あるいはスキップする）
                normalised_window = [0.0 for p in window]
            else:
                normalised_window = [((float(p) / p_0) - 1) for p in window]
            normalised_data.append(normalised_window)
        except ZeroDivisionError:
            normalised_data.append([0.0 for p in window])
            
    return normalised_data

def denormalise_windows(window_data):
    denormalised_data = []
    for window in window_data:
        denormalised_window = [(float(window[0]) * (float(p) + 1)) for p in window]
        denormalised_data.append(denormalised_window)
    return denormalised_data


def build_model(layers):
    """
    LSTMモデルを構築する
    layers: [入力特徴量数, 中間層1のユニット数, 中間層2のユニット数, 出力層のユニット数]
    """
    model = Sequential()

    # 第1層: LSTM
    # Keras 2.x/TF 2.x 対応: output_dim -> units
    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        units=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2)) # 過学習防止のためのドロップアウト

    # 第2層: LSTM
    model.add(LSTM(
        units=layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    # 第3層: 全結合層
    model.add(Dense(
        units=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    # 損失関数として MSE (平均二乗誤差)、最適化アルゴリズムとして RMSprop を使用
    model.compile(loss="mse", optimizer="rmsprop")
    print("> モデル構築完了。コンパイル時間 : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    """
    1ステップ（1日）ずつ予測を行う
    """
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    """
    シークエンス全体を予測する（自身の予測値を次の入力として使い続ける）
    非常に難易度が高く、誤差が蓄積しやすい
    """
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    """
    一定の長さ（prediction_len）の予測を繰り返し行う
    50日分の予測を行ったら、次の50日へ窓をずらして再予測する
    """
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs