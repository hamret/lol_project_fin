# main_challenger_full.py
import os
import json
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from feature_builder_challenger_full import FeatureBuilderChallengerFull


DATA_DIR = "match_data"
LANES = ["TOP", "JUNGLE", "MID", "BOTTOM"]
FB = FeatureBuilderChallengerFull()


# ============================================
# R² metric
# ============================================
def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


# ============================================
# JSON Loader (Safe)
# ============================================
def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except:
        return None

    return d[0] if isinstance(d, list) and len(d) > 0 else d if isinstance(d, dict) else None


# ============================================
# Y label: 21~25분 평균 골드 격차 (4개 라벨)
# ============================================
def extract_y_21_25(match, timeline):
    frames = timeline["info"]["frames"]

    if len(frames) <= 25:
        return None

    minutes = [21, 22, 23, 24, 25]
    sums = {l: 0 for l in LANES}
    count = 0

    for minute in minutes:
        pf = frames[minute]["participantFrames"]
        lane_diff = {l: 0 for l in LANES}

        for p in match["info"]["participants"]:
            pos = p["teamPosition"]
            if pos == "MIDDLE": pos = "MID"
            if pos == "UTILITY": pos = "BOTTOM"
            if pos not in LANES:
                continue

            pid = str(p["participantId"])
            gold = pf[pid]["totalGold"]

            if p["teamId"] == 100:
                lane_diff[pos] += gold
            else:
                lane_diff[pos] -= gold

        for l in LANES:
            sums[l] += lane_diff[l]
        count += 1

    return [sums[l] / count for l in LANES]


# ============================================
# MAIN TRAINING LOOP
# ============================================
def main():
    X = []
    Y = []

    print("📂 Building CHALLENGER dataset... (SUPER features enabled)")

    for i in tqdm(range(1, 1088)):
        m = load_json(f"{DATA_DIR}/match_{i}.json")
        t = load_json(f"{DATA_DIR}/timeline_{i}.json")

        if m is None or t is None:
            continue

        ts = FB.extract_timeseries(m, t)
        if ts["TOP"].shape[0] < 16:
            continue

        y = extract_y_21_25(m, t)
        if y is None:
            continue

        # 하나의 거대한 feature 벡터로 합치기
        merged = np.concatenate(
            [ts["TOP"], ts["JUNGLE"], ts["MID"], ts["BOTTOM"]], axis=1
        )  # shape: (16, F_total)

        X.append(merged)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    print(f"Samples: {len(X)}")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # ============================================
    # Scaling
    # ============================================
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_flat = X.reshape(-1, X.shape[2])
    X_scaled = x_scaler.fit_transform(X_flat).reshape(X.shape)
    Y_scaled = y_scaler.fit_transform(Y)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=0.15, shuffle=True, random_state=42
    )

    # ============================================
    # CHALLENGER LSTM 모델 구조
    # ============================================
    seq_len = 16
    feat_dim = X.shape[2]

    model = Sequential([
        LSTM(192, return_sequences=True, input_shape=(seq_len, feat_dim)),
        LayerNormalization(),

        LSTM(128, return_sequences=True),
        Dropout(0.25),
        LayerNormalization(),

        LSTM(96, return_sequences=False),
        Dropout(0.25),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(4)  # lane diffs
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae", r2_score]
    )

    es = EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )

    print("\n🔥 Training **CHALLENGER SUPER LSTM**...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=180,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    # ============================================
    # 성능 평가
    # ============================================
    y_pred_test = model.predict(X_test)
    y_pred_real = y_scaler.inverse_transform(y_pred_test)
    y_true_real = y_scaler.inverse_transform(Y_test)

    mae = np.mean(np.abs(y_true_real - y_pred_real))
    print("\n📊 Final Test Results (Real Scale)")
    print(f"MAE: {mae:.2f}")

    # ============================================
    # 모델 저장
    # ============================================
    os.makedirs("lstm_models", exist_ok=True)
    model.save("lstm_models/lstm_challenger_full.h5")
    print("💾 Saved model → lstm_models/lstm_challenger_full.h5")

    # 스케일러 저장
    import pickle
    os.makedirs("scalers", exist_ok=True)

    with open("scalers/x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)

    with open("scalers/y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    print("💾 Saved scalers → scalers/x_scaler.pkl, y_scaler.pkl")


if __name__ == "__main__":
    main()
