# ============================================
# challenger_full_all_in_one.py (선그래프 포함 최종본)
# ============================================

import json
import numpy as np
import pickle
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from tensorflow.keras.models import load_model

from feature_builder_challenger_full import FeatureBuilderChallengerFull

# -----------------------------------------------------
# 한글 폰트 설정
# -----------------------------------------------------
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
matplotlib.rc("font", family=font_name)
matplotlib.rcParams["axes.unicode_minus"] = False

# 저장 폴더
SAVE_DIR = "real_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# 라인 정보
LANES = ["TOP", "JUNGLE", "MID", "BOTTOM"]
LANE_KR = {"TOP": "탑", "JUNGLE": "정글", "MID": "미드", "BOTTOM": "바텀"}

FB = FeatureBuilderChallengerFull()


# -----------------------------------------------------
# 공통 함수
# -----------------------------------------------------
def save_fig(fig, filename):
    """이미지 저장 + 닫기"""
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"📁 Saved → {path}")
    plt.close(fig)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d[0] if isinstance(d, list) else d


def compute_contribution(values):
    vals = np.array(values)
    abs_sum = np.sum(np.abs(vals))
    if abs_sum == 0:
        return np.zeros_like(vals)
    return np.abs(vals) / abs_sum


# -----------------------------------------------------
# 15분 골드 격차 계산
# -----------------------------------------------------
def extract_15min_gold(match, timeline):
    frames = timeline["info"]["frames"]
    if len(frames) <= 15:
        return None, None

    frame = frames[15]["participantFrames"]
    lane_diff = {l: 0 for l in LANES}
    lane_gold = {l: {"ally": 0, "enemy": 0} for l in LANES}

    for p in match["info"]["participants"]:
        pos = p["teamPosition"]
        if pos == "MIDDLE": pos = "MID"
        if pos == "UTILITY": pos = "BOTTOM"
        if pos not in LANES:
            continue

        pid = str(p["participantId"])
        gold = frame[pid]["totalGold"]

        if p["teamId"] == 100:
            lane_gold[pos]["ally"] += gold
            lane_diff[pos] += gold
        else:
            lane_gold[pos]["enemy"] += gold
            lane_diff[pos] -= gold

    return lane_diff, lane_gold


# -----------------------------------------------------
# 실제 21~25분 골드 격차
# -----------------------------------------------------
def extract_real(match, timeline):
    frames = timeline["info"]["frames"]
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

    return {l: sums[l] / count for l in LANES}


# -----------------------------------------------------
# LSTM 예측
# -----------------------------------------------------
def predict_match(match_id):

    model = load_model("lstm_models/lstm_challenger_full.h5", compile=False)

    with open("scalers/x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open("scalers/y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    match = load_json(f"match_data/match_{match_id}.json")
    timeline = load_json(f"match_data/timeline_{match_id}.json")

    ts = FB.extract_timeseries(match, timeline)

    merged = np.concatenate(
        [ts["TOP"], ts["JUNGLE"], ts["MID"], ts["BOTTOM"]],
        axis=1
    )

    X = merged[np.newaxis, :, :]
    flat = X.reshape(-1, X.shape[2])
    X_scaled = x_scaler.transform(flat).reshape(X.shape)

    pred_scaled = model.predict(X_scaled, verbose=0)
    pred_real = y_scaler.inverse_transform(pred_scaled)[0]

    return pred_real, match, timeline


# -----------------------------------------------------
# 그래프들
# -----------------------------------------------------
def plot_contrib_15(contrib):
    fig = plt.figure(figsize=(8, 5))
    plt.bar([LANE_KR[l] for l in LANES], contrib)
    plt.title("15분 라인별 기여도")
    plt.grid(alpha=0.3)
    save_fig(fig, "images/15min_contribution.png")


def plot_future_contrib(pred, real):
    x = np.arange(4)
    fig = plt.figure(figsize=(8, 5))
    plt.bar(x - 0.15, pred, width=0.3, label="예측")
    plt.bar(x + 0.15, real, width=0.3, label="실제")
    plt.xticks(x, [LANE_KR[l] for l in LANES])
    plt.title("21~25분 라인별 기여도 (예측 vs 실제)")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(fig, "images/future_contribution.png")


def plot_line_contrib(contrib15, pred, real):
    """🔥 선그래프: 15분 → 예측 → 실제 기여도 변화"""
    fig = plt.figure(figsize=(10, 5))
    x = np.arange(4)

    plt.plot(x, contrib15, marker="o", label="15분 기여도")
    plt.plot(x, pred, marker="o", label="예측 기여도")
    plt.plot(x, real, marker="o", label="실제 기여도")

    plt.xticks(x, [LANE_KR[l] for l in LANES])
    plt.ylim(0, 1)
    plt.title("기여도 변화 선그래프 (15분 → 예측 → 실제)")
    plt.grid(alpha=0.3)
    plt.legend()

    save_fig(fig, "images/line_contribution.png")


def plot_line_gold(pred_real, real_vals):
    """🔥 선그래프: 골드 격차 예측 vs 실제"""
    fig = plt.figure(figsize=(10, 5))
    x = np.arange(4)

    plt.plot(x, pred_real, marker="o", label="예측 골드 격차")
    plt.plot(x, real_vals, marker="o", label="실제 골드 격차")

    plt.xticks(x, [LANE_KR[l] for l in LANES])
    plt.title("골드 격차 선그래프 (예측 vs 실제)")
    plt.grid(alpha=0.3)
    plt.legend()

    save_fig(fig, "images/line_gold_diff.png")


def plot_bar(pred_real, real):
    x = np.arange(4)
    true_vals = [real[l] for l in LANES]

    fig = plt.figure(figsize=(9, 5))
    plt.bar(x - 0.15, pred_real, width=0.3, label="예측")
    plt.bar(x + 0.15, true_vals, width=0.3, label="실제")
    plt.xticks(x, [LANE_KR[l] for l in LANES])
    plt.title("21~25분 골드 격차 (예측 vs 실제)")
    plt.legend()
    plt.grid(alpha=0.3)

    save_fig(fig, "images/bar_pred_vs_real.png")


def plot_scatter(pred_real, real):
    true_vals = [real[l] for l in LANES]

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_real, s=90, color="green")
    mn = min(true_vals + list(pred_real))
    mx = max(true_vals + list(pred_real))
    plt.plot([mn, mx], [mn, mx], "--", color="gray")
    plt.xlabel("실제")
    plt.ylabel("예측")
    plt.grid(alpha=0.3)

    save_fig(fig, "images/scatter_pred_vs_real.png")


def plot_radar(pred_real, real_vals):

    labels = [LANE_KR[l] for l in LANES]
    stats_pred = list(pred_real)
    stats_real = list(real_vals)

    angles = np.linspace(0, 2 * np.pi, len(LANES), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    stats_pred = np.concatenate((stats_pred, [stats_pred[0]]))
    stats_real = np.concatenate((stats_real, [stats_real[0]]))

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, stats_pred, "r-", linewidth=2, label="예측")
    ax.fill(angles, stats_pred, "r", alpha=0.25)

    ax.plot(angles, stats_real, "b-", linewidth=2, label="실제")
    ax.fill(angles, stats_real, "b", alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("골드 격차 Radar Chart")
    ax.legend()

    save_fig(fig, "images/radar_chart.png")


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main(match_id):

    pred_real, match, timeline = predict_match(match_id)
    real = extract_real(match, timeline)

    # ---- 15분 기여도 ----
    diff15, _ = extract_15min_gold(match, timeline)
    diff_vals = [diff15[l] for l in LANES]
    contrib15 = compute_contribution(diff_vals)

    print("\n===== 15분 기여도 =====")
    for i, lane in enumerate(LANES):
        print(f"{LANE_KR[lane]}: {contrib15[i]*100:.1f}%")
    plot_contrib_15(contrib15)

    # ---- 미래 기여도 ----
    pred_vals = pred_real
    real_vals = [real[l] for l in LANES]

    contrib_pred = compute_contribution(pred_vals)
    contrib_real = compute_contribution(real_vals)

    print("\n===== 미래 기여도 (예측) =====")
    for i, lane in enumerate(LANES):
        print(f"{LANE_KR[lane]}: {contrib_pred[i]*100:.1f}%")

    print("\n===== 미래 기여도 (실제) =====")
    for i, lane in enumerate(LANES):
        print(f"{LANE_KR[lane]}: {contrib_real[i]*100:.1f}%")

    plot_future_contrib(contrib_pred, contrib_real)

    # ---- 신규 추가: 선그래프 ----
    plot_line_contrib(contrib15, contrib_pred, contrib_real)
    plot_line_gold(pred_real, real_vals)

    # ---- 기존 그래프 ----
    plot_bar(pred_real, real)
    plot_scatter(pred_real, real)
    plot_radar(pred_real, real_vals)

    print("\n🎉 모든 분석 완료! (real_final 폴더 확인하세요)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", type=int, default=1)
    args = parser.parse_args()
    main(args.match)
