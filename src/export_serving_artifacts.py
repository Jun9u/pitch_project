from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch


# =========================
# 경로 설정
# =========================

LSTM_CHECKPOINT_PATH = Path("models/lstm_weighted_macro_best.pt")
LSTM_SERVING_PATH = Path("models/lstm_serving_artifact.pt")

OUTCOME_PATH = Path("data/processed/outcome_sequence_data.csv")
OUTCOME_ARTIFACT_PATH = Path("models/outcome_ranker_artifact.pkl")

SEQUENCE_PATH = Path("data/processed/sequence_data.csv")

NUMERIC_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "runner_1b",
    "runner_2b",
    "runner_3b",
]

TARGET_COL = "delta_run_exp"

ALPHA = 50.0

MIN_COUNT_PITCHER_COUNT_ACTION_ZONE = 5
MIN_COUNT_PITCHER_ACTION_ZONE = 10
MIN_COUNT_COUNT_ACTION_ZONE = 20
MIN_COUNT_ACTION_ZONE = 30
MIN_COUNT_ACTION = 50


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_key_value(x):
    if pd.isna(x):
        return None

    if isinstance(x, (np.integer, int)):
        return int(x)

    if isinstance(x, (np.floating, float)):
        if float(x).is_integer():
            return int(x)
        return float(x)

    return str(x)


def make_key(row, cols):
    return tuple(normalize_key_value(row[c]) for c in cols)


def build_stat_dict(train_df, group_cols):
    stats = (
        train_df
        .groupby(group_cols, dropna=False)[TARGET_COL]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )

    stat_dict = {}

    for _, row in stats.iterrows():
        key = make_key(row, group_cols)

        stat_dict[key] = {
            "sum": float(row["sum"]),
            "count": int(row["count"]),
            "mean": float(row["mean"]),
        }

    return stat_dict


def zone_sort_key(z):
    try:
        return int(str(z).replace("Z", ""))
    except ValueError:
        return 999


def compute_scaler_from_sequence_data():
    if not SEQUENCE_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {SEQUENCE_PATH}")

    usecols = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
    ] + NUMERIC_COLS

    df = pd.read_csv(SEQUENCE_PATH, usecols=usecols)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    train_end = int(len(df) * 0.7)
    train_df = df.iloc[:train_end].copy()

    numeric = train_df[NUMERIC_COLS].astype(float).values

    mean = numeric.mean(axis=0)
    scale = numeric.std(axis=0)

    # StandardScaler와 동일하게 0분산 방지
    scale[scale == 0] = 1.0

    return mean.astype(float).tolist(), scale.astype(float).tolist()


def export_lstm_serving_artifact():
    if not LSTM_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {LSTM_CHECKPOINT_PATH}")

    checkpoint = safe_torch_load(LSTM_CHECKPOINT_PATH)

    # 예전 checkpoint에는 scaler가 없으므로 sequence_data에서 복원
    if "scaler_mean" not in checkpoint or "scaler_scale" not in checkpoint:
        print("scaler_mean/scaler_scale이 checkpoint에 없습니다.")
        print("sequence_data.csv의 train 70% 구간에서 scaler 값을 재계산합니다.")

        scaler_mean, scaler_scale = compute_scaler_from_sequence_data()

        checkpoint["scaler_mean"] = scaler_mean
        checkpoint["scaler_scale"] = scaler_scale
        checkpoint["numeric_cols"] = NUMERIC_COLS

    required_keys = [
        "model_state_dict",
        "target_to_idx",
        "idx_to_target",
        "pitch_vocab",
        "zone_vocab",
        "pitcher_vocab",
        "stand_vocab",
        "throws_vocab",
        "numeric_cols",
        "scaler_mean",
        "scaler_scale",
    ]

    missing = [k for k in required_keys if k not in checkpoint]

    if missing:
        raise ValueError(f"LSTM checkpoint에 서빙 필수 정보가 없습니다. 누락: {missing}")

    LSTM_SERVING_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, LSTM_SERVING_PATH)

    print("===== LSTM serving artifact 저장 완료 =====")
    print(LSTM_SERVING_PATH)


def export_outcome_ranker_artifact():
    if not OUTCOME_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {OUTCOME_PATH}")

    df = pd.read_csv(OUTCOME_PATH)

    sort_cols = [
        c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # outcome recommender와 동일하게 80/20 split의 train 구간만 사용
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()

    global_mean = float(train_df[TARGET_COL].mean())

    print("\n===== Outcome artifact 생성 =====")
    print("전체:", df.shape)
    print("Train:", train_df.shape)
    print("global_mean:", global_mean)

    stats_pitcher_count_action_zone = build_stat_dict(
        train_df,
        ["pitcher", "balls", "strikes", "action_pitch_type", "action_zone"],
    )

    stats_pitcher_action_zone = build_stat_dict(
        train_df,
        ["pitcher", "action_pitch_type", "action_zone"],
    )

    stats_count_action_zone = build_stat_dict(
        train_df,
        ["balls", "strikes", "action_pitch_type", "action_zone"],
    )

    stats_action_zone = build_stat_dict(
        train_df,
        ["action_pitch_type", "action_zone"],
    )

    stats_action = build_stat_dict(
        train_df,
        ["action_pitch_type"],
    )

    valid_zones = sorted(
        [
            z for z in train_df["action_zone"].dropna().unique().tolist()
            if str(z).startswith("Z") and z not in ["ZPAD", "ZUNK"]
        ],
        key=zone_sort_key,
    )

    pitcher_top_pitches = (
        train_df
        .groupby("pitcher")["action_pitch_type"]
        .agg(
            lambda x: [
                p for p in x.value_counts().index.tolist()
                if p != "OTHER"
            ][:3]
        )
        .to_dict()
    )

    # key를 int로 정리
    pitcher_top_pitches = {
        int(k): v for k, v in pitcher_top_pitches.items()
    }

    global_top_pitches = [
        p for p in train_df["action_pitch_type"].value_counts().index.tolist()
        if p != "OTHER"
    ][:3]

    artifact = {
        "target_col": TARGET_COL,
        "global_mean": global_mean,
        "alpha": ALPHA,
        "thresholds": {
            "pitcher_count_action_zone": MIN_COUNT_PITCHER_COUNT_ACTION_ZONE,
            "pitcher_action_zone": MIN_COUNT_PITCHER_ACTION_ZONE,
            "count_action_zone": MIN_COUNT_COUNT_ACTION_ZONE,
            "action_zone": MIN_COUNT_ACTION_ZONE,
            "action": MIN_COUNT_ACTION,
        },
        "valid_zones": valid_zones,
        "pitcher_top_pitches": pitcher_top_pitches,
        "global_top_pitches": global_top_pitches,
        "stats": {
            "pitcher_count_action_zone": stats_pitcher_count_action_zone,
            "pitcher_action_zone": stats_pitcher_action_zone,
            "count_action_zone": stats_count_action_zone,
            "action_zone": stats_action_zone,
            "action": stats_action,
        },
    }

    OUTCOME_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTCOME_ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print("\n===== Outcome ranker artifact 저장 완료 =====")
    print(OUTCOME_ARTIFACT_PATH)
    print("valid_zones:", valid_zones)
    print("global_top_pitches:", global_top_pitches)

    print("\n통계 테이블 크기:")
    for name, value in artifact["stats"].items():
        print(name, len(value))


def main():
    export_lstm_serving_artifact()
    export_outcome_ranker_artifact()


if __name__ == "__main__":
    main()