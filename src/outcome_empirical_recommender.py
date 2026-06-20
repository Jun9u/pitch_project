from pathlib import Path
import math
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 설정
# =========================

DATA_PATH = Path("data/processed/outcome_sequence_data.csv")
LSTM_PRED_PATH = Path("outputs/lstm_weighted_macro_predictions.csv")

EVAL_OUTPUT_PATH = Path("outputs/outcome_empirical_eval.csv")
RECOMMEND_OUTPUT_PATH = Path("outputs/outcome_empirical_recommendations.csv")

TARGET_COL = "delta_run_exp"

ALPHA = 50.0

MIN_COUNT_PITCHER_COUNT_ACTION_ZONE = 5
MIN_COUNT_PITCHER_ACTION_ZONE = 10
MIN_COUNT_COUNT_ACTION_ZONE = 20
MIN_COUNT_ACTION_ZONE = 30
MIN_COUNT_ACTION = 50

TOP_N_RECOMMENDATIONS = 5

KEY_COLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
]


# =========================
# 유틸
# =========================

def zone_sort_key(z):
    try:
        return int(str(z).replace("Z", ""))
    except ValueError:
        return 999


def smoothed_mean(sum_value, count_value, global_mean, alpha=50.0):
    return (sum_value + alpha * global_mean) / (count_value + alpha)


def build_stat_dict(train_df, group_cols):
    stats = (
        train_df
        .groupby(group_cols, dropna=False)[TARGET_COL]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )

    stat_dict = {}

    for _, row in stats.iterrows():
        key = tuple(row[c] for c in group_cols)
        stat_dict[key] = {
            "sum": float(row["sum"]),
            "count": int(row["count"]),
            "mean": float(row["mean"]),
        }

    return stat_dict


def get_stat_score(stat, global_mean):
    return smoothed_mean(
        stat["sum"],
        stat["count"],
        global_mean,
        ALPHA,
    )


def load_lstm_predictions(test_df):
    if not LSTM_PRED_PATH.exists():
        print("\nLSTM 예측 파일 없음. 투수별/전체 후보로 대체합니다.")
        test_df["top3_pitch_type"] = np.nan
        return test_df

    pred_cols = pd.read_csv(LSTM_PRED_PATH, nrows=0).columns.tolist()

    needed = KEY_COLS + ["top3_pitch_type"]
    missing = [c for c in needed if c not in pred_cols]

    if missing:
        print("\nLSTM 예측 파일에 필요한 컬럼이 없습니다:", missing)
        test_df["top3_pitch_type"] = np.nan
        return test_df

    pred_df = pd.read_csv(LSTM_PRED_PATH, usecols=needed)
    pred_df = pred_df.drop_duplicates(KEY_COLS)

    merged = test_df.merge(pred_df, on=KEY_COLS, how="left")

    print("\n===== LSTM top3 결합 =====")
    print("test rows:", len(test_df))
    print("merged rows:", len(merged))
    print("top3 결측:", merged["top3_pitch_type"].isna().sum())

    return merged


def parse_top3(text):
    if pd.isna(text):
        return []

    values = str(text).split("|")
    values = [v for v in values if v and v != "nan"]

    return values


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    sort_cols = [
        c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # LSTM의 test 구간과 맞추기 위해 80/20 분할
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("===== 데이터 분할 =====")
    print("전체:", df.shape)
    print("Train:", train_df.shape)
    print("Test :", test_df.shape)

    global_mean = train_df[TARGET_COL].mean()

    print("\n===== train delta_run_exp =====")
    print(train_df[TARGET_COL].describe())
    print("global_mean:", global_mean)

    # 계층적 통계 생성
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

    print("\n===== 통계 테이블 크기 =====")
    print("pitcher+count+action+zone:", len(stats_pitcher_count_action_zone))
    print("pitcher+action+zone      :", len(stats_pitcher_action_zone))
    print("count+action+zone        :", len(stats_count_action_zone))
    print("action+zone              :", len(stats_action_zone))
    print("action                   :", len(stats_action))

    # 후보 zone
    valid_zones = sorted(
        [
            z for z in train_df["action_zone"].dropna().unique().tolist()
            if str(z).startswith("Z") and z not in ["ZPAD", "ZUNK"]
        ],
        key=zone_sort_key,
    )

    print("\n===== 후보 zone =====")
    print(valid_zones)

    # 투수별 후보 구종 fallback
    pitcher_top_pitches = (
        train_df
        .groupby("pitcher")["action_pitch_type"]
        .agg(lambda x: x.value_counts().index[:3].tolist())
        .to_dict()
    )

    global_top_pitches = train_df["action_pitch_type"].value_counts().index[:3].tolist()

    print("\n===== 전체 top pitches =====")
    print(global_top_pitches)

    def predict_delta(row, action_pitch_type, action_zone):
        """
        낮을수록 투수에게 유리한 추천 점수.
        """
        key1 = (
            row["pitcher"],
            row["balls"],
            row["strikes"],
            action_pitch_type,
            action_zone,
        )

        stat = stats_pitcher_count_action_zone.get(key1)

        if stat is not None and stat["count"] >= MIN_COUNT_PITCHER_COUNT_ACTION_ZONE:
            return get_stat_score(stat, global_mean), "pitcher+count+action+zone", stat["count"]

        key2 = (
            row["pitcher"],
            action_pitch_type,
            action_zone,
        )

        stat = stats_pitcher_action_zone.get(key2)

        if stat is not None and stat["count"] >= MIN_COUNT_PITCHER_ACTION_ZONE:
            return get_stat_score(stat, global_mean), "pitcher+action+zone", stat["count"]

        key3 = (
            row["balls"],
            row["strikes"],
            action_pitch_type,
            action_zone,
        )

        stat = stats_count_action_zone.get(key3)

        if stat is not None and stat["count"] >= MIN_COUNT_COUNT_ACTION_ZONE:
            return get_stat_score(stat, global_mean), "count+action+zone", stat["count"]

        key4 = (
            action_pitch_type,
            action_zone,
        )

        stat = stats_action_zone.get(key4)

        if stat is not None and stat["count"] >= MIN_COUNT_ACTION_ZONE:
            return get_stat_score(stat, global_mean), "action+zone", stat["count"]

        key5 = (action_pitch_type,)

        stat = stats_action.get(key5)

        if stat is not None and stat["count"] >= MIN_COUNT_ACTION:
            return get_stat_score(stat, global_mean), "action", stat["count"]

        return global_mean, "global", len(train_df)

    # 실제 test action에 대한 예측 성능 확인
    actual_pred = []
    actual_sources = []

    for _, row in test_df.iterrows():
        pred, source, count = predict_delta(
            row,
            row["action_pitch_type"],
            row["action_zone"],
        )

        actual_pred.append(pred)
        actual_sources.append(source)

    y_true = test_df[TARGET_COL].astype(float).values
    y_pred = np.array(actual_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    eval_df = pd.DataFrame(
        [
            {
                "model": "Empirical Outcome Ranker",
                "target": "delta_run_exp",
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "global_mean": global_mean,
                "test_rows": len(test_df),
            }
        ]
    )

    EVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n===== 실제 action delta_run_exp 예측 성능 =====")
    print(eval_df.to_string(index=False))
    print("저장:", EVAL_OUTPUT_PATH)

    # LSTM top3와 결합
    test_df = load_lstm_predictions(test_df)

    recommendation_rows = []

    for _, row in test_df.iterrows():
        lstm_candidates = parse_top3(row.get("top3_pitch_type", np.nan))

        if not lstm_candidates:
            lstm_candidates = pitcher_top_pitches.get(
                row["pitcher"],
                global_top_pitches,
            )

        # 중복 제거 + OTHER 제외
        candidate_pitches = []

        # 1. LSTM Top-3
        for p in lstm_candidates:
            p = str(p).strip()

            if p in ["", "nan", "None", "OTHER"]:
                continue

            if p not in candidate_pitches:
                candidate_pitches.append(p)

        # 2. 해당 투수의 자주 던지는 구종 Top-3 추가
        fallback_candidates = pitcher_top_pitches.get(row["pitcher"], [])

        for p in fallback_candidates:
            p = str(p).strip()

            if p in ["", "nan", "None", "OTHER"]:
                continue

            if p not in candidate_pitches:
                candidate_pitches.append(p)

        # 3. 그래도 비면 전체 top pitch 사용
        if not candidate_pitches:
            for p in global_top_pitches:
                p = str(p).strip()

                if p in ["", "nan", "None", "OTHER"]:
                    continue

                if p not in candidate_pitches:
                    candidate_pitches.append(p)

        candidates = []

        for pitch in candidate_pitches:
            for zone in valid_zones:
                score, source, count = predict_delta(row, pitch, zone)

                candidates.append(
                    {
                        "pitch_type": pitch,
                        "zone": zone,
                        "pred_delta_run_exp": score,
                        "source": source,
                        "source_count": count,
                    }
                )

        candidates = sorted(candidates, key=lambda x: x["pred_delta_run_exp"])
        top_recs = candidates[:TOP_N_RECOMMENDATIONS]

        out = {
            "game_date": row.get("game_date"),
            "game_pk": row.get("game_pk"),
            "at_bat_number": row.get("at_bat_number"),
            "pitch_number": row.get("pitch_number"),

            "pitcher": row.get("pitcher"),
            "batter": row.get("batter"),
            "balls": row.get("balls"),
            "strikes": row.get("strikes"),
            "outs_when_up": row.get("outs_when_up"),
            "inning": row.get("inning"),
            "runner_1b": row.get("runner_1b"),
            "runner_2b": row.get("runner_2b"),
            "runner_3b": row.get("runner_3b"),

            "seq_pitch_type": row.get("seq_pitch_type"),
            "seq_zone": row.get("seq_zone"),
            "lstm_top3_pitch_type": row.get("top3_pitch_type"),

            "actual_pitch_type": row.get("action_pitch_type"),
            "actual_zone": row.get("action_zone"),
            "actual_delta_run_exp": row.get("delta_run_exp"),
            "actual_description": row.get("description"),
            "actual_events": row.get("events"),
        }

        if top_recs:
            best = top_recs[0]

            out["recommended_pitch_type"] = best["pitch_type"]
            out["recommended_zone"] = best["zone"]
            out["recommended_pred_delta_run_exp"] = best["pred_delta_run_exp"]
            out["recommendation_source"] = best["source"]
            out["recommendation_source_count"] = best["source_count"]

            # 실제 선택의 예측 점수와 추천 선택의 예측 점수 차이
            actual_score, actual_source, actual_count = predict_delta(
                row,
                row["action_pitch_type"],
                row["action_zone"],
            )

            out["actual_action_pred_delta_run_exp"] = actual_score
            out["predicted_improvement_vs_actual_action"] = (
                actual_score - best["pred_delta_run_exp"]
            )

        for i, rec in enumerate(top_recs, start=1):
            out[f"rec_{i}"] = (
                f"{rec['pitch_type']}/{rec['zone']}:"
                f"{rec['pred_delta_run_exp']:.4f}"
                f"({rec['source']},n={rec['source_count']})"
            )

        recommendation_rows.append(out)

    rec_df = pd.DataFrame(recommendation_rows)
    rec_df.to_csv(RECOMMEND_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n===== 추천 결과 샘플 =====")
    show_cols = [
        "lstm_top3_pitch_type",
        "recommended_pitch_type",
        "recommended_zone",
        "recommended_pred_delta_run_exp",
        "actual_pitch_type",
        "actual_zone",
        "actual_delta_run_exp",
        "predicted_improvement_vs_actual_action",
        "rec_1",
        "rec_2",
        "rec_3",
    ]

    show_cols = [c for c in show_cols if c in rec_df.columns]
    print(rec_df[show_cols].head(10).to_string(index=False))

    print("\n추천 결과 저장:", RECOMMEND_OUTPUT_PATH)


if __name__ == "__main__":
    main()