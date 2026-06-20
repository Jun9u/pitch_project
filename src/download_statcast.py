from pathlib import Path
import pandas as pd
from pybaseball import statcast


# =========================
# 설정
# =========================

START_DATE = "2023-03-01"
END_DATE = "2025-09-27"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 현재 피치타입 예측 모델에 필요한 기본 컬럼
MODEL_COLS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "p_throws",
    "stand",
    "zone",
    "plate_x",
    "plate_z",
    "description",
    "events",
]

# 추후 기대득점 기반 outcome model용 후보 컬럼
OUTCOME_COLS = [
    "delta_run_exp",
    "delta_home_win_exp",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "post_home_score",
    "post_away_score",
    "post_bat_score",
    "post_fld_score",
    "home_score_diff",
    "bat_score_diff",
    "home_win_exp",
    "bat_win_exp",
]

# 추후 더 정교한 모델에 쓸 수 있는 추가 컬럼
OPTIONAL_DETAIL_COLS = [
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "type",
    "des",
    "bb_type",
    "hit_location",
    "launch_speed",
    "launch_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
]

MODEL_OUTPUT_PATH = RAW_DIR / "savant_data.csv"
WIDE_OUTPUT_PATH = RAW_DIR / "savant_data_with_outcome.csv"
COLUMN_LIST_PATH = RAW_DIR / "statcast_columns.txt"


def save_column_list(cols):
    with open(COLUMN_LIST_PATH, "w", encoding="utf-8") as f:
        for col in cols:
            f.write(f"{col}\n")


def print_col_status(cols, target_cols, title):
    print(f"\n===== {title} =====")
    for col in target_cols:
        print(f"{col}: {col in cols}")


def main():
    print("===== Statcast 데이터 다운로드 =====")
    print("기간:", START_DATE, "~", END_DATE)

    df = statcast(START_DATE, END_DATE)

    print("\n===== 원본 데이터 크기 =====")
    print(df.shape)

    cols = df.columns.tolist()
    save_column_list(cols)

    print("\n===== 전체 컬럼 수 =====")
    print(len(cols))
    print("컬럼 목록 저장:", COLUMN_LIST_PATH)

    print_col_status(cols, MODEL_COLS, "현재 피치타입 예측 모델 컬럼 존재 여부")
    print_col_status(cols, OUTCOME_COLS, "기대득점/승리확률 관련 컬럼 존재 여부")
    print_col_status(cols, OPTIONAL_DETAIL_COLS, "추가 세부 컬럼 존재 여부")

    missing_model_cols = [c for c in MODEL_COLS if c not in cols]

    if missing_model_cols:
        raise ValueError(f"현재 모델 필수 컬럼 누락: {missing_model_cols}")

    # 1. 현재 프로젝트용 파일 저장
    model_df = df[MODEL_COLS].copy()
    model_df.to_csv(MODEL_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n===== 현재 모델용 CSV 저장 완료 =====")
    print(MODEL_OUTPUT_PATH)
    print(model_df.shape)

    # 2. 기대득점 기반 후속 모델용 넓은 파일 저장
    wide_cols = []
    for col in MODEL_COLS + OUTCOME_COLS + OPTIONAL_DETAIL_COLS:
        if col in cols and col not in wide_cols:
            wide_cols.append(col)

    wide_df = df[wide_cols].copy()
    wide_df.to_csv(WIDE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n===== 후속 확장용 CSV 저장 완료 =====")
    print(WIDE_OUTPUT_PATH)
    print(wide_df.shape)

    print("\n===== delta_run_exp 확인 =====")
    if "delta_run_exp" in df.columns:
        delta = pd.to_numeric(df["delta_run_exp"], errors="coerce")

        print("delta_run_exp 존재: True")
        print("결측치 수:", int(delta.isna().sum()))
        print("유효값 수:", int(delta.notna().sum()))
        print("min:", delta.min())
        print("max:", delta.max())
        print("mean:", delta.mean())
        print("std:", delta.std())
        print("quantile:")
        print(delta.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    else:
        print("delta_run_exp 존재: False")
        print("현재 pybaseball 결과에 delta_run_exp가 없습니다.")
        print("이 경우 기대득점 기반 최적 추천은 현재 데이터셋으로는 바로 구현하기 어렵습니다.")

    print("\n===== pitch_type 분포 =====")
    print(model_df["pitch_type"].value_counts(dropna=False).head(30))


if __name__ == "__main__":
    main()