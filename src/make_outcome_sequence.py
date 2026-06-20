from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# 설정
# =========================

INPUT_PATH = Path("data/raw/savant_data_with_outcome.csv")
OUT_PATH = Path("data/processed/outcome_sequence_data.csv")

N = 5
PAD_TOKEN = "PAD"
ZONE_PAD = "ZPAD"

MIN_PITCH_COUNT = 500
GROUP_RARE_PITCHES = True

USE_COLS = [
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
    "delta_run_exp",
    "bat_score",
    "fld_score",
    "bat_score_diff",
]

FORCE_OTHER_PITCHES = {"CS", "EP", "FA", "FO", "KN", "PO", "SC", "SV", "UN"}

def zone_to_token(x):
    if pd.isna(x):
        return "ZUNK"

    try:
        return f"Z{int(float(x))}"
    except ValueError:
        return "ZUNK"


def left_pad(values, n, pad_token):
    values = list(values)[-n:]
    pad_len = n - len(values)
    return [pad_token] * pad_len + values


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {INPUT_PATH}")

    cols = pd.read_csv(INPUT_PATH, nrows=0).columns.tolist()
    missing = [c for c in USE_COLS if c not in cols]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = pd.read_csv(INPUT_PATH, usecols=USE_COLS)

    print("===== 원본 크기 =====")
    print(df.shape)

    df["delta_run_exp"] = pd.to_numeric(df["delta_run_exp"], errors="coerce")

    # 기대득점 모델에서는 현재 pitch_type, 현재 zone, delta_run_exp가 모두 필요함
    df = df.dropna(subset=["pitch_type", "zone", "delta_run_exp"]).copy()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # 주자 상황 변환
    df["runner_1b"] = df["on_1b"].notna().astype(int)
    df["runner_2b"] = df["on_2b"].notna().astype(int)
    df["runner_3b"] = df["on_3b"].notna().astype(int)

    # zone 토큰화
    df["zone_token"] = df["zone"].apply(zone_to_token)

    # 이상한 zone은 현재 action으로 쓰기 어렵기 때문에 제외
    df = df[df["zone_token"] != "ZUNK"].copy()

    # 희귀 구종 통합
    df["pitch_type_original"] = df["pitch_type"].astype(str)
    counts = df["pitch_type_original"].value_counts()

    if GROUP_RARE_PITCHES:
        rare_set = set(counts[counts < MIN_PITCH_COUNT].index)
    else:
        rare_set = set()

    rare_set = rare_set.union(FORCE_OTHER_PITCHES)

    df["pitch_type_mapped"] = df["pitch_type_original"].apply(
        lambda x: "OTHER" if x in rare_set else x
    )

    print("\n===== 희귀 구종 처리 전 pitch_type 분포 =====")
    print(counts.head(30))

    print("\n===== OTHER로 묶을 희귀 구종 =====")
    print(sorted(rare_set))

    print("\n===== 희귀 구종 처리 후 pitch_type 분포 =====")
    print(df["pitch_type_mapped"].value_counts())

    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    samples = []

    for _, group in df.groupby(["game_pk", "at_bat_number"], sort=False):
        group = group.sort_values("pitch_number").reset_index(drop=True)

        # 첫 투구는 이전 투구가 없어서 제외
        for i in range(1, len(group)):
            prev = group.iloc[max(0, i - N):i]
            current = group.iloc[i]

            prev_pitch_types = left_pad(
                prev["pitch_type_mapped"].tolist(),
                N,
                PAD_TOKEN,
            )

            prev_zones = left_pad(
                prev["zone_token"].tolist(),
                N,
                ZONE_PAD,
            )

            sample = {
                "game_date": current["game_date"],
                "game_pk": current["game_pk"],
                "at_bat_number": current["at_bat_number"],
                "pitch_number": current["pitch_number"],

                # state: 이전 흐름
                "seq_pitch_type": "|".join(prev_pitch_types),
                "seq_zone": "|".join(prev_zones),
                "seq_len": len(prev),

                "prev_pitch_type": prev_pitch_types[-1],
                "prev_zone": prev_zones[-1],

                # state: 현재 경기 상황
                "balls": current["balls"],
                "strikes": current["strikes"],
                "outs_when_up": current["outs_when_up"],
                "inning": current["inning"],
                "runner_1b": current["runner_1b"],
                "runner_2b": current["runner_2b"],
                "runner_3b": current["runner_3b"],

                "bat_score": current.get("bat_score", np.nan),
                "fld_score": current.get("fld_score", np.nan),
                "bat_score_diff": current.get("bat_score_diff", np.nan),

                "pitcher": current["pitcher"],
                "batter": current["batter"],
                "stand": current["stand"],
                "p_throws": current["p_throws"],

                # action: 실제 선택된 구종 + 실제 도착 zone
                "action_pitch_type": current["pitch_type_mapped"],
                "action_pitch_type_original": current["pitch_type_original"],
                "action_zone": current["zone_token"],
                "action_zone_raw": current["zone"],
                "plate_x": current["plate_x"],
                "plate_z": current["plate_z"],

                # target
                "delta_run_exp": current["delta_run_exp"],

                # 분석용
                "description": current["description"],
                "events": current["events"],
            }

            samples.append(sample)

    seq_df = pd.DataFrame(samples)

    print("\n===== outcome sequence 크기 =====")
    print(seq_df.shape)

    print("\n===== action_pitch_type 분포 =====")
    print(seq_df["action_pitch_type"].value_counts())

    print("\n===== action_zone 분포 =====")
    print(seq_df["action_zone"].value_counts().sort_index())

    print("\n===== delta_run_exp 요약 =====")
    print(seq_df["delta_run_exp"].describe())
    print("\nquantile:")
    print(seq_df["delta_run_exp"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", OUT_PATH)


if __name__ == "__main__":
    main()