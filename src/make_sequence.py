from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/clean_pitch_data.csv")
OUT_PATH = Path("data/processed/sequence_data.csv")

N = 5
PAD_TOKEN = "PAD"
ZONE_PAD = "ZPAD"

# 너무 적은 구종은 OTHER로 묶음
# 보고서에 "희귀 구종은 OTHER로 통합"이라고 쓰면 됨
MIN_TARGET_COUNT = 500
GROUP_RARE_PITCHES = True
FORCE_OTHER_PITCHES = {"CS", "EP", "FA", "FO", "KN", "PO", "SC", "SV", "UN"} 


def zone_to_token(x):
    """zone 값을 문자열 토큰으로 변환"""
    if pd.isna(x):
        return "ZUNK"

    try:
        return f"Z{int(float(x))}"
    except ValueError:
        return f"Z{x}"


def left_pad(values, n, pad_token):
    """길이가 n보다 짧으면 왼쪽에 PAD 추가"""
    values = list(values)[-n:]
    pad_len = n - len(values)
    return [pad_token] * pad_len + values


def map_rare_token_sequence(seq_text, rare_set):
    """seq_pitch_type 안의 희귀 구종을 OTHER로 변경"""
    tokens = seq_text.split("|")
    mapped = [
        "OTHER" if token in rare_set else token
        for token in tokens
    ]
    return "|".join(mapped)


def main():
    df = pd.read_csv(INPUT_PATH)

    required_cols = [
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
        "runner_1b",
        "runner_2b",
        "runner_3b",
        "stand",
        "p_throws",
        "zone",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["pitch_type"] = df["pitch_type"].astype(str)
    df["zone_token"] = df["zone"].apply(zone_to_token)

    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    samples = []

    for _, group in df.groupby(["game_pk", "at_bat_number"], sort=False):
        group = group.sort_values("pitch_number").reset_index(drop=True)

        # 첫 투구는 이전 투구가 없으므로 제외
        # 두 번째 투구부터는 부족한 부분을 PAD로 채워 사용
        for i in range(1, len(group)):
            prev = group.iloc[max(0, i - N):i]
            current = group.iloc[i]

            prev_pitch_types = left_pad(
                prev["pitch_type"].tolist(),
                N,
                PAD_TOKEN
            )

            prev_zones = left_pad(
                prev["zone_token"].tolist(),
                N,
                ZONE_PAD
            )

            sample = {
                "game_date": current["game_date"],
                "game_pk": current["game_pk"],
                "at_bat_number": current["at_bat_number"],
                "pitch_number": current["pitch_number"],

                "seq_pitch_type": "|".join(prev_pitch_types),
                "seq_zone": "|".join(prev_zones),
                "seq_len": len(prev),

                "prev_pitch_type": prev_pitch_types[-1],
                "prev_zone": prev_zones[-1],

                "balls": current["balls"],
                "strikes": current["strikes"],
                "outs_when_up": current["outs_when_up"],
                "inning": current["inning"],

                "runner_1b": current["runner_1b"],
                "runner_2b": current["runner_2b"],
                "runner_3b": current["runner_3b"],

                "pitcher": current["pitcher"],
                "batter": current["batter"],
                "stand": current["stand"],
                "p_throws": current["p_throws"],

                "target_pitch_type": current["pitch_type"],
            }

            samples.append(sample)

    # samples 리스트를 DataFrame으로 변환
    seq_df = pd.DataFrame(samples)

    if seq_df.empty:
        raise ValueError("생성된 시퀀스 데이터가 없습니다. 입력 데이터와 그룹 구성 로직을 확인하세요.")

    # 날짜/경기/타석/투구 순서 정렬
    seq_df = seq_df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    print("===== 희귀 구종 처리 전 target 분포 =====")
    print(seq_df["target_pitch_type"].value_counts().head(30))

    if GROUP_RARE_PITCHES:
        # train 기준으로 희귀 구종 결정
        train_end = int(len(seq_df) * 0.7)
        train_counts = seq_df.iloc[:train_end]["target_pitch_type"].value_counts()

        rare_set = set(train_counts[train_counts < MIN_TARGET_COUNT].index)
        rare_set = rare_set.union(FORCE_OTHER_PITCHES)

        print("\n===== OTHER로 묶을 희귀 구종, train 기준 =====")
        print(sorted(rare_set))

        seq_df["target_pitch_type_original"] = seq_df["target_pitch_type"]

        seq_df["target_pitch_type"] = seq_df["target_pitch_type"].apply(
            lambda x: "OTHER" if x in rare_set else x
        )

        seq_df["seq_pitch_type"] = seq_df["seq_pitch_type"].apply(
            lambda x: map_rare_token_sequence(x, rare_set)
        )

        seq_df["prev_pitch_type"] = seq_df["prev_pitch_type"].apply(
            lambda x: "OTHER" if x in rare_set else x
        )

    print("\n===== 최종 target 분포 =====")
    print(seq_df["target_pitch_type"].value_counts().head(30))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n시퀀스 데이터 크기:", seq_df.shape)
    print(seq_df.head())
    print("저장 완료:", OUT_PATH)


if __name__ == "__main__":
    main()