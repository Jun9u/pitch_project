from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/clean_pitch_data.csv")
OUT_PATH = Path("data/processed/sequence_data.csv")

N = 5

def main():
    df = pd.read_csv(INPUT_PATH)

    samples = []

    group_cols = ["game_pk", "at_bat_number"]

    for _, group in df.groupby(group_cols):
        group = group.sort_values("pitch_number").reset_index(drop=True)

        for i in range(N, len(group)):
            prev = group.iloc[i - N:i]
            current = group.iloc[i]

            sample = {
                "seq_pitch_type": "|".join(prev["pitch_type"].astype(str).tolist()),
                "seq_zone": "|".join(prev["zone"].astype(str).tolist()),

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

    seq_df = pd.DataFrame(samples)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("시퀀스 데이터 크기:", seq_df.shape)
    print(seq_df.head())
    print("저장 완료:", OUT_PATH)

if __name__ == "__main__":
    main()