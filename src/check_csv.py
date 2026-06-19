from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/savant_data.csv")

required_cols = [
    "pitch_type",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "stand",
    "p_throws",
    "zone",
    "description",
    "events",
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

    # 전체 파일을 읽지 않고 컬럼만 확인
    cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

    print("===== 컬럼 수 =====")
    print(len(cols))

    print("\n===== 컬럼 목록 =====")
    print(cols)

    print("\n===== 필수 컬럼 누락 여부 =====")
    missing = [c for c in required_cols if c not in cols]
    print(missing)

    # 행 수 확인
    with open(DATA_PATH, "rb") as f:
        row_count = sum(1 for _ in f) - 1

    print("\n===== 행 수 =====")
    print(row_count)

    # 필요한 컬럼만 샘플 확인
    usecols = [c for c in required_cols if c in cols]
    sample = pd.read_csv(DATA_PATH, usecols=usecols, nrows=5)

    print("\n===== 샘플 5행 =====")
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()