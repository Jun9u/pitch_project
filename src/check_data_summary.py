from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/savant_data.csv")

usecols = [
    "pitch_type",
    "game_date",
    "game_pk",
    "pitcher",
    "batter",
]

def main():
    cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    available_cols = [c for c in usecols if c in cols]

    if not available_cols:
        raise ValueError("확인 가능한 컬럼이 없습니다.")

    pitch_counts = pd.Series(dtype="float64")
    pitch_type_nulls = 0
    date_min = None
    date_max = None
    game_ids = set()
    pitcher_ids = set()
    batter_ids = set()

    for chunk in pd.read_csv(DATA_PATH, usecols=available_cols, chunksize=100_000):
        if "pitch_type" in chunk.columns:
            pitch_counts = pitch_counts.add(
                chunk["pitch_type"].value_counts(dropna=False),
                fill_value=0
            )
            pitch_type_nulls += chunk["pitch_type"].isna().sum()

        if "game_date" in chunk.columns:
            dates = pd.to_datetime(chunk["game_date"], errors="coerce")
            cur_min = dates.min()
            cur_max = dates.max()

            if pd.notna(cur_min):
                date_min = cur_min if date_min is None else min(date_min, cur_min)
            if pd.notna(cur_max):
                date_max = cur_max if date_max is None else max(date_max, cur_max)

        if "game_pk" in chunk.columns:
            game_ids.update(chunk["game_pk"].dropna().unique().tolist())

        if "pitcher" in chunk.columns:
            pitcher_ids.update(chunk["pitcher"].dropna().unique().tolist())

        if "batter" in chunk.columns:
            batter_ids.update(chunk["batter"].dropna().unique().tolist())

    print("===== 날짜 범위 =====")
    print(date_min, "~", date_max)

    print("\n===== 경기 수 =====")
    print(len(game_ids))

    print("\n===== 투수 수 =====")
    print(len(pitcher_ids))

    print("\n===== 타자 수 =====")
    print(len(batter_ids))

    print("\n===== pitch_type 결측치 수 =====")
    print(pitch_type_nulls)

    print("\n===== pitch_type 분포 상위 20개 =====")
    print(pitch_counts.sort_values(ascending=False).head(20).astype(int))

if __name__ == "__main__":
    main()