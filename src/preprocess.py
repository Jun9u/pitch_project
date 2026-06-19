from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/savant_data.csv")
OUT_PATH = Path("data/processed/clean_pitch_data.csv")

USE_COLS = [
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
    df = pd.read_csv(RAW_PATH, usecols=USE_COLS)

    print("원본 크기:", df.shape)

    # 정답 없는 행 제거
    df = df.dropna(subset=["pitch_type"])

    # 날짜 변환
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # 주자 상황: 비어 있으면 주자 없음, 값이 있으면 주자 있음
    df["runner_1b"] = df["on_1b"].notna().astype(int)
    df["runner_2b"] = df["on_2b"].notna().astype(int)
    df["runner_3b"] = df["on_3b"].notna().astype(int)

    # 원래 주자 ID 컬럼 제거
    df = df.drop(columns=["on_1b", "on_2b", "on_3b"])

    # 정렬
    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("전처리 후 크기:", df.shape)
    print("저장 완료:", OUT_PATH)

if __name__ == "__main__":
    main()