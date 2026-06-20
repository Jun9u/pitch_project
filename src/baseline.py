from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = Path("data/processed/sequence_data.csv")
OUTPUT_PATH = Path("outputs/baseline_results.csv")

TARGET_COL = "target_pitch_type"


def topk_accuracy(y_true, y_topk):
    correct = 0

    for true_label, pred_list in zip(y_true, y_topk):
        if true_label in pred_list:
            correct += 1

    return correct / len(y_true)


def normalize_key(key):
    if isinstance(key, tuple):
        return key
    return (key,)


def build_lookup(train_df, feature_cols, target_col, k=3):
    top1_lookup = {}
    topk_lookup = {}

    grouped = train_df.groupby(feature_cols, dropna=False)

    for key, group in grouped:
        key = normalize_key(key)
        counts = group[target_col].value_counts()

        top1_lookup[key] = counts.index[0]
        topk_lookup[key] = counts.index[:k].tolist()

    return top1_lookup, topk_lookup


def evaluate_global_baseline(train_df, test_df, target_col):
    counts = train_df[target_col].value_counts()

    global_top1 = counts.index[0]
    global_top3 = counts.index[:3].tolist()

    y_true = test_df[target_col].tolist()
    y_pred = [global_top1] * len(test_df)
    y_top3 = [global_top3] * len(test_df)

    return {
        "model": "Global Majority",
        "features": "-",
        "accuracy": accuracy_score(y_true, y_pred),
        "top3_accuracy": topk_accuracy(y_true, y_top3),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fallback": global_top1,
    }


def evaluate_condition_baseline(name, train_df, test_df, feature_cols, target_col):
    global_counts = train_df[target_col].value_counts()
    fallback_top1 = global_counts.index[0]
    fallback_top3 = global_counts.index[:3].tolist()

    top1_lookup, topk_lookup = build_lookup(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        k=3
    )

    y_true = test_df[target_col].tolist()
    y_pred = []
    y_top3 = []

    for _, row in test_df.iterrows():
        key = tuple(row[c] for c in feature_cols)

        pred = top1_lookup.get(key, fallback_top1)
        pred_top3 = topk_lookup.get(key, fallback_top3)

        y_pred.append(pred)
        y_top3.append(pred_top3)

    return {
        "model": name,
        "features": ", ".join(feature_cols),
        "accuracy": accuracy_score(y_true, y_pred),
        "top3_accuracy": topk_accuracy(y_true, y_top3),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fallback": fallback_top1,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼이 없습니다.")

    sort_cols = [
        c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("===== 데이터 크기 =====")
    print("전체:", df.shape)
    print("Train:", train_df.shape)
    print("Test:", test_df.shape)

    print("\n===== Train target 분포 =====")
    print(train_df[TARGET_COL].value_counts())

    print("\n===== Test target 분포 =====")
    print(test_df[TARGET_COL].value_counts())

    baseline_specs = [
        (
            "Count/Base Majority",
            [
                "balls",
                "strikes",
                "outs_when_up",
                "runner_1b",
                "runner_2b",
                "runner_3b",
            ],
        ),
        (
            "Pitcher Majority",
            [
                "pitcher",
            ],
        ),
        (
            "Pitcher + Count Majority",
            [
                "pitcher",
                "balls",
                "strikes",
            ],
        ),
        (
            "Previous Pitch Markov",
            [
                "prev_pitch_type",
            ],
        ),
        (
            "Previous Pitch + Count",
            [
                "prev_pitch_type",
                "balls",
                "strikes",
            ],
        ),
    ]

    results = []

    results.append(
        evaluate_global_baseline(
            train_df=train_df,
            test_df=test_df,
            target_col=TARGET_COL
        )
    )

    for name, feature_cols in baseline_specs:
        missing = [c for c in feature_cols if c not in df.columns]

        if missing:
            print(f"\n{name} 건너뜀. 누락 컬럼: {missing}")
            continue

        results.append(
            evaluate_condition_baseline(
                name=name,
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                target_col=TARGET_COL
            )
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by=["accuracy", "top3_accuracy"],
        ascending=False
    )

    print("\n===== Baseline 결과 =====")
    print(result_df.to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", OUTPUT_PATH)


if __name__ == "__main__":
    main()