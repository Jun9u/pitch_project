from pathlib import Path
import random
import time
import platform

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler


# =========================
# 설정
# =========================

EXPERIMENT_NAME = "lstm_weighted_macro"

DATA_PATH = Path("data/processed/sequence_data.csv")

MODEL_PATH = Path(f"models/{EXPERIMENT_NAME}_best.pt")
RESULT_PATH = Path(f"outputs/{EXPERIMENT_NAME}_result.csv")
PRED_PATH = Path(f"outputs/{EXPERIMENT_NAME}_predictions.csv")
HISTORY_PATH = Path(f"outputs/{EXPERIMENT_NAME}_training_log.csv")
REPORT_PATH = Path(f"outputs/{EXPERIMENT_NAME}_classification_report.csv")

TARGET_COL = "target_pitch_type"

SEQ_PITCH_COL = "seq_pitch_type"
SEQ_ZONE_COL = "seq_zone"

NUMERIC_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "runner_1b",
    "runner_2b",
    "runner_3b",
]

PITCHER_COL = "pitcher"
STAND_COL = "stand"
THROWS_COL = "p_throws"

SEED = 42
N_EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

PITCH_EMB_DIM = 32
ZONE_EMB_DIM = 8
PITCHER_EMB_DIM = 16
STAND_EMB_DIM = 4
THROWS_EMB_DIM = 4

HIDDEN_DIM = 96
NUM_LAYERS = 1
DROPOUT = 0.2

USE_CLASS_WEIGHTS = True
BEST_METRIC = "macro_f1"  # "accuracy", "top3_accuracy", "macro_f1", "weighted_f1"


# =========================
# 유틸 함수
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def split_tokens(text):
    if pd.isna(text):
        return []
    return str(text).split("|")


def build_vocab_from_sequences(series, special_tokens):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    for text in series.dropna():
        for token in split_tokens(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def build_vocab_from_column(series, special_tokens):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    for value in series.dropna().astype(str):
        if value not in vocab:
            vocab[value] = len(vocab)

    return vocab


def encode_sequence(text, vocab, max_len=5, pad_token="PAD", unk_token="UNK"):
    tokens = split_tokens(text)

    if len(tokens) > max_len:
        tokens = tokens[-max_len:]

    if len(tokens) < max_len:
        tokens = [pad_token] * (max_len - len(tokens)) + tokens

    unk_id = vocab.get(unk_token, 1)

    return [vocab.get(token, unk_id) for token in tokens]


def encode_category(value, vocab, unk_token="UNK"):
    if pd.isna(value):
        return vocab[unk_token]

    return vocab.get(str(value), vocab[unk_token])


def topk_accuracy_score(y_true, y_topk):
    correct = 0

    for true_label, pred_list in zip(y_true, y_topk):
        if true_label in pred_list:
            correct += 1

    return correct / len(y_true)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================
# Dataset
# =========================

class PitchSequenceDataset(Dataset):
    def __init__(
        self,
        df,
        pitch_vocab,
        zone_vocab,
        pitcher_vocab,
        stand_vocab,
        throws_vocab,
        target_to_idx,
        scaler,
        max_len=5,
    ):
        self.df = df.reset_index(drop=True)

        self.pitch_vocab = pitch_vocab
        self.zone_vocab = zone_vocab
        self.pitcher_vocab = pitcher_vocab
        self.stand_vocab = stand_vocab
        self.throws_vocab = throws_vocab
        self.target_to_idx = target_to_idx
        self.scaler = scaler
        self.max_len = max_len

        numeric_data = self.df[NUMERIC_COLS].astype(float).values
        self.numeric = self.scaler.transform(numeric_data).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq_pitch = encode_sequence(
            row[SEQ_PITCH_COL],
            self.pitch_vocab,
            max_len=self.max_len,
            pad_token="PAD",
            unk_token="UNK",
        )

        seq_zone = encode_sequence(
            row[SEQ_ZONE_COL],
            self.zone_vocab,
            max_len=self.max_len,
            pad_token="ZPAD",
            unk_token="ZUNK",
        )

        pitcher = encode_category(row[PITCHER_COL], self.pitcher_vocab)
        stand = encode_category(row[STAND_COL], self.stand_vocab)
        p_throws = encode_category(row[THROWS_COL], self.throws_vocab)

        label = self.target_to_idx[row[TARGET_COL]]

        return {
            "seq_pitch": torch.tensor(seq_pitch, dtype=torch.long),
            "seq_zone": torch.tensor(seq_zone, dtype=torch.long),
            "numeric": torch.tensor(self.numeric[idx], dtype=torch.float32),
            "pitcher": torch.tensor(pitcher, dtype=torch.long),
            "stand": torch.tensor(stand, dtype=torch.long),
            "p_throws": torch.tensor(p_throws, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


# =========================
# Model
# =========================

class LSTMPitchModel(nn.Module):
    def __init__(
        self,
        num_pitch_tokens,
        num_zone_tokens,
        num_pitchers,
        num_stands,
        num_throws,
        num_numeric_features,
        num_classes,
    ):
        super().__init__()

        self.pitch_emb = nn.Embedding(
            num_embeddings=num_pitch_tokens,
            embedding_dim=PITCH_EMB_DIM,
            padding_idx=0,
        )

        self.zone_emb = nn.Embedding(
            num_embeddings=num_zone_tokens,
            embedding_dim=ZONE_EMB_DIM,
            padding_idx=0,
        )

        lstm_input_dim = PITCH_EMB_DIM + ZONE_EMB_DIM

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0 if NUM_LAYERS == 1 else DROPOUT,
        )

        self.pitcher_emb = nn.Embedding(
            num_embeddings=num_pitchers,
            embedding_dim=PITCHER_EMB_DIM,
        )

        self.stand_emb = nn.Embedding(
            num_embeddings=num_stands,
            embedding_dim=STAND_EMB_DIM,
        )

        self.throws_emb = nn.Embedding(
            num_embeddings=num_throws,
            embedding_dim=THROWS_EMB_DIM,
        )

        final_input_dim = (
            HIDDEN_DIM
            + PITCHER_EMB_DIM
            + STAND_EMB_DIM
            + THROWS_EMB_DIM
            + num_numeric_features
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, num_classes),
        )

    def forward(self, seq_pitch, seq_zone, numeric, pitcher, stand, p_throws):
        pitch_emb = self.pitch_emb(seq_pitch)
        zone_emb = self.zone_emb(seq_zone)

        seq_x = torch.cat([pitch_emb, zone_emb], dim=-1)

        _, (hidden, _) = self.lstm(seq_x)

        seq_repr = hidden[-1]

        pitcher_repr = self.pitcher_emb(pitcher)
        stand_repr = self.stand_emb(stand)
        throws_repr = self.throws_emb(p_throws)

        x = torch.cat(
            [
                seq_repr,
                numeric,
                pitcher_repr,
                stand_repr,
                throws_repr,
            ],
            dim=1,
        )

        logits = self.classifier(x)

        return logits


# =========================
# 학습 / 평가 함수
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        seq_pitch = batch["seq_pitch"].to(device)
        seq_zone = batch["seq_zone"].to(device)
        numeric = batch["numeric"].to(device)
        pitcher = batch["pitcher"].to(device)
        stand = batch["stand"].to(device)
        p_throws = batch["p_throws"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(
            seq_pitch=seq_pitch,
            seq_zone=seq_zone,
            numeric=numeric,
            pitcher=pitcher,
            stand=stand,
            p_throws=p_throws,
        )

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def evaluate(model, loader, criterion, device, idx_to_target, return_predictions=False):
    model.eval()

    total_loss = 0.0
    total_count = 0

    y_true = []
    y_pred = []
    y_top3 = []

    with torch.no_grad():
        for batch in loader:
            seq_pitch = batch["seq_pitch"].to(device)
            seq_zone = batch["seq_zone"].to(device)
            numeric = batch["numeric"].to(device)
            pitcher = batch["pitcher"].to(device)
            stand = batch["stand"].to(device)
            p_throws = batch["p_throws"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                seq_pitch=seq_pitch,
                seq_zone=seq_zone,
                numeric=numeric,
                pitcher=pitcher,
                stand=stand,
                p_throws=p_throws,
            )

            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            pred = torch.argmax(logits, dim=1)

            k = min(3, logits.size(1))
            top3 = torch.topk(logits, k=k, dim=1).indices

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_top3.extend(top3.cpu().numpy().tolist())

    avg_loss = total_loss / total_count
    acc = accuracy_score(y_true, y_pred)
    top3_acc = topk_accuracy_score(y_true, y_top3)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    if not return_predictions:
        return metrics

    pred_names = [idx_to_target[i] for i in y_pred]
    true_names = [idx_to_target[i] for i in y_true]
    top3_names = [
        [idx_to_target[i] for i in pred_list]
        for pred_list in y_top3
    ]

    return metrics, true_names, pred_names, top3_names


# =========================
# main
# =========================

def main():
    total_start_time = time.perf_counter()

    set_seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = [
        SEQ_PITCH_COL,
        SEQ_ZONE_COL,
        TARGET_COL,
        PITCHER_COL,
        STAND_COL,
        THROWS_COL,
    ] + NUMERIC_COLS

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    sort_cols = [
        c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # 시간 순서 기반 분할
    n = len(df)
    train_end = int(n * 0.7)
    valid_end = int(n * 0.8)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    print("===== 실행 환경 =====")
    print("experiment:", EXPERIMENT_NAME)
    print("python:", platform.python_version())
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())

    print("\n===== 데이터 분할 =====")
    print("전체:", df.shape)
    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    # target mapping은 train 기준
    target_classes = sorted(train_df[TARGET_COL].dropna().unique().tolist())
    target_to_idx = {label: idx for idx, label in enumerate(target_classes)}
    idx_to_target = {idx: label for label, idx in target_to_idx.items()}

    # train에 없는 target이 valid/test에 있으면 제거
    valid_df = valid_df[valid_df[TARGET_COL].isin(target_to_idx)].copy()
    test_df = test_df[test_df[TARGET_COL].isin(target_to_idx)].copy()

    print("\n===== 클래스 목록 =====")
    print(target_classes)
    print("클래스 수:", len(target_classes))

    print("\n===== Train target 분포 =====")
    print(train_df[TARGET_COL].value_counts())

    print("\n===== Valid target 분포 =====")
    print(valid_df[TARGET_COL].value_counts())

    print("\n===== Test target 분포 =====")
    print(test_df[TARGET_COL].value_counts())

    # vocab은 train 기준으로만 생성
    pitch_vocab = build_vocab_from_sequences(
        train_df[SEQ_PITCH_COL],
        special_tokens=["PAD", "UNK"],
    )

    zone_vocab = build_vocab_from_sequences(
        train_df[SEQ_ZONE_COL],
        special_tokens=["ZPAD", "ZUNK"],
    )

    pitcher_vocab = build_vocab_from_column(
        train_df[PITCHER_COL],
        special_tokens=["UNK"],
    )

    stand_vocab = build_vocab_from_column(
        train_df[STAND_COL],
        special_tokens=["UNK"],
    )

    throws_vocab = build_vocab_from_column(
        train_df[THROWS_COL],
        special_tokens=["UNK"],
    )

    print("\n===== Vocab 크기 =====")
    print("pitch_vocab  :", len(pitch_vocab))
    print("zone_vocab   :", len(zone_vocab))
    print("pitcher_vocab:", len(pitcher_vocab))
    print("stand_vocab  :", len(stand_vocab))
    print("throws_vocab :", len(throws_vocab))

    # numeric scaling
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_COLS].astype(float).values)

    train_dataset = PitchSequenceDataset(
        train_df,
        pitch_vocab,
        zone_vocab,
        pitcher_vocab,
        stand_vocab,
        throws_vocab,
        target_to_idx,
        scaler,
        max_len=5,
    )

    valid_dataset = PitchSequenceDataset(
        valid_df,
        pitch_vocab,
        zone_vocab,
        pitcher_vocab,
        stand_vocab,
        throws_vocab,
        target_to_idx,
        scaler,
        max_len=5,
    )

    test_dataset = PitchSequenceDataset(
        test_df,
        pitch_vocab,
        zone_vocab,
        pitcher_vocab,
        stand_vocab,
        throws_vocab,
        target_to_idx,
        scaler,
        max_len=5,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n사용 device:", device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = LSTMPitchModel(
        num_pitch_tokens=len(pitch_vocab),
        num_zone_tokens=len(zone_vocab),
        num_pitchers=len(pitcher_vocab),
        num_stands=len(stand_vocab),
        num_throws=len(throws_vocab),
        num_numeric_features=len(NUMERIC_COLS),
        num_classes=len(target_classes),
    ).to(device)

    print("\n===== 모델 정보 =====")
    print("trainable parameters:", count_parameters(model))

    # class imbalance 보정
    if USE_CLASS_WEIGHTS:
        train_label_idx = train_df[TARGET_COL].map(target_to_idx)
        counts = train_label_idx.value_counts().sort_index()

        weights = []

        for i in range(len(target_classes)):
            count = counts.get(i, 1)
            weights.append(1.0 / np.sqrt(count))

        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.mean()

        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        print("\n===== Class weights =====")
        for i, w in enumerate(weights):
            print(idx_to_target[i], round(float(w), 4))

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_score = -1.0
    best_epoch = 0
    history = []

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n===== 학습 시작 =====")

    train_start_time = time.perf_counter()

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start_time = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            idx_to_target=idx_to_target,
        )

        epoch_time = time.perf_counter() - epoch_start_time
        elapsed_time = time.perf_counter() - train_start_time
        avg_epoch_time = elapsed_time / epoch
        eta_time = avg_epoch_time * (N_EPOCHS - epoch)

        score = valid_metrics[BEST_METRIC]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_top3_accuracy": valid_metrics["top3_accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "valid_weighted_f1": valid_metrics["weighted_f1"],
                "epoch_time_sec": round(epoch_time, 4),
                "elapsed_time_sec": round(elapsed_time, 4),
                "eta_time_sec": round(eta_time, 4),
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_acc={valid_metrics['accuracy']:.4f} | "
            f"valid_top3={valid_metrics['top3_accuracy']:.4f} | "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f} | "
            f"valid_weighted_f1={valid_metrics['weighted_f1']:.4f} | "
            f"epoch_time={format_time(epoch_time)} | "
            f"elapsed={format_time(elapsed_time)} | "
            f"eta={format_time(eta_time)}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_to_idx": target_to_idx,
                    "idx_to_target": idx_to_target,
                    "pitch_vocab": pitch_vocab,
                    "zone_vocab": zone_vocab,
                    "pitcher_vocab": pitcher_vocab,
                    "stand_vocab": stand_vocab,
                    "throws_vocab": throws_vocab,
                    "numeric_cols": NUMERIC_COLS,
                    "best_metric": BEST_METRIC,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "experiment_name": EXPERIMENT_NAME,
                    "use_class_weights": USE_CLASS_WEIGHTS,
                    "model_config": {
                        "pitch_emb_dim": PITCH_EMB_DIM,
                        "zone_emb_dim": ZONE_EMB_DIM,
                        "pitcher_emb_dim": PITCHER_EMB_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                    },
                },
                MODEL_PATH,
            )

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

    print("\n===== Best model =====")
    print("best_epoch:", best_epoch)
    print(f"best_valid_{BEST_METRIC}:", best_score)
    print("모델 저장:", MODEL_PATH)
    print("학습 로그 저장:", HISTORY_PATH)

    # best model 로드 후 test 평가
    checkpoint = safe_torch_load(MODEL_PATH, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, true_names, pred_names, top3_names = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        idx_to_target=idx_to_target,
        return_predictions=True,
    )

    total_time = time.perf_counter() - total_start_time

    print("\n===== Test 결과 =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n===== 실행 시간 =====")
    print("전체 실행 시간:", format_time(total_time))
    print("전체 실행 시간_sec:", round(total_time, 2))

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "experiment_name": EXPERIMENT_NAME,
                "best_epoch": best_epoch,
                "best_metric": BEST_METRIC,
                "best_valid_score": best_score,
                "use_class_weights": USE_CLASS_WEIGHTS,
                "device": str(device),
                "n_epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "total_time_sec": round(total_time, 4),
                "trainable_parameters": count_parameters(model),
                **test_metrics,
            }
        ]
    )

    result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["true_pitch_type"] = true_names
    pred_df["pred_pitch_type"] = pred_names
    pred_df["top3_pitch_type"] = ["|".join(x) for x in top3_names]

    pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

    report_dict = classification_report(
        true_names,
        pred_names,
        labels=target_classes,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(REPORT_PATH, encoding="utf-8-sig")

    print("\n결과 저장:", RESULT_PATH)
    print("예측 결과 저장:", PRED_PATH)
    print("분류 리포트 저장:", REPORT_PATH)


if __name__ == "__main__":
    main()