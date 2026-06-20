from pathlib import Path
import sys
import pickle
from typing import List, Optional, Dict, Any

import numpy as np
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# train_lstm.py 안의 LSTMPitchModel 구조를 그대로 사용
from src.train_lstm import LSTMPitchModel  # noqa: E402


# ============================================================
# Artifact paths
# ============================================================

LSTM_ARTIFACT_PATH = PROJECT_ROOT / "models" / "lstm_serving_artifact.pt"
OUTCOME_ARTIFACT_PATH = PROJECT_ROOT / "models" / "outcome_ranker_artifact.pkl"

FRONTEND_PATH = PROJECT_ROOT / "web" / "frontend" / "index.html"


# ============================================================
# Request / Response schema
# ============================================================

class RecommendRequest(BaseModel):
    # 선수 정보
    pitcher: int
    batter: Optional[int] = None

    # 현재 경기 상황
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    outs_when_up: int = Field(ge=0, le=2)
    inning: int = Field(ge=1, le=20)

    runner_1b: int = Field(default=0, ge=0, le=1)
    runner_2b: int = Field(default=0, ge=0, le=1)
    runner_3b: int = Field(default=0, ge=0, le=1)

    # 좌우 정보
    stand: str = Field(default="R")      # batter side: R/L
    p_throws: str = Field(default="R")   # pitcher hand: R/L

    # 같은 타석 내 이전 투구 정보
    # 5개보다 적어도 됨. 서버에서 왼쪽 PAD 처리.
    prev_pitch_types: List[str] = Field(default_factory=list)
    prev_zones: List[str] = Field(default_factory=list)

    # 사용자가 후보 구종을 직접 제한하고 싶을 때 사용
    # 예: ["FF", "SL", "CH"]
    candidate_pitch_types: Optional[List[str]] = None

    # 최종 추천 개수
    top_n: int = Field(default=5, ge=1, le=20)


def req_to_dict(req: RecommendRequest) -> Dict[str, Any]:
    """
    Pydantic v1/v2 호환용.
    """
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()


# ============================================================
# Utility functions
# ============================================================

def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_zone(z):
    """
    입력 zone을 모델에서 쓰는 Z1, Z2 ... 형태로 통일.
    """
    if z is None:
        return "ZUNK"

    z = str(z).strip().upper()

    if z in ["", "NAN", "NONE"]:
        return "ZUNK"

    if z.startswith("Z"):
        return z

    try:
        return f"Z{int(float(z))}"
    except ValueError:
        return "ZUNK"


def left_pad(values, n, pad_token):
    values = list(values)[-n:]
    return [pad_token] * (n - len(values)) + values


def encode_sequence(tokens, vocab, max_len=5, pad_token="PAD", unk_token="UNK"):
    tokens = left_pad(tokens, max_len, pad_token)
    unk_id = vocab.get(unk_token, 1)
    return [vocab.get(token, unk_id) for token in tokens]


def encode_category(value, vocab, unk_token="UNK"):
    if value is None:
        return vocab.get(unk_token, 0)

    return vocab.get(str(value), vocab.get(unk_token, 0))


def smoothed_mean(sum_value, count_value, global_mean, alpha):
    return (sum_value + alpha * global_mean) / (count_value + alpha)


def clean_pitch_token(p):
    if p is None:
        return None

    p = str(p).strip().upper()

    if p in ["", "NAN", "NONE", "OTHER"]:
        return None

    return p


# ============================================================
# Recommendation service
# ============================================================

class PitchRecommendationService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not LSTM_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"LSTM artifact가 없습니다: {LSTM_ARTIFACT_PATH}\n"
                "먼저 python -u src/export_serving_artifacts.py 를 실행하세요."
            )

        if not OUTCOME_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Outcome artifact가 없습니다: {OUTCOME_ARTIFACT_PATH}\n"
                "먼저 python -u src/export_serving_artifacts.py 를 실행하세요."
            )

        # -------------------------
        # Load LSTM artifact
        # -------------------------
        self.lstm_artifact = safe_torch_load(LSTM_ARTIFACT_PATH, self.device)

        required_lstm_keys = [
            "model_state_dict",
            "target_to_idx",
            "idx_to_target",
            "pitch_vocab",
            "zone_vocab",
            "pitcher_vocab",
            "stand_vocab",
            "throws_vocab",
            "numeric_cols",
            "scaler_mean",
            "scaler_scale",
        ]

        missing = [k for k in required_lstm_keys if k not in self.lstm_artifact]
        if missing:
            raise ValueError(f"LSTM artifact 필수 키 누락: {missing}")

        self.pitch_vocab = self.lstm_artifact["pitch_vocab"]
        self.zone_vocab = self.lstm_artifact["zone_vocab"]
        self.pitcher_vocab = self.lstm_artifact["pitcher_vocab"]
        self.stand_vocab = self.lstm_artifact["stand_vocab"]
        self.throws_vocab = self.lstm_artifact["throws_vocab"]

        self.target_to_idx = self.lstm_artifact["target_to_idx"]
        self.idx_to_target = self.lstm_artifact["idx_to_target"]

        # torch 저장 dict에서 key가 str로 바뀌는 경우 방어
        self.idx_to_target = {
            int(k): v for k, v in self.idx_to_target.items()
        }

        self.numeric_cols = self.lstm_artifact["numeric_cols"]
        self.scaler_mean = np.array(self.lstm_artifact["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.array(self.lstm_artifact["scaler_scale"], dtype=np.float32)

        self.scaler_scale[self.scaler_scale == 0] = 1.0

        # LSTM 모델 복원
        self.model = LSTMPitchModel(
            num_pitch_tokens=len(self.pitch_vocab),
            num_zone_tokens=len(self.zone_vocab),
            num_pitchers=len(self.pitcher_vocab),
            num_stands=len(self.stand_vocab),
            num_throws=len(self.throws_vocab),
            num_numeric_features=len(self.numeric_cols),
            num_classes=len(self.target_to_idx),
        ).to(self.device)

        self.model.load_state_dict(self.lstm_artifact["model_state_dict"])
        self.model.eval()

        # -------------------------
        # Load outcome artifact
        # -------------------------
        with open(OUTCOME_ARTIFACT_PATH, "rb") as f:
            self.outcome_artifact = pickle.load(f)

        required_outcome_keys = [
            "global_mean",
            "alpha",
            "thresholds",
            "valid_zones",
            "pitcher_top_pitches",
            "global_top_pitches",
            "stats",
        ]

        missing = [k for k in required_outcome_keys if k not in self.outcome_artifact]
        if missing:
            raise ValueError(f"Outcome artifact 필수 키 누락: {missing}")

        self.valid_zones = self.outcome_artifact["valid_zones"]
        self.global_top_pitches = self.outcome_artifact["global_top_pitches"]
        self.pitcher_top_pitches = self.outcome_artifact["pitcher_top_pitches"]
        self.stats = self.outcome_artifact["stats"]

        # outcome artifact에 존재하는 실제 추천 가능 구종 목록
        self.supported_pitch_types = sorted(
            [
                key[0]
                for key in self.stats["action"].keys()
                if key and key[0] != "OTHER"
            ]
        )

        print("===== PitchRecommendationService loaded =====")
        print("device:", self.device)
        print("num_pitchers:", len(self.pitcher_vocab))
        print("valid_zones:", self.valid_zones)
        print("supported_pitch_types:", self.supported_pitch_types)

    # --------------------------------------------------------
    # LSTM Top-K pitch type prediction
    # --------------------------------------------------------
    def predict_lstm_topk(self, req: RecommendRequest, k=3):
        prev_pitch_types = []

        for p in req.prev_pitch_types:
            p = clean_pitch_token(p)
            if p is None:
                continue
            prev_pitch_types.append(p)

        prev_zones = [normalize_zone(z) for z in req.prev_zones]

        seq_pitch = encode_sequence(
            prev_pitch_types,
            self.pitch_vocab,
            max_len=5,
            pad_token="PAD",
            unk_token="UNK",
        )

        seq_zone = encode_sequence(
            prev_zones,
            self.zone_vocab,
            max_len=5,
            pad_token="ZPAD",
            unk_token="ZUNK",
        )

        numeric_values = np.array(
            [
                req.balls,
                req.strikes,
                req.outs_when_up,
                req.inning,
                req.runner_1b,
                req.runner_2b,
                req.runner_3b,
            ],
            dtype=np.float32,
        )

        numeric_values = (numeric_values - self.scaler_mean) / self.scaler_scale

        pitcher_id = encode_category(req.pitcher, self.pitcher_vocab)
        stand_id = encode_category(req.stand.upper(), self.stand_vocab)
        throws_id = encode_category(req.p_throws.upper(), self.throws_vocab)

        batch = {
            "seq_pitch": torch.tensor([seq_pitch], dtype=torch.long, device=self.device),
            "seq_zone": torch.tensor([seq_zone], dtype=torch.long, device=self.device),
            "numeric": torch.tensor(
                numeric_values.reshape(1, -1),
                dtype=torch.float32,
                device=self.device,
            ),
            "pitcher": torch.tensor([pitcher_id], dtype=torch.long, device=self.device),
            "stand": torch.tensor([stand_id], dtype=torch.long, device=self.device),
            "p_throws": torch.tensor([throws_id], dtype=torch.long, device=self.device),
        }

        with torch.no_grad():
            logits = self.model(**batch)
            probs = torch.softmax(logits, dim=1)[0]

            topk = torch.topk(
                probs,
                k=min(k, probs.shape[0]),
                dim=0,
            )

        results = []

        for idx, prob in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()):
            pitch_type = self.idx_to_target[int(idx)]

            results.append(
                {
                    "pitch_type": pitch_type,
                    "probability": float(prob),
                }
            )

        return results

    # --------------------------------------------------------
    # Outcome ranker
    # --------------------------------------------------------
    def _predict_delta(self, req: RecommendRequest, action_pitch_type: str, action_zone: str):
        thresholds = self.outcome_artifact["thresholds"]
        global_mean = float(self.outcome_artifact["global_mean"])
        alpha = float(self.outcome_artifact["alpha"])

        pitcher = int(req.pitcher)
        balls = int(req.balls)
        strikes = int(req.strikes)

        action_pitch_type = str(action_pitch_type).upper()
        action_zone = str(action_zone).upper()

        # 1. pitcher + count + action + zone
        key1 = (pitcher, balls, strikes, action_pitch_type, action_zone)
        stat = self.stats["pitcher_count_action_zone"].get(key1)

        if stat is not None and stat["count"] >= thresholds["pitcher_count_action_zone"]:
            return (
                smoothed_mean(stat["sum"], stat["count"], global_mean, alpha),
                "pitcher+count+action+zone",
                stat["count"],
            )

        # 2. pitcher + action + zone
        key2 = (pitcher, action_pitch_type, action_zone)
        stat = self.stats["pitcher_action_zone"].get(key2)

        if stat is not None and stat["count"] >= thresholds["pitcher_action_zone"]:
            return (
                smoothed_mean(stat["sum"], stat["count"], global_mean, alpha),
                "pitcher+action+zone",
                stat["count"],
            )

        # 3. count + action + zone
        key3 = (balls, strikes, action_pitch_type, action_zone)
        stat = self.stats["count_action_zone"].get(key3)

        if stat is not None and stat["count"] >= thresholds["count_action_zone"]:
            return (
                smoothed_mean(stat["sum"], stat["count"], global_mean, alpha),
                "count+action+zone",
                stat["count"],
            )

        # 4. action + zone
        key4 = (action_pitch_type, action_zone)
        stat = self.stats["action_zone"].get(key4)

        if stat is not None and stat["count"] >= thresholds["action_zone"]:
            return (
                smoothed_mean(stat["sum"], stat["count"], global_mean, alpha),
                "action+zone",
                stat["count"],
            )

        # 5. action only
        key5 = (action_pitch_type,)
        stat = self.stats["action"].get(key5)

        if stat is not None and stat["count"] >= thresholds["action"]:
            return (
                smoothed_mean(stat["sum"], stat["count"], global_mean, alpha),
                "action",
                stat["count"],
            )

        # 6. global fallback
        return global_mean, "global", 0

    # --------------------------------------------------------
    # Final recommendation
    # --------------------------------------------------------
    def build_candidate_pitches(self, req: RecommendRequest, lstm_top3):
        candidate_pitches = []

        # 0. 사용자가 후보 구종을 직접 넣은 경우: 강한 제약으로 사용
        if req.candidate_pitch_types:
            for p in req.candidate_pitch_types:
                p = clean_pitch_token(p)

                if p is None:
                    continue

                if p not in self.supported_pitch_types:
                    continue

                if p not in candidate_pitches:
                    candidate_pitches.append(p)

            # 사용자가 직접 후보를 넣은 경우에는 pitcher fallback을 추가하지 않음
            if candidate_pitches:
                return candidate_pitches

        # 1. 직접 후보가 없으면 LSTM Top-3 사용
        for item in lstm_top3:
            p = clean_pitch_token(item["pitch_type"])

            if p is None:
                continue

            if p not in self.supported_pitch_types:
                continue

            if p not in candidate_pitches:
                candidate_pitches.append(p)

        # 2. 해당 투수의 주 구종 fallback 추가
        pitcher_top = self.pitcher_top_pitches.get(int(req.pitcher), [])

        for p in pitcher_top:
            p = clean_pitch_token(p)

            if p is None:
                continue

            if p not in self.supported_pitch_types:
                continue

            if p not in candidate_pitches:
                candidate_pitches.append(p)

        # 3. 그래도 비면 전체 top pitch
        if not candidate_pitches:
            for p in self.global_top_pitches:
                p = clean_pitch_token(p)

                if p is None:
                    continue

                if p not in self.supported_pitch_types:
                    continue

                if p not in candidate_pitches:
                    candidate_pitches.append(p)

        return candidate_pitches

    def recommend(self, req: RecommendRequest):
        lstm_top3 = self.predict_lstm_topk(req, k=3)

        candidate_pitches = self.build_candidate_pitches(req, lstm_top3)

        candidates = []

        for pitch in candidate_pitches:
            for zone in self.valid_zones:
                score, source, count = self._predict_delta(req, pitch, zone)

                candidates.append(
                    {
                        "pitch_type": pitch,
                        "zone": zone,
                        "pred_delta_run_exp": float(score),
                        "source": source,
                        "source_count": int(count),
                    }
                )

        candidates = sorted(candidates, key=lambda x: x["pred_delta_run_exp"])
        recommendations = candidates[: req.top_n]

        warning_messages = []

        if len(req.prev_pitch_types) == 0:
            warning_messages.append(
                "이전 투구 정보가 없습니다. 현재 모델은 이전 투구 이력이 있는 상황에서 더 안정적으로 동작합니다."
            )

        if len(req.prev_pitch_types) != len(req.prev_zones):
            warning_messages.append(
                "prev_pitch_types와 prev_zones의 길이가 다릅니다. 부족한 부분은 PAD/ZUNK로 처리됩니다."
            )

        return {
            "input": req_to_dict(req),
            "warnings": warning_messages,
            "lstm_top3": lstm_top3,
            "candidate_pitches": candidate_pitches,
            "recommendations": recommendations,
            "notes": {
                "zone_meaning": "추천 zone은 Statcast 실제 도착 zone 기반입니다. 포수의 의도 위치와는 다를 수 있습니다.",
                "delta_run_exp_meaning": "pred_delta_run_exp가 낮을수록 투수 관점에서 유리한 과거 기대득점 변화 경향을 의미합니다.",
            },
        }

class CompareRequest(RecommendRequest):
    action_pitch_type: str
    action_zone: str

# ============================================================
# FastAPI app
# ============================================================

service = PitchRecommendationService()

app = FastAPI(
    title="Pitch Recommendation API",
    description="LSTM 기반 구종 후보 생성 + delta_run_exp 기반 zone 랭킹 추천 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH)

    return HTMLResponse(
        """
        <h1>Pitch Recommendation API</h1>
        <p>Frontend file not found. Use <a href="/docs">/docs</a> to test API.</p>
        """
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(service.device),
        "num_pitchers_lstm_vocab": len(service.pitcher_vocab),
        "num_pitchers_outcome": len(service.pitcher_top_pitches),
        "valid_zones": service.valid_zones,
        "supported_pitch_types": service.supported_pitch_types,
    }


@app.get("/meta")
def meta():
    return {
        "valid_zones": service.valid_zones,
        "supported_pitch_types": service.supported_pitch_types,
        "global_top_pitches": service.global_top_pitches,
        "supported_stand": list(service.stand_vocab.keys()),
        "supported_p_throws": list(service.throws_vocab.keys()),
        "request_example": {
            "pitcher": 622503,
            "batter": None,
            "balls": 1,
            "strikes": 2,
            "outs_when_up": 1,
            "inning": 7,
            "runner_1b": 0,
            "runner_2b": 1,
            "runner_3b": 0,
            "stand": "R",
            "p_throws": "R",
            "prev_pitch_types": ["FF", "SL", "FF", "CH", "SL"],
            "prev_zones": ["Z5", "Z13", "Z2", "Z14", "Z7"],
            "candidate_pitch_types": ["FF", "SL", "CH"],
            "top_n": 5,
        },
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    return service.recommend(req)


@app.post("/compare")
def compare(req: CompareRequest):
    pitch = clean_pitch_token(req.action_pitch_type)
    zone = normalize_zone(req.action_zone)

    if pitch is None:
        return {
            "error": "유효하지 않은 구종입니다.",
            "input_pitch_type": req.action_pitch_type,
        }

    if pitch not in service.supported_pitch_types:
        return {
            "error": "지원하지 않는 구종입니다.",
            "input_pitch_type": pitch,
            "supported_pitch_types": service.supported_pitch_types,
        }

    if zone not in service.valid_zones:
        return {
            "error": "지원하지 않는 zone입니다.",
            "input_zone": zone,
            "valid_zones": service.valid_zones,
        }

    score, source, count = service._predict_delta(req, pitch, zone)

    # 같은 상황에서 추천 결과도 같이 계산
    rec_result = service.recommend(req)
    best = rec_result["recommendations"][0] if rec_result["recommendations"] else None

    result = {
        "input": req_to_dict(req),
        "selected_action": {
            "pitch_type": pitch,
            "zone": zone,
            "pred_delta_run_exp": float(score),
            "source": source,
            "source_count": int(count),
        },
        "best_recommendation": best,
    }

    if best is not None:
        result["comparison"] = {
            "selected_minus_best": float(score - best["pred_delta_run_exp"]),
            "is_selected_better_or_equal": bool(score <= best["pred_delta_run_exp"]),
        }

    return result