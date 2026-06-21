# MLB Statcast 데이터를 활용한 LSTM 기반 구종·위치 추천 시스템

## 1. 프로젝트 개요

본 프로젝트는 MLB Statcast pitch-level 데이터를 활용하여 야구 경기 상황에서 다음 투구의 **구종과 위치를 추천하는 웹 기반 프로토타입**을 구현한 딥러닝 실습 프로젝트이다.

야구의 볼배합은 이전 투구 흐름, 볼카운트, 주자 상황, 투수와 타자의 좌우 관계 등에 영향을 받는 순차적 의사결정 문제이다. 본 프로젝트에서는 이를 시계열 예측 문제로 보고, 이전 투구 시퀀스를 입력으로 받아 다음 구종 후보를 예측하는 **Long Short-Term Memory (LSTM)** 모델을 학습하였다.

최종 시스템은 단순히 다음 구종 하나를 예측하는 데서 끝나지 않고, LSTM이 생성한 구종 후보에 대해 Statcast의 `delta_run_exp` 값을 활용한 경험적 랭킹 모듈을 결합하여 **구종과 Statcast zone 조합**을 추천한다.

## 2. 핵심 아이디어

본 프로젝트의 최종 추천 과정은 다음과 같다.

1. 사용자가 현재 경기 상황과 이전 투구 이력을 입력한다.
2. LSTM 모델이 다음에 나올 가능성이 높은 구종 Top-3 후보를 예측한다.
3. 각 구종 후보에 대해 Statcast zone별 `delta_run_exp` 경험적 평균을 계산한다.
4. 투수에게 유리한 방향, 즉 예상 기대득점 변화가 낮은 구종-zone 조합을 추천한다.
5. 사용자가 직접 선택한 구종-zone 조합과 모델 추천 결과를 비교할 수 있다.

## 3. 사용 데이터

데이터는 `pybaseball` 라이브러리의 Statcast 수집 기능을 이용해 수집하였다.

- 데이터 출처: MLB Statcast
- 수집 도구: `pybaseball.statcast`
- 수집 기간: 2023-03-01부터 2025-09-27까지
- 원본 데이터 크기: 약 2,286,376개 투구
- 전처리 후 데이터 크기: 약 2,231,601개 투구
- LSTM 학습용 시퀀스 데이터: 약 1,657,869개 샘플
- `delta_run_exp` 기반 랭킹용 데이터: 약 1,657,848개 샘플

대용량 원본 데이터 파일은 저장소에 포함하지 않을 수 있으며, `src/download_statcast.py`를 실행하여 동일한 조건으로 다시 수집할 수 있다.

## 4. 주요 기능

### 4.1 데이터 수집 및 전처리

- `pybaseball`을 이용한 Statcast pitch-level 데이터 수집
- `pitch_type` 결측 제거
- 희귀 구종을 `OTHER` 클래스로 통합
- 주자 상황, 볼카운트, 아웃카운트, 이닝, 투수/타자 좌우 정보 정리
- 같은 타석 내 이전 투구 구종과 zone을 시퀀스 형태로 구성

### 4.2 Baseline 모델 비교

딥러닝 모델의 성능을 평가하기 위해 여러 Baseline 모델을 구현하였다.

- Global Majority
- Count/Base Majority
- Pitcher Majority
- Pitcher + Count Majority
- Previous Pitch Markov
- Previous Pitch + Count

### 4.3 LSTM 기반 구종 후보 생성

이전 5개 투구의 구종과 zone을 입력 시퀀스로 사용하고, 현재 경기 상황 정보를 함께 입력하여 다음 구종을 예측한다.

사용한 주요 입력 정보는 다음과 같다.

- 이전 구종 시퀀스
- 이전 Statcast zone 시퀀스
- 볼카운트
- 아웃카운트
- 이닝
- 주자 상황
- 투수 ID
- 투수 손
- 타자 좌우

### 4.4 Transformer Encoder 비교 실험

LSTM과 비교하기 위해 Transformer Encoder 기반 모델도 구현하였다. 동일한 데이터 분할과 평가 지표를 사용하여 두 모델을 비교하였다.

### 4.5 `delta_run_exp` 기반 구종-zone 랭킹

구종 예측 결과를 실제 추천 시스템으로 확장하기 위해 `delta_run_exp` 기반 경험적 랭킹 모듈을 구현하였다.

랭킹은 다음과 같은 계층적 기준을 사용한다.

1. 같은 투수 + 같은 볼카운트 + 같은 구종 + 같은 zone
2. 같은 투수 + 같은 구종 + 같은 zone
3. 같은 볼카운트 + 같은 구종 + 같은 zone
4. 같은 구종 + 같은 zone
5. 같은 구종
6. 전체 평균

이를 통해 데이터가 충분한 경우에는 더 구체적인 조건을 사용하고, 표본 수가 부족한 경우에는 더 일반적인 기준으로 fallback하도록 구성하였다.

### 4.6 웹 데모

FastAPI 백엔드와 HTML/CSS/JavaScript 프론트엔드를 구현하여 사용자가 직접 경기 상황을 입력하고 추천 결과를 확인할 수 있도록 하였다.

웹 데모 기능은 다음과 같다.

- 현재 경기 상황 입력
- 이전 투구 구종 및 위치 입력
- 후보 구종 제한
- LSTM Top-3 구종 후보 표시
- 최종 구종-zone 추천 표시
- 추천 위치 시각화
- 직접 선택한 구종-zone과 모델 추천 결과 비교
- 추천 근거와 표본 수 기반 신뢰도 표시

## 5. 프로젝트 구조

```text
pitch_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── lstm_weighted_macro_best.pt
│   ├── lstm_serving_artifact.pt
│   └── outcome_ranker_artifact.pkl
│
├── outputs/
│   ├── baseline_results.csv
│   ├── lstm_weighted_macro_result.csv
│   ├── lstm_weighted_macro_predictions.csv
│   ├── transformer_weighted_macro_result.csv
│   ├── transformer_weighted_macro_predictions.csv
│   ├── outcome_empirical_eval.csv
│   └── outcome_empirical_recommendations.csv
│
├── src/
│   ├── download_statcast.py
│   ├── check_csv.py
│   ├── check_data_summary.py
│   ├── preprocess.py
│   ├── make_sequence.py
│   ├── baseline.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── make_outcome_sequence.py
│   ├── outcome_empirical_recommender.py
│   └── export_serving_artifacts.py
│
├── web/
│   ├── backend/
│   │   └── app.py
│   └── frontend/
│       └── index.html
│
├── requirements.txt
└── README.md
```

## 6. 실행 환경

개발 및 실험 환경은 다음과 같다.

- Python 3.12
- PyTorch
- pandas
- scikit-learn
- pybaseball
- FastAPI
- Uvicorn

필요 패키지는 다음 명령어로 설치할 수 있다.

```powershell
pip install -r requirements.txt
```

가상환경을 사용하는 경우 예시는 다음과 같다.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 7. 전체 실행 순서

프로젝트 루트 디렉터리에서 다음 순서로 실행한다.

```powershell
python -u src/download_statcast.py
python -u src/check_csv.py
python -u src/check_data_summary.py
python -u src/preprocess.py
python -u src/make_sequence.py
python -u src/baseline.py
python -u src/train_lstm.py
python -u src/train_transformer.py
python -u src/make_outcome_sequence.py
python -u src/outcome_empirical_recommender.py
python -u src/export_serving_artifacts.py
```

각 단계의 역할은 다음과 같다.

| 단계 | 파일                               | 역할                               |
| ---- | ---------------------------------- | ---------------------------------- |
| 1    | `download_statcast.py`             | Statcast 데이터 수집               |
| 2    | `check_csv.py`                     | 필수 컬럼 확인                     |
| 3    | `check_data_summary.py`            | 데이터 요약 통계 확인              |
| 4    | `preprocess.py`                    | 결측 제거 및 기본 전처리           |
| 5    | `make_sequence.py`                 | LSTM 학습용 시퀀스 데이터 생성     |
| 6    | `baseline.py`                      | Baseline 모델 평가                 |
| 7    | `train_lstm.py`                    | LSTM 모델 학습                     |
| 8    | `train_transformer.py`             | Transformer Encoder 비교 실험      |
| 9    | `make_outcome_sequence.py`         | `delta_run_exp` 랭킹용 데이터 생성 |
| 10   | `outcome_empirical_recommender.py` | 경험적 구종-zone 랭킹 실험         |
| 11   | `export_serving_artifacts.py`      | 웹서비스용 모델 artifact 생성      |

## 8. 웹 데모 실행

모델 artifact 생성 후 다음 명령어로 웹 서버를 실행한다.

```powershell
uvicorn web.backend.app:app --reload --host 127.0.0.1 --port 8000
```

브라우저에서 다음 주소로 접속한다.

```text
http://127.0.0.1:8000
```

확인용 엔드포인트는 다음과 같다.

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## 9. 주요 실험 결과

### 9.1 모델 성능 비교

| 모델                     | Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 | 해석                           |
| ------------------------ | -------: | -------------: | -------: | ----------: | ------------------------------ |
| Pitcher + Count Baseline |   0.4060 |         0.8005 |   0.3329 |      0.3808 | Top-1 기준 가장 높은 Baseline  |
| LSTM                     |   0.3896 |         0.8353 |   0.3972 |      0.3890 | Top-3와 Macro F1에서 가장 우수 |
| Transformer Encoder      |   0.3722 |         0.8293 |   0.3896 |      0.3691 | LSTM 대비 낮은 성능            |

Pitcher + Count Baseline은 단순 최빈값 기반 방식임에도 Top-1 Accuracy가 가장 높았다. 그러나 추천 시스템에서는 하나의 정답을 맞히는 것뿐 아니라, 가능한 후보 구종을 안정적으로 제시하는 것이 중요하다.

LSTM은 Top-1 Accuracy에서는 Baseline보다 낮았지만, Top-3 Accuracy와 Macro F1에서 더 높은 성능을 보였다. 따라서 최종 시스템에서는 LSTM을 단일 정답 예측기가 아니라 **구종 후보 생성기**로 사용하였다.

Transformer Encoder는 비교 모델로 실험하였으나, 본 프로젝트의 입력 시퀀스 길이가 짧은 편이기 때문에 LSTM 대비 뚜렷한 성능 우위를 보이지 않았다.

### 9.2 최종 추천 방식

최종 추천 시스템은 다음 두 단계를 결합한다.

```text
LSTM 구종 후보 생성
        ↓
delta_run_exp 기반 구종-zone 랭킹
        ↓
최종 구종·위치 추천
```

이 구조를 통해 다음 투구의 구종 후보를 예측하는 것에서 나아가, 과거 기대득점 변화 경향을 바탕으로 투수에게 유리한 위치까지 함께 추천할 수 있도록 하였다.

## 10. 추천 결과 해석

추천 결과의 `pred_delta_run_exp`는 해당 구종-zone 조합이 과거 데이터에서 보인 기대득점 변화의 경험적 추정값이다.

- 값이 낮을수록 투수에게 유리하다.
- 음수 값은 공격팀의 기대득점을 낮추는 경향을 의미한다.
- 양수 값은 공격팀에게 유리한 결과로 이어진 경향을 의미한다.

표본 수에 따른 신뢰도 기준은 다음과 같다.

| 신뢰도 | 기준                            |
| ------ | ------------------------------- |
| 높음   | 표본 수 1,000개 이상            |
| 보통   | 표본 수 100개 이상 1,000개 미만 |
| 낮음   | 표본 수 100개 미만              |

추천 근거는 다음과 같이 해석한다.

| 내부 source                 | 의미                                                |
| --------------------------- | --------------------------------------------------- |
| `pitcher+count+action+zone` | 같은 투수, 같은 볼카운트, 같은 구종, 같은 위치 기준 |
| `pitcher+action+zone`       | 같은 투수, 같은 구종, 같은 위치 기준                |
| `count+action+zone`         | 같은 볼카운트, 같은 구종, 같은 위치 기준            |
| `action+zone`               | 같은 구종, 같은 위치 기준                           |
| `action`                    | 같은 구종 기준                                      |
| `global`                    | 전체 평균 기준                                      |

## 11. 한계점

본 프로젝트는 실제 야구 경기의 모든 의사결정 요소를 반영한 완전한 최적 볼배합 시스템은 아니다. 주요 한계는 다음과 같다.

1. 추천 zone은 포수가 의도한 목표 위치가 아니라 Statcast에 기록된 실제 도착 zone을 기준으로 한다.
2. `delta_run_exp` 기반 랭킹은 과거 데이터의 경험적 경향이며, 엄밀한 인과적 효과를 의미하지 않는다.
3. 점수 차, 오늘 투구 수, 선발/구원 여부, 동일 타자와의 누적 승부 횟수 등은 현재 모델에 직접 반영하지 않았다.
4. 구속, 회전수, 무브먼트, 릴리스 포인트 등은 실제 투구 후 측정되는 값이 많기 때문에 현재 추천 입력에서는 제외하였다.
5. 타자 개인별 약점, 포수 리드 성향, 경기 운영 전략 등은 현재 모델에 포함되지 않았다.

## 12. 향후 개선 방향

향후 연구에서는 다음과 같은 방향으로 시스템을 확장할 수 있다.

1. 점수 차와 접전 여부를 반영한 상황별 볼배합 추천
2. 투수의 당일 투구 수와 피로도 반영
3. 선발/구원 여부에 따른 투구 패턴 차이 반영
4. 동일 타자와의 누적 승부 횟수 반영
5. 타자별 약점 zone 및 구종 대응 성향 반영
6. 구속, 회전수, 무브먼트 정보를 활용한 고도화 모델 설계
7. 실제 포수 사인 또는 의도 위치 데이터가 확보될 경우 목표 위치 기반 추천으로 확장
8. 강화학습 또는 오프라인 정책 평가 기반의 장기적 기대 실점 최소화 모델 연구

## 13. 결론

본 프로젝트는 MLB Statcast pitch-level 데이터를 활용하여 다음 구종 후보를 예측하고, 기대득점 변화량 기반 경험적 랭킹을 통해 구종과 위치를 함께 추천하는 하이브리드 볼배합 추천 시스템을 구현하였다.

LSTM 모델은 Top-1 Accuracy에서는 일부 Baseline보다 낮았지만, 추천 시스템 관점에서 중요한 Top-3 Accuracy와 Macro F1에서 우수한 성능을 보였다. 또한 Transformer Encoder와의 비교 실험을 통해 짧은 이전 투구 시퀀스 기반 예측에서는 LSTM이 더 안정적으로 동작함을 확인하였다.

최종적으로 FastAPI 백엔드와 HTML 프론트엔드 웹 데모를 구현하여, 사용자가 직접 현재 경기 상황과 이전 투구 이력을 입력하고 추천 결과를 확인할 수 있는 end-to-end 프로토타입을 완성하였다.
