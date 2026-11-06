# YOLO-Seg + CNN 분류기 + 3D 변환 연동 (2단계: CNN 분류기)

전체 파이프라인(YOLO-Seg 검출 -> **CNN 분류** -> 3DGS 3D 역투영) 중 이 저장소는
**2단계 CNN 분류기**와, 1단계 YOLO-Seg 출력을 받아 3단계 3D 모듈로 넘기는
**입출력 연결부(`infer_2d.py`)**를 담당한다. YOLO-Seg 학습, 3D 역투영 구현은 범위 밖.

## 폴더 구조
```
configs/
  classes.yaml      # 클래스 정의 단일 소스 (모든 모듈이 여기만 참조)
  cnn.yaml           # 학습/평가 설정. 비교실험 시 복사해서 exp_name만 바꿔 사용
src/
  data.py            # YOLO-seg 라벨 파싱, on-the-fly crop/letterbox/마스크/증강
  model.py           # CNN 구조 정의 (custom_cnn 메인 + mobilenet_v2 베이스라인)
  viz.py             # 모든 PNG 산출물 (구조/데이터/학습곡선/필터/특징맵/예측분석)
  utils.py           # config 로딩, 시드 고정, runs/ 저장 유틸
train.py             # python train.py --config configs/cnn.yaml
eval.py              # python eval.py --config ... --ckpt ... [--yolo-weights ...]
infer_2d.py          # 이미지 폴더 -> YOLO-Seg(직접 실행 또는 JSON) -> CNN -> 오버레이 PNG + 3D용 JSON
yolo_export.py       # YOLO 담당자가 자기 환경에서 실행 (TensorFlow 불필요, 박스/마스크만 JSON으로 저장)
environment.md        # YOLO(PyTorch)+CNN(TF) 환경 충돌 조사 결과와 해결책
datasets/tools/{train,val,test}/{images,labels}/   # 데이터 (세션 분할 완료 상태로 채워 넣을 것)
runs/<exp_name>/      # 실행마다 자동 생성되는 결과물
runs/summary.csv      # 모든 실험의 핵심 지표가 한 줄씩 누적
```

## 파일별 설명
코드 파일 맨 위에는 긴 설명을 적지 않고(교수님 코드 스타일에 맞춰 짧은 한 줄 주석만 사용),
사용법과 설계 설명은 모두 여기에 모아뒀다.

### `src/utils.py`
config/classes.yaml 로딩, `runs/<exp_name>/` 생성, `runs/summary.csv` 누적 등 공통 유틸.
시드 고정은 여기서 함수로 감싸지 않고 `train.py`/`eval.py`/`infer_2d.py` 각 스크립트 맨 위에
교수님 코드와 똑같은 형태로 직접 적어둠 (`np.random.seed`+`random.seed`+`tf.random.set_seed`,
`os.environ["CUDA_VISIBLE_DEVICES"]="-1"`).

### `src/data.py`
YOLO-seg 라벨(polygon txt)을 읽어 crop을 그 자리에서 만드는 모듈.
- `build_instance_index`: train/val/test 폴더(세션 분할)를 그대로 읽어 "공구 1개 = instance"
  목록을 만든다. 빈 라벨(배경) 이미지는 `other` 클래스 랜덤 crop으로 변환.
- `expand_bbox`/`jitter_bbox`/`morph_mask`/`crop_and_letterbox`/`build_input_array`:
  bbox 여백 확장 -> (학습 시) bbox 흔들기 + 마스크 침식/팽창 -> crop -> letterbox resize ->
  입력모드(rgb/masked_rgb/rgbm) 적용까지 한 단계씩 처리하는 순수 함수들.
- `ToolCropSequence`: 위 함수들을 묶은 `tf.keras.utils.Sequence`. `train=True`면 매 epoch
  새로 증강, `train=False`면 고정 crop(재현 가능).
- `build_dataset_arrays`: eval/시각화용으로 전체 instance를 증강 없이 한 번에 메모리로 로드.

### `src/model.py`
CNN 구조 정의. `build_custom_cnn`이 메인 모델(직접 설계), `build_mobilenet_v2`는 비교용
전이학습 베이스라인. 설계 근거는 아래 "모델 설계 근거" 섹션 참고.

### `src/viz.py`
발표 슬라이드용 PNG를 만드는 함수 모음 — 모델 구조표, 클래스 분포, 샘플/증강 그리드,
학습곡선, 필터/특징맵 시각화(`deep_cnn_filterVis.py`/`deep_cnn_featureMapVis.py` 방식),
confusion matrix, 맞은/틀린 예측 그리드.

### `train.py`
```
python train.py --config configs/cnn.yaml
```
instance 목록 생성 -> 클래스분포/샘플그리드/증강비교 저장 -> 모델 생성 -> 학습 ->
`runs/<exp_name>/`에 체크포인트·history·그림 저장 -> `runs/summary.csv`에 한 줄 추가.

### `eval.py`
```
python eval.py --config configs/cnn.yaml --ckpt runs/<exp>/best.h5 --split test
python eval.py --config configs/cnn.yaml --ckpt runs/<exp>/best.h5 --split test --yolo-weights best.pt
```
`--yolo-weights` 없이는 GT polygon crop으로만 평가(`metrics_gt_<split>.json`).
지정하면 YOLO 실제 예측 박스/마스크로 crop해 end-to-end 평가도 추가로 수행
(`metrics_e2e_<split>.json`) — 이때 예측 박스가 GT와 IoU로 매칭 안 되면 "오검출"로 보고
정답을 `other`로 간주해 CNN이 이를 걸러내는지까지 확인한다. 필터/특징맵 시각화와
confusion matrix, 맞은/틀린 예측 그리드, 오분류 CSV는 매번 같이 생성됨.

### `yolo_export.py` (YOLO 담당자용, TensorFlow 불필요)
```
python yolo_export.py --source image_folder --weights best.pt --out yolo_results
```
YOLO-Seg 담당자가 **자기 ultralytics 환경에서만** 돌리는 스크립트. CNN 분류는 하지 않고
박스/마스크/conf만 이미지별 JSON으로 저장한다. 이 PC에 TF+ultralytics를 같이 설치할 필요
없이, 이 JSON 폴더만 넘겨받으면 CNN 쪽(`infer_2d.py --yolo-json-dir`)에서 이어받아 처리
가능 — `environment.md`의 환경 충돌 문제를 완전히 피해가는 역할 분담 방식.

### `infer_2d.py`
```
# 이 PC에서 YOLO까지 직접 돌릴 때 (ultralytics 설치 필요)
python infer_2d.py --source image_folder --yolo-weights best.pt \
                    --config configs/cnn.yaml --ckpt runs/<exp>/best.h5 --out runs/<exp>/demo

# YOLO 담당자가 yolo_export.py로 미리 뽑아준 JSON을 받아서 처리할 때 (ultralytics 불필요)
python infer_2d.py --source image_folder --yolo-json-dir yolo_results \
                    --config configs/cnn.yaml --ckpt runs/<exp>/best.h5 --out runs/<exp>/demo
```
`--yolo-weights`/`--yolo-json-dir` 둘 중 하나만 지정한다. 이미지 폴더의 각 이미지에 대해
박스/마스크를 얻고(직접 YOLO를 돌리거나, JSON을 읽거나), CNN으로 4클래스 확률을 계산해
`out/overlays/`에 시각화 PNG를, `out/results/<파일명>.json`(+`out/results.jsonl`)에 3D
모듈용 결과를 저장한다. 합성 데이터와 실제 다운로드 데이터 양쪽에서 `--yolo-json-dir` 경로를
실제로 실행해 검증 완료(`environment.md` 참고). JSON 한 건의 형식:
```json
{
  "image": "frame_0001.jpg",
  "image_wh": [1887, 1059],
  "detections": [{
    "det_id": 0,
    "bbox_xyxy": [x1, y1, x2, y2],
    "mask_polygon": [[x, y], ...],
    "yolo_conf": 0.87,
    "cls_probs": {"drill": 0.91, "hammer": 0.03, "clamp": 0.02, "other": 0.04},
    "pred_class": "drill",
    "pred_conf": 0.91
  }]
}
```
3D 모듈은 `mask_polygon` 픽셀로 depth를 샘플링해 3D 포인트를 얻고, `cls_probs`를 다중 시점
가중 투표에 쓰면 된다. `pred_class=="other"`인 검출은 3D 모듈에서 걸러내는 걸 권장.

## 빠른 시작
```powershell
# 1) datasets/tools/{train,val,test}/{images,labels}/ 에 데이터를 채워 넣는다
#    (세션 A·B·C->train, D->val, E->test. 코드가 알아서 재분할하지 않음)
python train.py --config configs/cnn.yaml
python eval.py  --config configs/cnn.yaml --ckpt runs/cnn_rgb128_customcnn/best.h5 --split test
# YOLO 가중치가 준비되면 end-to-end(실전) 평가도:
python eval.py  --config configs/cnn.yaml --ckpt runs/cnn_rgb128_customcnn/best.h5 \
                 --split test --yolo-weights path/to/yolo_seg_best.pt
# 시연:
python infer_2d.py --source path/to/image_folder --yolo-weights path/to/yolo_seg_best.pt \
                    --config configs/cnn.yaml --ckpt runs/cnn_rgb128_customcnn/best.h5 \
                    --out runs/cnn_rgb128_customcnn/demo
```
실행 환경은 `environment.md` 참고 (TensorFlow와 ultralytics를 같이 쓸 수 있는 환경이
현재 PC에 없어서 새 conda 환경 생성을 권장함 — `train.py`/`eval.py`(GT 평가)는 지금의
`pmi` 환경(TF 2.10)으로 바로 실행 가능, `--yolo-weights`를 쓰는 부분과 `infer_2d.py`는
ultralytics가 필요).

## 데이터 파이프라인: 왜 별도 crop 데이터셋을 만들지 않았나
`datasets/tools/`는 YOLO-Seg와 완전히 같은 원본 이미지+polygon 라벨을 공유한다.
`src/data.py`가 학습/평가 때마다 그 자리에서 bbox+polygon으로 crop을 만든다
(`build_instance_index` -> `ToolCropSequence`). 장점:
- YOLO 라벨만 갱신하면 CNN 쪽도 자동으로 최신 데이터를 쓴다 (crop 이미지를 손으로 다시 안 만듦).
- 매 epoch마다 bbox jitter / mask 침식·팽창을 **원본 해상도에서** 새로 적용할 수 있어
  "실제 YOLO 예측이 약간 어긋나 있는 상황"에 강인하게 학습된다 (고정된 crop PNG로는 불가능).

`other` 클래스는 라벨 txt가 비어 있는(배경) 프레임에서 랜덤 crop해서 만든다. 이때는 실제
polygon이 없으므로 **마스크를 crop 영역 전체(1로 채움)로 대체**한다 — 즉 `masked_rgb`/`rgbm`
모드에서 other 샘플의 마스크 채널은 "진짜 세그멘테이션"이 아니라는 한계가 있음 (알려진 한계,
아래 참고).

## 클래스 설계: drill / hammer / clamp / other
`configs/classes.yaml` 참고. YOLO-Seg가 class-agnostic(nc=1)이라 실제로는 공구가 아닌
영역(손, 선반, 그림자 등)도 박스를 치는 경우가 생기므로, 공구 3종 + **other**로 4클래스를
두어 CNN이 이런 오검출을 3D 모듈로 넘어가기 전에 걸러내는 필터 역할을 하게 했다.

## 모델 설계 근거 (`src/model.py`)
메인 모델은 **직접 설계한 custom_cnn** (`configs/cnn.yaml`의 `model.custom_cnn`):
- Conv 블록 3개, 필터 수 16->32->64, 커널 3x3: 128x128 입력을 64->32->16 공간크기로
  줄이며 특징을 단계적으로 추출. 교수님 코드(`deep_cnn_fashion_train.py`)의
  "Conv2D+MaxPooling2D 반복 + Flatten + Dense" 패턴을 그대로 따르되, Fashion-MNIST(6만 장)
  기준이던 필터 수(32/64/512 Dense)를 소량 데이터(크롭 수백~수천 장)에 맞춰 크게 줄였다.
- BatchNorm: 작은 배치/소량 데이터에서 학습 안정화.
- **head="gap"**: `Flatten` 대신 `GlobalAveragePooling2D`로 16x16x64=16,384차원을 64차원으로
  압축한 뒤 FC(Dense)로 보낸다. FC층(Dense(64)+Dense(4))은 명시적으로 존재하되, 전체
  파라미터 수를 약 2천 개 수준으로 작게 유지해 과적합을 억제한다. `head="flatten"`은
  "모델 변형" 비교실험용으로 config 하나로 전환 가능.
- Dropout(0.3): FC 직전에 배치.
- 비교용 베이스라인은 `model.type: mobilenet_v2` (전이학습, ImageNet 사전학습, alpha=0.35로
  경량화). `input_mode: rgbm`(4채널)과는 같이 쓸 수 없음 (사전학습 가중치가 3채널 전제) —
  `train.py`가 이 조합이면 즉시 에러를 내도록 체크함.

## 교수님 코드에서 가져온 것 / 바꾼 것 / 왜 바꿨는지
| 항목 | 교수님 코드 | 이번 프로젝트 | 왜 바꿨나 |
|---|---|---|---|
| 프레임워크/스타일 | `tensorflow.keras`, `Sequential`, `model.compile/fit` | 동일하게 유지 | 팀/수업 전체가 이 스타일로 통일돼 있어야 리뷰가 쉬움 |
| 시드 고정 | `np.random.seed`+`random.seed`+`tf.random.set_seed` (`deep_cnn_fashion_train.py`) | 동일 코드 그대로, `train.py`/`eval.py`/`infer_2d.py` 맨 위에 직접 적음 | 스크립트 하나만 보고도 흐름이 끝까지 보이게(교수님 코드처럼) 하기 위해 함수로 숨기지 않음 |
| CPU 강제 | `os.environ["CUDA_VISIBLE_DEVICES"]="-1"` (`deep_cnn_fashion_train.py` 등) | 동일 코드, config의 `train.cpu_only`로 켜고 끄게만 함 | 이번 PC는 TF/torch 둘 다 GPU가 안 잡혀 있어 당장은 사실상 항상 True로 씀 (environment.md 참고) |
| 모델 구조 | Fashion-MNIST용 Conv+Pool 반복, 필터 32/64, Dense 512 | 그대로 쓰지 않고 필터 16/32/64, GAP+Dense(64)로 재설계 | 데이터가 훨씬 적고(수만 장 vs 수백~수천 장), 클래스도 10->4개라 그대로 쓰면 과적합이 거의 확실함 |
| 필터/특징맵 시각화 | `deep_cnn_filterVis.py`, `deep_cnn_featureMapVis.py` (레이어 선택 -> get_weights/activation model) | 동일한 방식을 `src/viz.py`의 `plot_conv_filters`/`plot_feature_maps`로 함수화, 모든 conv층에 자동 적용 | 매 실험마다 레이어 인덱스를 수동으로 바꿔 코드를 고쳐 쓰지 않도록 일반화 |
| Confusion Matrix | `deep_cnn_fashion_train_confMatrix.py` (`sklearn.metrics.confusion_matrix`, 클래스별 비율 정규화) | 동일 라이브러리 사용, 정규화 confusion matrix + precision/recall/f1까지 확장 | 평가 기준표가 P/R/F1을 요구함 |
| 데이터 로딩 | `pd.read_csv` 등 고정 CSV/폴더를 한 번에 로드 | YOLO-seg 라벨을 즉석에서 파싱해 crop하는 `Sequence` 직접 구현 | 교수님 예제엔 "검출 영역 crop + bbox/mask 증강"에 해당하는 코드가 없어 이 부분만 새로 설계 |
| train/test 분할 | `train_test_split`로 랜덤 분할 (`deep_1st_holdout.py`) | **사용하지 않음** — 폴더(session) 분할을 그대로 사용 | 요구사항에 "코드 안에서 랜덤 재분할 금지"가 명시돼 있음 (연속 프레임 섞임으로 인한 성능 부풀림 방지) |
| 모델 저장/로딩 | `model.save('my_model.h5')` / `load_model` | 동일 | 바꿀 이유 없음 |

## 비교 실험(추가 시도) 하는 법
`configs/cnn.yaml`을 복사해 `exp_name`과 비교하고 싶은 값만 바꾼다 (예: `input_mode`,
`model.custom_cnn.head`, `augment.enable`, `data.input_size`). `train.py`/`eval.py`를
그 config로 실행하면 `runs/<exp_name>/`에 결과가 쌓이고, **`runs/summary.csv`에 실험별로
한 줄씩 자동 누적**되므로 발표 때 이 CSV를 표로 바로 붙이면 된다.

## 평가 기준표 항목과 산출물 매핑
| 평가 항목 | 산출물 |
|---|---|
| 분류기 구조 설계 | `runs/<exp>/model_summary.txt`, `layer_table.csv`, `figures/model_diagram.png`(graphviz 있을 때) |
| 데이터 획득·가공 | `figures/class_distribution.png`, `figures/sample_grid_{train,val}.png`, `figures/augmentation_before_after.png` |
| 학습 분석 | `config_used.yaml`(compile/fit 설정), `figures/training_curves.png`, `figures/filters_*.png`, `figures/feature_maps.png`, `metrics_gt_test.json`의 accuracy |
| 예측 분석 | `figures/pred_correct_*.png`, `figures/pred_wrong_*.png`, `figures/confusion_matrix_*.png`, `metrics_*.json`의 per_class P/R/F1 |
| 시연 | `infer_2d.py` 결과 (`demo/overlays/*.png`, `demo/results.jsonl`) |
| 강점·약점 | `misclassified_<split>_<tag>.csv` (오분류 사례 + 원인 메모 칸, e2e 평가 시 `yolo_gt_matched` 컬럼으로 "YOLO 오검출을 other로 거른 사례" 식별 가능) |
| 추가 시도·시행착오 | `runs/summary.csv` (실험별 비교표, 자동 누적) |

## 알려진 한계 / TODO
- `other` 클래스 샘플은 실제 세그멘테이션이 없어 마스크 채널이 "crop 전체=1"로 더미 처리됨
  (`src/data.py: crop_and_letterbox`). `masked_rgb`/`rgbm` 모드 비교실험 결과 해석 시 염두에 둘 것.
- `tf.keras.utils.plot_model` 구조도는 pydot+graphviz가 설치돼 있어야 생성됨 (`environment.md`).
- `infer_2d.py`와 `eval.py --yolo-weights`는 ultralytics가 설치된 환경이 필요하며, 이 PC에는
  아직 TF+ultralytics가 공존하는 환경을 만들지 않은 상태라 **실제 실행 검증은 하지 못함**
  (`train.py`/`eval.py`의 GT-crop 경로는 합성 데이터로 end-to-end 검증 완료). YOLO 가중치가
  준비되면 `environment.md`의 안내대로 환경을 만든 뒤 가장 먼저 `infer_2d.py`부터 검증할 것.
- GPU 미설정 (CPU로만 동작) — 소량 데이터라 현재는 문제 없음, 필요해지면 `environment.md`
  "GPU를 나중에 쓰고 싶다면" 참고.
