# 실행 환경 (YOLO + CNN 공존 문제와 해결책)

## 문제: 조사 결과, 이 컴퓨터에 "YOLO(ultralytics, PyTorch)"와 "CNN(TensorFlow/Keras)"를
둘 다 쓸 수 있는 conda 환경이 **하나도 없었다.**

실제로 확인한 내용 (2026-11-06 기준, `C:\Users\user\anaconda3\envs\`):

| 환경 | Python | TensorFlow | PyTorch | ultralytics | 비고 |
|---|---|---|---|---|---|
| `pmi` | 3.10.21 | **2.10.0** (Keras 2.10.0) | 없음 | 없음 | numpy 1.23.5, opencv 4.5.5, sklearn 1.7.2, matplotlib 3.10.9 — 교수님 코드와 같은 TF 버전대 |
| `opencv_yolo` | 3.11.16 | 없음 | 2.14.0+cpu | 8.4.138 | |
| `team1` | 3.10.20 | 없음 | 2.12.0+cpu | 8.4.69 | |
| `cv` | 3.11.16 | 없음 | 없음 | 없음 | opencv만 있음 |

GPU는 물리적으로 있음(`nvidia-smi` 정상 동작, 드라이버 616.92) **but** 두 프레임워크 모두
현재 CUDA 런타임 라이브러리가 안 잡혀서(`cudart64_110.dll` 없음 / torch가 애초에 CPU 전용
휠로 설치됨) **전부 CPU로만 동작 중**이다. YOLO-Seg 학습은 이번 범위 밖이고, CNN은
소량 데이터(크롭 수백~수천 장, 128x128, 파라미터 2천~2만 개 수준)라 CPU 학습으로 충분하다
(smoke test: 42장 2 epoch에 1초 미만). `infer_2d.py` 데모에서 YOLO 추론도 이미지 1장당
CPU로 수백 ms면 충분하므로, **GPU 설정은 지금 당장 블로커가 아니다.**

## 해결책: 새 conda 환경 하나에 TensorFlow + ultralytics(torch)를 같이 설치

기존 3개 환경(`pmi`, `opencv_yolo`, `team1`)은 각각 다른 용도로 쓰고 있을 가능성이 높아
건드리지 않고, **이번 프로젝트 전용 새 환경**을 만드는 걸 권장한다. 이유:
- `pmi`의 TF 2.10은 numpy 1.23.5 / protobuf 3.19.6에 강하게 묶여 있어 최신 ultralytics를
  얹으면 의존성이 깨질 위험이 있다.
- `opencv_yolo`/`team1`은 이미 다른 작업에 쓰이고 있을 수 있는 환경이라 TF를 얹었다가
  망가뜨리면 되돌리기 번거롭다.
- 새 환경은 깨지면 그냥 지우고 다시 만들면 되므로 가장 안전하다.

### 설치 명령 (한 번만 실행)
```powershell
conda create -n toolcls python=3.10 -y
conda activate toolcls
pip install tensorflow==2.16.1 ultralytics opencv-python scikit-learn matplotlib pyyaml pandas pydot
```
- `tensorflow==2.16.1` : Python 3.10 CPU 휠이 안정적이고 numpy>=1.26, protobuf 최신판과도
  충돌 없이 ultralytics와 같이 설치 가능함 (교수님 코드의 `tensorflow.keras` API는
  2.10과 2.16 사이에서 동일하게 동작 — `Sequential`, `model.compile/fit`, `load_model` 등
  이번 코드에서 쓰는 API는 변경되지 않았음).
- `pydot` : `tf.keras.utils.plot_model`로 구조도 PNG를 뽑기 위함. Graphviz도 별도 설치 필요
  (`conda install -c conda-forge graphviz` 또는 https://graphviz.org/download/ 설치 후 PATH 등록).
  없어도 train.py/eval.py는 정상 동작하고, 구조도 PNG만 생략됨(콘솔에 안내 메시지 출력).
- GPU를 나중에 쓰고 싶다면: CUDA 11.8 + cuDNN 8.6을 설치하고
  `pip install tensorflow[and-cuda]==2.16.1` 로 교체하거나, torch는
  `pip install torch --index-url https://download.pytorch.org/whl/cu118` 로 재설치.
  (지금은 범위 밖이라 생략.)

### 검증
```powershell
python -c "import tensorflow as tf, ultralytics; print(tf.__version__, ultralytics.__version__)"
```
둘 다 에러 없이 import되면 끝. 이후 모든 스크립트(`train.py`, `eval.py`, `infer_2d.py`)는
이 `toolcls` 환경 하나로 실행하면 된다.

## 실제로 채택한 방식: 환경을 합치지 않고 역할로 나눔
`toolcls`를 만드는 대신, 팀 역할 분담 자체를 환경 경계와 맞췄다 — YOLO-Seg 담당자는
자기 ultralytics 환경에서 `yolo_export.py`만 돌려 박스/마스크를 JSON으로 저장하고,
CNN 담당자(이 저장소)는 그 JSON을 `infer_2d.py --yolo-json-dir`로 읽어 CNN 분류 +
3D용 JSON 생성까지 처리한다. TensorFlow와 ultralytics를 같은 환경에 설치할 필요가
아예 없어지므로 `toolcls` 환경 구성보다 더 간단하고, 각자 자기 환경을 안 건드려도 된다.
`infer_2d.py --yolo-weights`(YOLO까지 이 PC에서 직접 돌리는 모드)는 여전히 지원하지만,
지금은 `--yolo-json-dir` 모드가 기본 워크플로우.

## GPU를 쓰고 싶어질 때 (지금은 보류, 참고용)
`nvidia-smi` 확인 결과 드라이버 616.92(CUDA 13.4까지 지원)로 최신이라 GPU 자체는 문제없음.
다만 TensorFlow와 PyTorch(YOLO)의 사정이 다르다.

- **PyTorch/ultralytics**: 비교적 쉬움. 지금 `opencv_yolo`/`team1`의 torch가 CPU 전용 휠이라
  안 잡히는 것뿐이므로 `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  (드라이버가 지원하는 CUDA 버전에 맞는 cuXXX 선택) 로 재설치하면 바로 GPU를 씀.
- **TensorFlow**: 더 까다로움. TF 2.11부터 Windows 네이티브 GPU 지원이 끊겼다. 두 가지 길:
  1. TF==2.10.x 유지 + **CUDA Toolkit 11.2 + cuDNN 8.1 수동 설치**(cuDNN은 NVIDIA 개발자
     계정 로그인 후 다운로드, DLL을 CUDA 설치 경로의 bin에 복사 후 PATH 등록 필요).
     `requirements.txt`에서 권장한 `tensorflow==2.16.1`을 쓸 경우 이 경로는 해당 안 됨
     (2.16은 애초에 Windows 네이티브 GPU를 지원하지 않음).
  2. WSL2 위에 Linux용 최신 TensorFlow(`pip install tensorflow[and-cuda]`)를 설치 — 최신
     TF 버전을 그대로 쓰면서 GPU도 쓰고 싶다면 이쪽이 더 안정적.
- 지금 데이터 규모(crop 수백~수천 장, 128x128, 파라미터 2천~수십만 개)에서는 CPU로도
  충분히 빠르다 (smoke test: 42장 2 epoch < 1초). GPU는 "있으면 좋지만 지금 당장 필요하진
  않음" 수준 — 데이터가 훨씬 늘어나거나 mobilenet_v2 베이스라인 학습(freeze_base: false로
  전체 파인튜닝)처럼 무거운 실험을 돌릴 때 고려하면 됨.

## 실제 동작 확인
- `train.py` + `eval.py`(GT crop) 파이프라인은 `pmi` 환경(TF 2.10)에서 **실제 데이터**로
  end-to-end 검증 완료. Roboflow Universe의 공개 공구 세그멘테이션 데이터셋
  (`tool-segmentation-kqxqr`, CC BY 4.0)을 drill/hammer/clamp(wrench로 대체)/other
  (hand·gun·배경크롭)로 변환해 train 1,820장/val 124장/test 39장으로 받아 학습 ->
  test accuracy 61.3%, macro F1 0.42 (클래스 불균형이 큰 대체 데이터라 수치 자체보다
  "파이프라인이 실제 사진에서 작동한다"는 데 의미, 최종 제출용 데이터는 아님).
  크롭/마스크/증강/필터·특징맵 시각화/혼동행렬/오분류 CSV 전부 정상 생성 확인.
- `infer_2d.py --yolo-json-dir` 경로(ultralytics 불필요)도 실제 test 이미지 + GT 라벨로
  만든 가짜 YOLO 출력 JSON으로 end-to-end 검증 완료 — 오버레이 PNG, 3D용 JSON 둘 다 정상.
- `infer_2d.py --yolo-weights`(이 PC에서 YOLO까지 직접 돌리는 모드)와
  `eval.py --yolo-weights`(end-to-end 평가)는 ultralytics가 설치된 환경이 없어
  아직 실제 실행 검증은 못 했음. 다만 위 두 모드와 크롭/CNN 로직을 100% 공유하므로
  (`get_detections()`에서 입력 소스만 다름) 리스크는 낮다고 판단.
