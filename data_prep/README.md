# data_prep — 영상 → 이미지 데이터셋 → COLMAP → 3DGS (0단계)

공간을 찍은 **영상 하나**로 아래 두 가지를 만드는 단계다. CNN 분류기(저장소 루트)보다 앞에 오는 작업이다.

- **`gs/images`**: 3DGS(3D 복원)용 이미지 → COLMAP으로 카메라 위치 추정 → 3DGS 학습
- **`yolo/images`**: YOLO-Seg 라벨링·학습용 이미지 (공구 근접 구간)

```text
영상 ─ 01 프레임 추출 ─┬─ gs/images   ─ 02 모션 블러 검수 ─ 03 COLMAP(+04 검사) → 왜곡 보정 → 3DGS
                       └─ yolo/images ─ 02 모션 블러 검수 ─ (YOLO 라벨링 / CNN 학습으로)
```

| 파일 | 하는 일 |
|---|---|
| `01_extract_frames.py` | 영상 → 프레임. 구간마다 가장 선명한 1장 선택, 흐린 사진 1차 제거 |
| `02_qa_motion_blur.py` | 모션 블러 검출(`analyze`) → 육안 검토 → 제거 + 번호 재정리(`apply`) |
| `03_colmap_and_3dgs_commands.md` | COLMAP → (필요하면) 전체 매칭 → 왜곡 보정 → 3DGS 학습 **명령어 모음** (골라서 복붙) |
| `04_check_colmap_trajectory.py` | COLMAP 카메라 경로 그림 + 튐 비율로 결과가 맞는지 검사 |

예시 데이터(classroom_v4: 영상 18분, gs 765장 / yolo 811장)와 구축 기록(`DATASET_v4.md`)은 **팀 Google Drive**에 있다.

---

## 0. 환경 (Linux + NVIDIA GPU 기준)

| 용도 | 설치 | 검증한 버전 |
|---|---|---|
| 01, 02, 04 | `pip install opencv-python numpy` + 시스템 `ffmpeg` (`sudo apt install ffmpeg`) | Python 3.10, OpenCV 4.x |
| COLMAP | `conda create -n colmap -c conda-forge "colmap=*=*cuda*"` (GPU 버전) | COLMAP 3.13 (CUDA) |
| 3DGS | 공식 저장소 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 설치 안내 | torch 2.1.2+cu118, CUDA 11.8 |

- 3DGS 확장 모듈 빌드 중 `No module named 'pkg_resources'` 오류 → `pip install "setuptools==69.5.1"` 후 다시 빌드.
- COLMAP GitHub 릴리스에는 Windows/Mac 실행 파일만 있다. Linux는 conda 설치가 가장 간단하다.

## 1. 촬영 체크리스트

- 가로, 1× 렌즈, HDR 끔, **녹화 전 카메라 화면의 해상도·fps 확인**
- 공간 전체를 천천히 걷기 (90° 도는 데 3~4초), 이웃 장면이 70~80% 이상 겹치게
- **천장만 / 흰 벽만 보이는 장면 금지** (COLMAP이 위치를 못 잡음)
- 공구는 주변을 돌며 근접 촬영 (물체당 1분 30초 이상, 물체끼리 비슷한 시간)
- 공구 근접 장면만 계속 찍지 말고, **방이 보이는 위치에서 공구로 다가가는 장면**을 넣기
  (classroom_v4에서 망치 근접 사진만 따로 떨어져 COLMAP이 연결을 못 한 적이 있음 → 5절 `exhaustive`로 해결)

## 2. 폴더 만들기 + 구간 확인

```bash
export DATA=~/gs_projects/mydataset          # ← 데이터셋 폴더
export VIDEO=/path/to/video.MOV              # ← 영상
mkdir -p $DATA/{gs,yolo,colmap,output}

# 영상 정보 (해상도, fps, HDR 여부 color_transfer, 길이)
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,color_transfer:format=duration -of default=nw=1 $VIDEO
# 20초 간격 썸네일 → 공구가 나오는 시간대 확인
ffmpeg -v error -i $VIDEO -vf "fps=1/20,scale=320:-2,tile=6x10" -frames:v 1 $DATA/contact_sheet.jpg
```
`color_transfer`가 `arib-std-b67`이면 HDR로 찍힌 것 → 색이 바래므로 HDR 끄고 재촬영 권장.

## 3. 프레임 추출 (`01`)

30fps 영상을 그대로 쓰면 거의 같은 사진이 너무 많아서, **일정 간격마다 가장 선명한 1장**을 고른다.

```bash
# 3DGS용: 영상 전체, 약 1.33초마다 1장 (target = 영상 길이(초) / 1.33)
python 01_extract_frames.py --video $VIDEO --out $DATA/gs/images --target 820

# YOLO용: 공구 구간(예: 650초 이후), 약 0.5초마다 1장
python 01_extract_frames.py --video $VIDEO --out $DATA/yolo/images --start 650 --target 880 --candidate_fps 10
```
| 옵션 | 의미 |
|---|---|
| `--target` | 뽑을 장수. 영상 구간을 이만큼 나눠 구간마다 1장 |
| `--candidate_fps` | 후보 프레임 수/초 (기본 6). 촘촘히 뽑을 때 올림 |
| `--start`, `--end` | 사용할 구간(초) |
| `--max_size` | 긴 변 최대 크기 (기본 1920, 0이면 원본) |

결과: `images/frame_00001.jpg ~`, 같은 폴더 위에 `extraction_log.csv` (사진 ↔ 영상 시간).
3DGS는 이웃 사진이 충분히 겹치면 되므로 간격을 넓게, YOLO는 공구를 여러 각도로 많이 확보해야 하므로 촘촘하게 뽑는다.

## 4. 모션 블러 검수 (`02`)

걸으며 찍은 영상에는 순간적으로 흔들린 사진이 섞인다. **지표로 후보를 고르고 → 사람이 보고 → 제거**한다.

```bash
# 1) 분석: 지표 계산 + 제거 후보 목록/그림 생성
python 02_qa_motion_blur.py analyze --images $DATA/gs/images --profile gs
python 02_qa_motion_blur.py analyze --images $DATA/yolo/images --profile yolo

# 2) 육안 검토: <gs|yolo>/qa/qa_flagged_*.jpg 를 보고, 지우면 안 되는 사진은
#    <gs|yolo>/qa/qa_remove_list.txt 에서 그 줄을 삭제

# 3) 적용: 목록의 사진을 images_qa_removed/ 로 옮기고(삭제 아님) 남은 사진을 frame_00001부터 재번호
python 02_qa_motion_blur.py apply --images $DATA/gs/images
python 02_qa_motion_blur.py apply --images $DATA/yolo/images
```

**흐림 판단 기준**
- 선명도 = **라플라시안 분산**: 흑백 사진에 경계선을 강조하는 필터를 걸고 그 값의 분산을 구함. 선명하면 크고, 흐리면 작다.
- **앞뒤 대비 선명도** = 이 사진의 선명도 ÷ 앞뒤 14장의 중앙값. 장면마다 무늬 양이 달라 절대값만 보면 흰 벽·회색 선반 위 물체가 흐리다고 잘못 판정되기 때문.

| profile | 제거 기준 | 이유 |
|---|---|---|
| `gs` | 앞뒤 대비 0.45 미만 **그리고** 전체 하위 15% | 보수적. 사진이 빠지면 카메라 경로가 끊겨 COLMAP이 실패할 수 있음 |
| `yolo` | 앞뒤 대비 0.45 미만, 또는 하위 4% **이면서** 앞뒤 대비 0.55 미만 | 흐린 사진은 잘못된 학습 데이터. 단 무늬 없는 배경 위 물체는 보호 |

그 밖에 노출(너무 어둡거나 밝음), 중복(직전 사진과 거의 같음)도 검사한다. 흰 벽처럼 특징점이 적은 사진은 표시만 하고 지우지 않는다.
결과: `qa/qa_report.csv`(사진별 지표), `qa/qa_mapping.csv`(새 번호 ↔ 원래 번호 ↔ 영상 시간), `extraction_log.csv` 갱신(원본은 `extraction_log_before_qa.csv`).

## 5. COLMAP → 왜곡 보정 → 3DGS (`03`)

**`03_colmap_and_3dgs_commands.md`**의 블록을 순서대로 복사해 터미널에서 실행한다.

| 블록 | 내용 | 언제 |
|---|---|---|
| 0 | 변수 설정 (`DATA`, `IMAGES`, `GS_REPO`) | 새 터미널마다 |
| A | 특징점 → 순차 매칭 → mapper → 결과 요약 | 항상 |
| B | 전체 매칭 추가 → mapper 다시 | A 결과가 여러 조각으로 나뉘었을 때 |
| C | 메인 모델(가장 큰 모델) 정하기 + 카메라 경로 검사(`04`) | 항상 |
| D | 왜곡 보정 → `gs_data/` | 항상 |
| E | 3DGS 학습 | 항상 |
| F | PSNR 확인, test 렌더링·지표 | 학습 후 |

- **왜곡 보정**: 스마트폰 렌즈는 가장자리가 휘어 보인다. 3DGS는 왜곡 없는 카메라를 가정하므로, COLMAP이 계산한 왜곡 값으로 사진을 펴서 `gs_data/`(이미지 + 카메라 정보)를 만든다.
- `sparse/0`이 항상 메인 모델은 아니다 (몇 장짜리 잔여 모델일 때가 있음). 결과 요약에서 등록 장수가 가장 많은 폴더를 고른다.
- 3DGS 통과 기준: test PSNR 22~27 dB.

## 6. COLMAP 결과 검사 (`04`)

```bash
python 04_check_colmap_trajectory.py --model $DATA/colmap/sparse_exh/1 --out $DATA/colmap/trajectory.png
```
- 위에서 본 카메라 경로 그림 + **튐 비율**(이웃 사진 사이 비정상적으로 먼 이동). 
- **튐 5% 이하 정상, 10% 이상이면 실패.** 등록률·오차가 좋아도 천장 타일 같은 반복 무늬 때문에 카메라가 엉뚱한 곳에 붙는 경우가 있어서 꼭 확인한다.

## 결과물이 쓰이는 곳

| 결과 | 다음 단계 |
|---|---|
| `yolo/images` (+ `extraction_log.csv` 시간 정보) | YOLO-Seg 라벨링·학습, CNN 학습 데이터 (`datasets/tools/`) |
| `gs_data/images` (왜곡 보정된 이미지) | 학습된 YOLO-Seg + CNN으로 검출 → 3DGS 깊이로 공구 3D 위치 계산 |
| `output/run1/point_cloud/iteration_30000/point_cloud.ply` | 3D 공간 뷰어 |

## 문제 해결

| 증상 | 해결 |
|---|---|
| `OpenBLAS Warning ... may hang` 반복 | `export OPENBLAS_NUM_THREADS=1` (03 명령어 모음 0번 블록에 포함) |
| COLMAP 모델이 여러 조각 | 03 명령어 모음의 B 블록 (전체 매칭). 그래도 안 되면 방에서 물체로 다가가는 장면 추가 촬영 |
| 흐린 사진이 너무 많이 후보로 나옴 | 무늬 없는 배경인지 `qa_flagged_*.jpg`로 확인 후 `qa_remove_list.txt` 수정 |
| `01` 실행 시 "이미 jpg가 있습니다" | 덮어쓰기 방지. 출력 폴더를 비우고 다시 실행 |
| 3DGS `CUDA out of memory` | 학습 명령에 `-r 2` (해상도 절반) |
