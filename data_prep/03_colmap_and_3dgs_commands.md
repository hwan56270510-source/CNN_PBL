# 03. COLMAP → 왜곡 보정 → 3DGS 명령어 모음

필요한 블록만 골라 터미널에 복사해서 실행한다. 순서: **0 → A → (조각났으면 B) → C → D → E → F**

---

## 0. 변수 설정 (새 터미널마다 먼저)

```bash
export DATA=~/gs_projects/mydataset              # ← 데이터셋 폴더 (gs/images 가 있는 곳)
export IMAGES=$DATA/gs/images                     # ← 02 검수까지 끝낸 3DGS용 이미지
export GS_REPO=~/gaussian-splatting               # ← gaussian-splatting 저장소 위치
export OPENBLAS_NUM_THREADS=1                     # conda COLMAP의 OpenBLAS 경고/멈춤 방지
cd $DATA && mkdir -p colmap/sparse output
```

## A. COLMAP 기본 (특징점 → 순차 매칭 → mapper)

`conda activate colmap` 후 실행. 800장 기준 mapper까지 수십 분.

```bash
# A-1. 특징점 추출 (몇 분)
colmap feature_extractor --database_path colmap/database.db --image_path $IMAGES \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 1 \
  2>&1 | tee colmap/log_1_extract.txt

# A-2. 순차 매칭: 앞뒤 20장 + 비슷한 장면 검색(loop detection) (몇 분)
colmap sequential_matcher --database_path colmap/database.db \
  --SequentialMatching.overlap 20 --SequentialMatching.loop_detection 1 --FeatureMatching.use_gpu 1 \
  2>&1 | tee colmap/log_2_match.txt

# A-3. mapper: 카메라 위치 + 3D 점 복원 (수십 분)
colmap mapper --database_path colmap/database.db --image_path $IMAGES --output_path colmap/sparse \
  2>&1 | tee colmap/log_3_mapper.txt
```

진행 상황 보기 (다른 터미널):
```bash
grep -oE "num_reg_frames=[0-9]+" $DATA/colmap/log_3_mapper.txt | tail -1
```

결과 요약:
```bash
for d in colmap/sparse/*/; do echo "== $d"; colmap model_analyzer --path $d 2>&1 | grep -E "Registered images|Points:|Mean track|reprojection" | sed 's/.*\] //'; done
```
- **정상**: 큰 모델 하나에 전체의 90% 이상 등록, Mean reprojection error 1px 미만 → **C로**
- **여러 조각** (큰 모델이 80% 이하이고 수십 장짜리 모델이 따로 있음) → **B로**
- `sparse/0`이 몇 장짜리이고 `sparse/1`이 큰 모델인 경우도 흔하다 (정상).

## B. 모델이 여러 조각일 때: 전체 매칭 추가

순차 매칭은 앞뒤 사진끼리만 비교해서, 근접 촬영 구간(화면 가득 물체만 보이는 장면)이 나머지 공간과 연결되지 못할 수 있다.
모든 사진 쌍을 비교하는 매칭을 **같은 DB에 추가**하고 mapper만 새 폴더에 다시 돌린다. (기존 `sparse/`는 그대로 남음)

```bash
# B-1. 전체 매칭 추가 (800장 기준 수 분)
colmap exhaustive_matcher --database_path colmap/database.db --FeatureMatching.use_gpu 1 \
  2>&1 | tee colmap/log_2b_exhaustive.txt

# B-2. mapper 다시
mkdir -p colmap/sparse_exh
colmap mapper --database_path colmap/database.db --image_path $IMAGES --output_path colmap/sparse_exh \
  2>&1 | tee colmap/log_3b_mapper_exh.txt

# B-3. 결과 요약
for d in colmap/sparse_exh/*/; do echo "== $d"; colmap model_analyzer --path $d 2>&1 | grep -E "Registered images|Points:|Mean track|reprojection" | sed 's/.*\] //'; done
```
그래도 조각나면: 방이 보이는 위치에서 그 물체로 **다가가는 장면**을 추가 촬영해 사진을 보강한다.

## C. 메인 모델 정하기 + 결과 검사

A 또는 B의 결과 요약에서 **Registered images가 가장 큰 폴더**를 메인 모델로 정한다.

```bash
export MODEL=colmap/sparse/0          # ← 예: colmap/sparse/1, colmap/sparse_exh/1 등 가장 큰 모델

# 카메라 경로 검사 (04 스크립트): 튐 5% 이하면 정상, 10% 이상이면 실패
python /path/to/CNN_PBL/data_prep/04_check_colmap_trajectory.py --model $MODEL --out colmap/trajectory.png
```
COLMAP 화면으로 직접 보기 (모니터 앞에서, 재구성 버튼은 누르지 말 것):
```bash
colmap gui --database_path colmap/database.db --image_path $IMAGES --import_path $MODEL
```

## D. 왜곡 보정 (3DGS 입력 만들기)

스마트폰 렌즈는 가장자리가 휘어 보인다. 3DGS는 왜곡 없는 카메라를 가정하므로, COLMAP이 계산한 왜곡 값으로 사진을 펴서
`gs_data/`(왜곡 없는 이미지 + PINHOLE 카메라 정보)를 만든다. 사진 가장자리가 조금 잘려 크기가 약간 줄어든다.

```bash
colmap image_undistorter --image_path $IMAGES --input_path $MODEL --output_path gs_data \
  --output_type COLMAP 2>&1 | tee colmap/log_4_undistort.txt
mkdir -p gs_data/sparse/0 && mv gs_data/sparse/*.bin gs_data/sparse/0/
ls gs_data/images | wc -l             # = 메인 모델의 Registered images 수
```

## E. 3DGS 학습

3DGS 환경을 켜고 실행 (`conda activate <3DGS 환경>`). A6000 기준 약 40분.

```bash
cd $GS_REPO
python train.py -s $DATA/gs_data -m $DATA/output/run1 --eval 2>&1 | tee $DATA/output/train_run1.log
```
- `--eval`: 8장마다 1장을 학습에서 빼서 test로 사용 (처음 보는 시점의 품질 측정)
- 이미지 폭이 1600보다 크면 자동으로 1600으로 줄인다는 안내가 나옴 (정상)
- GPU 메모리 부족이면 끝에 `-r 2` 추가 (해상도 절반)

## F. 3DGS 결과 확인

```bash
grep -E "ITER" $DATA/output/train_run1.log          # test PSNR 22~27 dB 이면 정상

# test 시점 렌더링 + 지표 (SSIM / PSNR / LPIPS)
cd $GS_REPO
python render.py -m $DATA/output/run1 --skip_train
python metrics.py -m $DATA/output/run1
```
결과 파일: `$DATA/output/run1/point_cloud/iteration_30000/point_cloud.ply` → SuperSplat(https://superspl.at/editor)에 드래그하면 바로 볼 수 있다.
