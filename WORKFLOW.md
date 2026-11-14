# 작업 흐름 (YOLO-Seg / CNN / 3DGS)

세 파트가 환경을 따로 두고, 파일(이미지 + JSON)로만 결과를 주고받는 방식으로 진행한다.

```
[원본 이미지]
     |
     v
1. YOLO-Seg 담당자  --(yolo_results/*.json)-->  2. CNN 담당자  --(results.jsonl)-->  3. 3DGS 담당자
                                                                                         |
                                                                                         v
                                                                                  공구별 3D 위치 + 클래스
```

## 1. YOLO-Seg 담당자

1. **데이터셋 준비**: 공구 마스크 라벨링, train/val/test 분할, (필요하면) 학습용 증강
2. **모델 학습**: `model.train(...)` 으로 YOLO-Seg 학습 → `best.pt` 생성
3. **추론 결과 내보내기**: 자기 ultralytics 환경에서
   ```
   python yolo_export.py --source 이미지폴더 --weights best.pt --out yolo_results
   ```
   → `best.pt`로 이미지들을 추론해서, 박스/마스크/확신도만 이미지별 JSON으로 저장
   (CNN 분류는 하지 않음 — "여기 공구 같은 게 있다"는 위치 정보만 뽑는 단계)
4. **전달**: `yolo_results/` 폴더 + 추론에 사용한 원본 이미지 폴더를 CNN 담당자에게 전달

## 2. CNN 담당자

1. **CNN 학습**: `python train.py --config configs/cnn.yaml` → `runs/<exp>/best.h5` 생성
2. **(선택) 평가**: `python eval.py --config configs/cnn.yaml --ckpt runs/<exp>/best.h5 --split test`
3. **YOLO 결과 받아서 분류 실행**: YOLO 담당자가 준 이미지 폴더 + `yolo_results/`로
   ```
   python infer_2d.py --source 이미지폴더 --yolo-json-dir yolo_results \
                       --config configs/cnn.yaml --ckpt runs/<exp>/best.h5 --out demo_out
   ```
   → YOLO가 찾은 각 위치를 crop해서 CNN으로 드릴/망치/클램프/other 분류, 확률까지 계산
   → `demo_out/results.jsonl` 생성 (bbox, mask_polygon, 클래스별 확률, 최종 예측 클래스)
4. **전달**: `demo_out/results.jsonl` (또는 `demo_out/results/`)을 3DGS 담당자에게 전달

## 3. 3DGS 담당자

1. **3DGS용 사진**(`gs_data/images`, 카메라 포즈 있는 사진)에 `infer_2d.py --yolo-weights`로 YOLO+CNN을 다시 돌려 `results.jsonl` 생성
   (YOLO 라벨링용 사진은 카메라 포즈가 없어서 이 단계엔 못 씀)
2. `lift3d/lift_to_3d.py`로 COLMAP 카메라 + 3DGS point_cloud.ply + `results.jsonl`을 합쳐 공구별 3D 위치 계산
   ```
   python lift3d/lift_to_3d.py --results results.jsonl --colmap gs_data/sparse/0 \
       --ply point_cloud.ply --images gs_data/images --out_dir runs/objects_3d
   ```
   → 여러 시점에서 보이는 Gaussian을 마스크로 투표해서 물체 Gaussian을 정하고, 중심/회전박스(OBB)/반경/클로즈업 시점까지 계산 (자세한 원리는 `lift3d/README.md`)
3. 결과: `runs/objects_3d/objects.json` (공구별 3D 위치) + `*_check.jpg` (확인용 그림)

## 최종 결과 (2026-11-14 기준)
- CNN 분류기: test accuracy 95.6%, k-fold(4개) 평균 95.0% (`runs/cnn_rgb128_classroom_v2_final`, `runs/cnn_rgb128_margin05_noearlystop_fold*`)
- 3D 위치: `runs/objects_3d/objects.json` — drill/hammer/clamp 3종 전부 계산 완료 (검출 프레임 72~102장, inlier 63~97장)

## 참고
- 세 파트 모두 각자 환경에서 독립적으로 돌아가며, 공유하는 건 이미지 파일 + JSON 파일뿐이다.
- CNN 쪽 코드/설정/실행법 상세는 `README.md`, 환경 구성 이유는 `environment.md`, 3D 변환 상세는 `lift3d/README.md` 참고.
