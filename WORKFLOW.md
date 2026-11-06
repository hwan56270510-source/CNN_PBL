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

1. `results.jsonl`의 각 검출마다 `mask_polygon` 안쪽 픽셀로 depth를 샘플링해 3D 포인트 획득
2. 같은 물체가 여러 프레임(여러 시점)에서 잡힌 경우, `cls_probs`(클래스별 확률)를 가중치로 삼아 투표 → 최종 클래스 확정
3. `pred_class`가 `other`인 검출은 실제 공구가 아닐 가능성이 높으므로 제외 권장
4. 결과: 공구별(드릴/망치/클램프) 3D 위치

## 참고
- 세 파트 모두 각자 환경에서 독립적으로 돌아가며, 공유하는 건 이미지 파일 + JSON 파일뿐이다.
- CNN 쪽 코드/설정/실행법 상세는 `README.md`, 환경 구성 이유는 `environment.md` 참고.
