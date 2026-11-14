# lift3d: 2D 검출 결과 → 공구별 3D 위치

`infer_2d.py` 결과(`results.jsonl`)와 COLMAP 카메라, 3DGS 모델로 drill, hammer, clamp의 3D 위치를 계산해 `objects.json`으로 저장한다.
numpy와 opencv만 필요하다. GPU와 3DGS 렌더러는 쓰지 않는다.

## 입력
| 입력 | 위치 (예) |
|---|---|
| `results.jsonl` | **3DGS용 사진**(`gs_data/images`)에 `infer_2d.py`를 돌린 결과. YOLO용 사진에는 카메라 위치가 없어서 쓸 수 없다. |
| COLMAP 모델 | `gs_data/sparse/0` (왜곡 보정된 PINHOLE 모델, `.bin` 또는 `.txt`) |
| 3DGS 모델 | `output/run1/point_cloud/iteration_30000/point_cloud.ply` |

## 실행
```bash
# 1) 3DGS용 사진에 YOLO + CNN
python infer_2d.py --source <데이터>/gs_data/images --yolo-weights tool_v2_best.pt \
  --ckpt runs/<exp>/best.h5 --out runs/infer_gs

# 2) 3D 위치
python lift3d/lift_to_3d.py --results runs/infer_gs/results.jsonl \
  --colmap <데이터>/gs_data/sparse/0 \
  --ply <데이터>/output/run1/point_cloud/iteration_30000/point_cloud.ply \
  --images <데이터>/gs_data/images --out_dir runs/objects_3d
```

## 방법
1. 종류별로, 검출된 사진마다 가려지지 않은 3DGS Gaussian 중 마스크 안에 들어오는 것을 모은다 (시점별 3D 점).
   가림은 불투명한 Gaussian 중심으로 만든 z-buffer로 판단한다.
2. 시점별 중앙점이 다수와 어긋나는 사진은 오검출로 보고 뺀다.
3. 여러 사진에서 보일 때마다 마스크 안에 들어오는 Gaussian을 물체로 정한다. 그 위치로 중심, 회전 박스(OBB), 반경을 계산한다.
4. 물체가 잘리지 않고 적당한 크기로 찍힌 실제 사진 위치를 클로즈업 카메라로 정한다.

## 출력 (`--out_dir`)
- `objects.json`: 물체별 `center`, `obb`, `radius`, `closeup`(카메라 위치, 바라보는 점, 위쪽 방향), `best_views` 등
- `<종류>_gaussian_ids.npy`: 물체 Gaussian 번호 (뷰어 하이라이트용)
- `<종류>_check.jpg`: 클로즈업 사진 위에 물체 Gaussian(초록)과 박스(빨강)를 그린 확인 그림 (`--images`를 줄 때)

## 주요 옵션
- `--min_cls_conf 0.6`, `--min_yolo_conf 0.4`: 이보다 확신도가 낮은 검출은 쓰지 않는다.
- `--classes drill:드릴,hammer:망치,clamp:클램프`: 계산할 종류와 한국어 이름(검색용)

## 검증
1차 데모(classroom_v3) 드릴 검출로 돌린 결과를 기존 3DGS 렌더러 방식과 비교했다.
중심 차이는 물체 크기의 약 2%였고, 클로즈업 시점은 같았다.
742장 기준 실행 시간 15초, 메모리 0.6GB.
