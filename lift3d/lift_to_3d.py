#!/usr/bin/env python3
"""
infer_2d.py 결과(results.jsonl) + COLMAP 카메라 + 3DGS point_cloud.ply  →  공구별 3D 위치(objects.json)

1차 데모 방식 그대로 (단, 3DGS 렌더러 대신 Gaussian 중심 z-buffer로 가림을 판단 → numpy + opencv 만 필요, GPU 불필요)
  1) 공구 종류별로, 그 종류가 검출된 사진마다 "보이는(가려지지 않은) Gaussian 중 마스크 안에 들어오는 것" = 시점별 3D 점
  2) 시점별 중앙점이 다수와 어긋나는 시점은 이상치(오검출)로 제거
  3) Gaussian 투표: 여러 시점에서 보일 때마다 마스크 안에 들어오는 비율이 높으면 물체 Gaussian
     → 중심 / 회전 박스(OBB) / 반경을 물체 Gaussian 위치로 계산
  4) 물체가 잘리지 않고 적당한 크기로 찍힌 실제 사진 시점 = 클로즈업 카메라

좌표계: COLMAP world (gs_data/sparse/0) = 3DGS PLY 좌표계. 카메라 축은 COLMAP 규약(x 오른쪽, y 아래, z 앞).
주의: results.jsonl 은 반드시 3DGS용 사진(gs_data/images, 왜곡 보정된 사진)에 infer_2d.py 를 돌린 결과여야 한다
      (카메라 위치는 그 사진들에만 있다).

사용 예:
  python lift3d/lift_to_3d.py \
    --results runs/infer_gs/results.jsonl \
    --colmap  <데이터>/gs_data/sparse/0 \
    --ply     <데이터>/output/run1/point_cloud/iteration_30000/point_cloud.ply \
    --images  <데이터>/gs_data/images \
    --out_dir runs/objects_3d
"""
import argparse
import datetime
import json
import os
import struct

import cv2
import numpy as np

# ---------------------------------------------------------------- COLMAP 읽기 (.bin / .txt)
CAM_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
CAM_NAMES = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def pinhole(model_id, params):
    if model_id == 0:  # SIMPLE_PINHOLE f cx cy
        f, cx, cy = params
        return f, f, cx, cy
    if model_id == 1:  # PINHOLE fx fy cx cy
        return tuple(params)
    raise SystemExit("[error] 카메라가 PINHOLE 이 아님 → 왜곡 보정(image_undistorter) 결과 gs_data/sparse/0 을 넣을 것")


def read_colmap(d):
    cams, imgs = {}, {}
    if os.path.exists(os.path.join(d, "cameras.bin")):
        with open(os.path.join(d, "cameras.bin"), "rb") as f:
            for _ in range(struct.unpack("<Q", f.read(8))[0]):
                cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
                p = struct.unpack("<" + "d" * CAM_NPARAMS[mid], f.read(8 * CAM_NPARAMS[mid]))
                cams[cid] = (w, h) + pinhole(mid, p)
        with open(os.path.join(d, "images.bin"), "rb") as f:
            for _ in range(struct.unpack("<Q", f.read(8))[0]):
                v = struct.unpack("<idddddddi", f.read(64))
                name = b""
                while (c := f.read(1)) != b"\x00":
                    name += c
                n2d = struct.unpack("<Q", f.read(8))[0]
                f.seek(24 * n2d, 1)
                imgs[name.decode()] = (np.array(v[1:5]), np.array(v[5:8]), v[8])
    else:
        for l in open(os.path.join(d, "cameras.txt")):
            if l.startswith("#") or not l.strip():
                continue
            p = l.split()
            cams[int(p[0])] = (int(p[2]), int(p[3])) + pinhole(CAM_NAMES.get(p[1], -1), list(map(float, p[4:])))
        lines = [l for l in open(os.path.join(d, "images.txt")) if not l.startswith("#")]
        for l in lines[0::2]:
            p = l.split()
            if len(p) >= 10:
                imgs[p[9]] = (np.array(list(map(float, p[1:5]))), np.array(list(map(float, p[5:8]))), int(p[8]))
    return cams, imgs


# ---------------------------------------------------------------- 3DGS PLY 읽기 (xyz + opacity)
def read_gaussians(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while (l := f.readline().decode().strip()) != "end_header":
            if l.startswith("format") and "binary_little_endian" not in l:
                raise SystemExit("[error] binary_little_endian PLY 만 지원")
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
            elif l.startswith("property"):
                t, name = l.split()[1:3]
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1"}[t]))
        a = np.fromfile(f, dtype=np.dtype(props), count=n)
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32)
    opacity = 1 / (1 + np.exp(-a["opacity"].astype(np.float32)))  # 3DGS PLY 는 sigmoid 이전 값 저장
    return xyz, opacity


# ---------------------------------------------------------------- 시점 하나
class View:
    def __init__(self, name, cam, q, t, scale):
        w, h, fx, fy, cx, cy = cam
        self.name, self.R, self.t = name, qvec2rotmat(q), t
        self.W, self.H = int(round(w * scale)), int(round(h * scale))
        self.K = np.array([fx, fy, cx, cy]) * scale
        self.full_wh = (w, h)
        self.center = -self.R.T @ t
        self.fovy = 2 * np.arctan(h / (2 * fy))

    def project(self, X):
        pc = X @ self.R.T.astype(np.float32) + self.t.astype(np.float32)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(self.K[0] * pc[:, 0] / z + self.K[2]).astype(np.int64)
            v = np.round(self.K[1] * pc[:, 1] / z + self.K[3]).astype(np.int64)
        ok = (z > 0.01) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, z, ok


def zbuffer(view, occ_xyz, fill_px):
    """불투명 Gaussian 중심으로 화면별 가장 가까운 깊이. 점 사이 빈틈은 최소값 필터(erode)로 메움"""
    u, v, z, ok = view.project(occ_xyz)
    pix, z = v[ok] * view.W + u[ok], z[ok]
    o = np.lexsort((z, pix))
    first = np.r_[True, pix[o][1:] != pix[o][:-1]]
    D = np.full(view.H * view.W, np.inf, np.float32)
    D[pix[o][first]] = z[o][first]
    D = D.reshape(view.H, view.W)
    if fill_px > 0:
        D = cv2.erode(D, np.ones((2 * fill_px + 1,) * 2, np.uint8))
    return D


def visible_in_mask(view, xyz, D, mask, tol):
    """보이는 Gaussian 인덱스, 그중 마스크 안에 들어오는지"""
    u, v, z, ok = view.project(xyz)
    idx = np.nonzero(ok)[0]
    with np.errstate(over="ignore", invalid="ignore"):  # 빈 화소(inf) = 뒤에 아무것도 없음 → 보임
        vis = z[idx] <= D[v[idx], u[idx]] * (1 + tol)
    idx = idx[vis]
    return idx, mask[v[idx], u[idx]] > 0


def poly_mask(view, poly, img_wh):
    sx, sy = view.W / img_wh[0], view.H / img_wh[1]
    m = np.zeros((view.H, view.W), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly) * [sx, sy]).astype(np.int32)], 1)
    return m


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="infer_2d.py 의 results.jsonl (3DGS용 사진에 돌린 것)")
    ap.add_argument("--colmap", required=True, help="왜곡 보정된 COLMAP 모델 폴더 (gs_data/sparse/0, .bin 또는 .txt)")
    ap.add_argument("--ply", required=True, help="3DGS point_cloud.ply")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--images", default=None, help="gs_data/images (주면 확인용 그림 저장)")
    ap.add_argument("--classes", default="drill:드릴,hammer:망치,clamp:클램프", help="키:한국어이름, 쉼표 구분")
    ap.add_argument("--min_cls_conf", type=float, default=0.6, help="CNN 확신도 최소값")
    ap.add_argument("--min_yolo_conf", type=float, default=0.4, help="YOLO 확신도 최소값")
    ap.add_argument("--scale", type=float, default=0.25, help="깊이 계산 해상도 (원본 대비)")
    ap.add_argument("--occ_opacity", type=float, default=0.3, help="가림 판단에 쓸 Gaussian 최소 불투명도")
    ap.add_argument("--fill_px", type=int, default=1, help="z-buffer 빈틈 메우기(px)")
    ap.add_argument("--depth_tol", type=float, default=0.03, help="표면보다 이 비율 이내 뒤까지 '보임'으로 인정")
    ap.add_argument("--erode", type=int, default=2, help="1단계에서 마스크 경계 깎기(px, 계산 해상도 기준)")
    ap.add_argument("--vote_ratio", type=float, default=0.6, help="물체 Gaussian 판정: 보일 때 마스크 안 비율")
    ap.add_argument("--min_votes", type=int, default=3)
    ap.add_argument("--min_views", type=int, default=3)
    ap.add_argument("--shape_opacity", type=float, default=0.3, help="형상 계산에 쓸 물체 Gaussian 최소 불투명도")
    ap.add_argument("--target_area", type=float, default=0.12, help="클로즈업: 물체 마스크가 화면에서 차지할 목표 비율")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cams, imgs = read_colmap(args.colmap)
    xyz, opac = read_gaussians(args.ply)
    occ = xyz[opac >= args.occ_opacity]
    print(f"[colmap] {len(imgs)} images  [gaussians] {len(xyz)} (가림 판단용 {len(occ)})")
    records = [json.loads(l) for l in open(args.results, encoding="utf-8") if l.strip()]
    nomatch = [r["image"] for r in records if r["image"] not in imgs]
    if len(nomatch) == len(records):
        raise SystemExit("[error] results.jsonl 의 사진 이름이 COLMAP 과 하나도 안 맞음 → gs_data/images 에 infer_2d.py 를 돌렸는지 확인")
    if nomatch:
        print(f"[warn] COLMAP 에 없는 사진 {len(nomatch)}장은 건너뜀 (예: {nomatch[:3]})")

    allR = [qvec2rotmat(q) for q, _, _ in imgs.values()]
    scene_up = np.mean([-R.T[:, 1] for R in allR], 0)
    scene_up /= np.linalg.norm(scene_up)

    jpath = os.path.join(args.out_dir, "objects.json")
    J = {"objects": {}, "coordinate_frame": "COLMAP world = 3DGS PLY (gs_data/sparse/0). camera axes: x right, y down, z forward",
         "scene_up": np.round(scene_up, 5).tolist(), "source": os.path.abspath(args.colmap), "model_ply": os.path.abspath(args.ply)}
    depth_cache = {}
    kernel = np.ones((2 * args.erode + 1,) * 2, np.uint8)

    for spec in args.classes.split(","):
        key, ko = (spec.split(":") + [""])[:2]
        # 사진마다 이 종류의 가장 확실한 검출 1개
        views = []
        for r in records:
            if r["image"] not in imgs:
                continue
            ds = [d for d in r["detections"] if d["pred_class"] == key and d["pred_conf"] >= args.min_cls_conf
                  and d["yolo_conf"] >= args.min_yolo_conf and len(d["mask_polygon"]) >= 3]
            if ds:
                views.append((r, max(ds, key=lambda d: d["pred_conf"] * d["yolo_conf"])))
        print(f"\n===== {key} ({ko}) : 검출된 사진 {len(views)}장")
        if len(views) < args.min_views:
            print(f"[skip] 검출 사진이 {args.min_views}장 미만")
            continue

        # ---------- 1) 시점별 3D 점 ----------
        per_view = []
        for r, d in views:
            q, t, cid = imgs[r["image"]]
            vw = View(r["image"], cams[cid], q, t, args.scale)
            if r["image"] not in depth_cache:
                depth_cache[r["image"]] = zbuffer(vw, occ, args.fill_px)
            D = depth_cache[r["image"]]
            full = poly_mask(vw, d["mask_polygon"], r["image_wh"])
            idx, inm = visible_in_mask(vw, xyz, D, cv2.erode(full, kernel), args.depth_tol)
            pts = xyz[idx[inm]]
            if len(pts) < 20:
                continue
            touch = bool(full[:2].any() or full[-2:].any() or full[:, :2].any() or full[:, -2:].any())
            per_view.append(dict(view=vw, det=d, conf=d["pred_conf"] * d["yolo_conf"], pts=pts, med=np.median(pts, 0),
                                 area=float(full.mean()), touch=touch, mask=full))
        if len(per_view) < args.min_views:
            print(f"[skip] 3D 점을 얻은 사진이 {len(per_view)}장뿐")
            continue

        # ---------- 2) 이상치 시점 제거 + 초기 형상 ----------
        meds = np.array([v["med"] for v in per_view])
        dist = np.linalg.norm(meds - np.median(meds, 0), axis=1)
        mad = np.median(np.abs(dist - np.median(dist))) * 1.4826 + 1e-9
        thr = np.median(dist) + 3 * mad
        inl = [v for v, di in zip(per_view, dist) if di <= thr]
        out = [v["view"].name for v, di in zip(per_view, dist) if di > thr]
        P = np.concatenate([v["pts"] for v in inl])
        c0 = np.median(P, 0)
        rr = np.linalg.norm(P - c0, axis=1)
        P = P[rr < np.percentile(rr, 90) * 1.5]
        c0 = np.median(P, 0)
        _, _, axes = np.linalg.svd(P - c0, full_matrices=False)
        loc = (P - c0) @ axes.T
        lo, hi = np.percentile(loc, 2, 0), np.percentile(loc, 98, 0)
        obb_center, extents = c0 + ((lo + hi) / 2) @ axes, hi - lo
        print(f"[views] inlier {len(inl)} / outlier {len(out)} {out[:8]}")

        # ---------- 3) Gaussian 투표 ----------
        loc_all = (xyz - obb_center) @ axes.T
        cand = np.nonzero((np.abs(loc_all) <= extents / 2 * 1.3 + 0.02).all(1))[0]  # 초기 박스 근처만 투표
        votes = np.zeros(len(cand), np.int32)
        seen = np.zeros(len(cand), np.int32)
        for v in inl:
            idx, inm = visible_in_mask(v["view"], xyz[cand], depth_cache[v["view"].name], v["mask"], args.depth_tol)
            seen[idx] += 1
            votes[idx] += inm
        sel = cand[(votes >= args.min_votes) & (votes >= args.vote_ratio * np.maximum(seen, 1))]
        np.save(os.path.join(args.out_dir, f"{key}_gaussian_ids.npy"), sel)
        print(f"[gaussians] {len(sel)} selected")

        G = xyz[sel][opac[sel] > args.shape_opacity]
        if len(G) >= 100:
            c = np.median(G, 0)
            _, _, axes = np.linalg.svd(G - c, full_matrices=False)
            axes = axes.copy()
            if np.linalg.det(axes) < 0:  # 오른손 좌표계(회전행렬)로
                axes[2] = -axes[2]
            loc = (G - c) @ axes.T
            lo, hi = np.percentile(loc, 1, 0), np.percentile(loc, 99, 0)
            obb_center, extents, P = c + ((lo + hi) / 2) @ axes, hi - lo, G
        else:
            print(f"[warn] 물체 Gaussian {len(G)}개뿐 → 1단계 점으로 형상 유지")
        radius = float(np.linalg.norm(extents) / 2)
        print(f"[shape] center {np.round(obb_center, 4).tolist()}  extents {np.round(extents, 4).tolist()}  radius {radius:.4f}")

        # ---------- 4) 클로즈업 시점 ----------
        def closeup_score(v):
            vw = v["view"]
            pc = vw.R @ obb_center + vw.t
            w, h = vw.full_wh
            du = vw.K[0] / args.scale * pc[0] / pc[2] / w
            dv = vw.K[1] / args.scale * pc[1] / pc[2] / h
            centered = np.exp(-(du ** 2 + dv ** 2) / 0.05)
            size = np.exp(-((v["area"] - args.target_area) / (args.target_area * 0.6)) ** 2)
            return v["conf"] * centered * size * (0.0 if v["touch"] else 1.0)
        ranked = sorted(inl, key=lambda v: v["conf"] * np.sqrt(v["area"]) * (0.3 if v["touch"] else 1.0), reverse=True)
        best_views = [{"image": v["view"].name, "conf": round(v["conf"], 4), "mask_area_frac": round(v["area"], 4),
                       "touches_border": v["touch"], "camera_center": np.round(v["view"].center, 5).tolist(),
                       "camera_R_world2cam": np.round(v["view"].R, 6).tolist()} for v in ranked[:5]]
        bv = max(inl, key=closeup_score)
        vd = obb_center - bv["view"].center
        vd /= np.linalg.norm(vd)
        up = -bv["view"].R.T[:, 1]
        up = up - vd * (up @ vd)
        up /= np.linalg.norm(up)
        dd = float(np.linalg.norm(obb_center - bv["view"].center))
        closeup = {"position": np.round(bv["view"].center, 5).tolist(), "look_at": np.round(obb_center, 5).tolist(),
                   "up": np.round(up, 5).tolist(), "fovy_deg": round(float(np.degrees(bv["view"].fovy)), 2),
                   "distance": round(dd, 5), "from_view": bv["view"].name, "mask_area_frac": round(bv["area"], 4)}
        print(f"[closeup] {bv['view'].name}  distance {dd:.3f}")

        J["objects"][key] = {
            "name": key, "name_ko": ko, "aliases": [a for a in (key, ko) if a],
            "center": np.round(obb_center, 5).tolist(),
            "obb": {"center": np.round(obb_center, 5).tolist(), "axes": np.round(axes, 6).tolist(), "extents": np.round(extents, 5).tolist()},
            "aabb": {"min": np.round(P.min(0), 5).tolist(), "max": np.round(P.max(0), 5).tolist()},
            "radius": round(radius, 5),
            "n_views_detected": len(views), "n_views_inlier": len(inl), "outlier_views": out,
            "gaussian_ids_file": f"{key}_gaussian_ids.npy", "n_gaussians": int(len(sel)),
            "best_views": best_views, "closeup": closeup,
            "detector": {"results": os.path.abspath(args.results), "min_cls_conf": args.min_cls_conf, "min_yolo_conf": args.min_yolo_conf},
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        # 확인용 그림: 클로즈업 사진 위에 물체 Gaussian(초록) + 박스(빨강) 투영
        if args.images:
            img = cv2.imread(os.path.join(args.images, bv["view"].name))
            if img is not None:
                vw = bv["view"]
                s = img.shape[1] / vw.W
                u, v, _, ok = vw.project(xyz[sel])
                for x, y in zip(u[ok], v[ok]):
                    cv2.circle(img, (int(x * s), int(y * s)), 1, (0, 255, 0), -1)
                corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * extents / 2 @ axes + obb_center
                cu, cv_, _, _ = vw.project(corners.astype(np.float32))
                cp = np.stack([cu, cv_], 1) * s
                for a in range(8):
                    for b in range(a + 1, 8):
                        if bin(a ^ b).count("1") == 1:
                            cv2.line(img, tuple(map(int, cp[a])), tuple(map(int, cp[b])), (0, 0, 255), 2)
                cv2.putText(img, f"{key}  views {len(inl)}  gaussians {len(sel)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(args.out_dir, f"{key}_check.jpg"), img)

    json.dump(J, open(jpath, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n[saved] {jpath}  objects: {list(J['objects'])}")


if __name__ == "__main__":
    main()
