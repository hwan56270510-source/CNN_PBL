#!/usr/bin/env python3
# COLMAP 카메라 경로 그림 + 튐(jump) 비율 검사. 사용법은 data_prep/README.md
import argparse
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np


def qvec2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="COLMAP 모델 폴더 (예: colmap/sparse/0, .bin 또는 .txt)")
    ap.add_argument("--out", required=True, help="출력 PNG 경로")
    ap.add_argument("--jump_ratio", type=float, default=6.0, help="median 이동거리의 몇 배 이상을 jump로 볼지")
    ap.add_argument("--colmap", default="colmap", help="바이너리 모델을 TXT로 바꿀 때 쓸 colmap 실행 파일")
    args = ap.parse_args()

    # 바이너리 모델(.bin)이면 임시 폴더에 TXT로 변환
    if not os.path.exists(os.path.join(args.model, "images.txt")):
        tmp = tempfile.mkdtemp(prefix="colmap_txt_")
        subprocess.run([args.colmap, "model_converter", "--input_path", args.model, "--output_path", tmp,
                        "--output_type", "TXT"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        args.model = tmp

    lines = [l for l in open(f"{args.model}/images.txt") if not l.startswith("#")]
    cams = []
    for l in lines[0::2]:
        p = l.split()
        R = qvec2R(np.array(list(map(float, p[1:5]))))
        t = np.array(list(map(float, p[5:8])))
        fid = int(re.findall(r"(\d+)", p[9])[-1])  # 파일 이름의 숫자 = 촬영 순서
        cams.append((fid, -R.T @ t, R.T @ np.array([0, 0, 1.0])))
    cams.sort(key=lambda c: c[0])
    fid = np.array([c[0] for c in cams])
    C = np.array([c[1] for c in cams])
    fwd = np.array([c[2] for c in cams])
    pts = np.array([list(map(float, l.split()[1:4])) for l in open(f"{args.model}/points3D.txt") if not l.startswith("#")])

    # 수직축 = 점구름에서 분산이 가장 작은 축 (실내: 높이 < 가로/세로)
    ctr = np.median(pts, 0)
    P = pts - ctr
    P = P[np.linalg.norm(P, axis=1) < np.percentile(np.linalg.norm(P, axis=1), 95)]
    _, _, vt = np.linalg.svd(P[::5], full_matrices=False)
    e1, e2, up = vt[0], vt[1], vt[2]
    proj = lambda X: np.stack([(X - ctr) @ e1, (X - ctr) @ e2], 1)

    p2, c2 = proj(pts), proj(C)
    lo, hi = np.percentile(p2, 1, 0), np.percentile(p2, 99, 0)
    W = 1100
    sc = (W - 100) / (hi - lo).max()
    H = int((hi - lo)[1] * sc) + 100
    img = np.full((H, W, 3), 255, np.uint8)
    to_pix = lambda q: ((q - lo) * sc + 50).astype(int)
    for x, y in to_pix(p2):
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = (170, 170, 170)
    cp = to_pix(c2)
    n = len(cp)
    for i in range(n - 1):
        col = cv2.applyColorMap(np.uint8([[int(255 * i / n)]]), cv2.COLORMAP_TURBO)[0, 0].tolist()
        cv2.line(img, tuple(cp[i]), tuple(cp[i + 1]), col, 2)
        if i % 10 == 0:
            tip = to_pix(proj(C[i:i + 1] + fwd[i:i + 1] * 0.4))[0]
            cv2.arrowedLine(img, tuple(cp[i]), tuple(tip), col, 1, tipLength=0.3)
    cv2.circle(img, tuple(cp[0]), 8, (0, 180, 0), -1)
    cv2.circle(img, tuple(cp[-1]), 8, (0, 0, 220), -1)
    cv2.putText(img, "TOP VIEW  gray=points  line=camera path (blue->red=time)  green=start red=end",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    hc = (C - ctr) @ up
    if np.mean(hc) < 0:
        hc = -hc
    hn = (hc - hc.min()) / (np.ptp(hc) + 1e-9)
    hp = np.full((220, W, 3), 255, np.uint8)
    for i in range(n - 1):
        a = (int(50 + (W - 100) * i / n), int(190 - 150 * hn[i]))
        b = (int(50 + (W - 100) * (i + 1) / n), int(190 - 150 * hn[i + 1]))
        cv2.line(hp, a, b, (80, 80, 80), 2)
    cv2.putText(hp, "camera height vs frame order", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(args.out, np.vstack([img, hp]))

    # jump 검사: 건너뛴 프레임 수로 정규화
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    gap = np.maximum(np.diff(fid), 1)
    med = np.median(step / gap)
    ratio = step / (med * gap)
    jumps = [(int(fid[i]), int(fid[i + 1]), round(float(ratio[i]), 1)) for i in range(len(ratio)) if ratio[i] > args.jump_ratio]
    print(f"[cameras] {n}  [points] {len(pts)}")
    print(f"[jumps > {args.jump_ratio}x median] {len(jumps)}  ({100 * len(jumps) / max(n - 1, 1):.1f}% of steps)")
    for a, b, r in jumps[:40]:
        print(f"  frame {a} -> {b}: {r}x")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
