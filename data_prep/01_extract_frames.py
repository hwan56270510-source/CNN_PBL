#!/usr/bin/env python3
# 영상 -> 프레임 추출 (구간마다 가장 선명한 1장 선택, 흐림/중복 1차 제거). 사용법은 data_prep/README.md
import argparse
import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np


def probe(video):
    # ffmpeg 버전별 show_entries 지원 차이를 피하려고 전체 stream/format 정보를 받아 파싱
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_streams", "-show_format", "-of", "json", video]
    info = json.loads(subprocess.check_output(cmd))
    s = info["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    dur = float(s.get("duration") or info.get("format", {}).get("duration") or 0)
    rot = int(s.get("tags", {}).get("rotate", 0))
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) % 180 == 90:  # ffmpeg autorotate가 적용되므로 출력 크기는 w/h 교환
        w, h = h, w
    return w, h, fps, dur, rot


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pack(best, quality):
    """선택된 프레임을 JPEG 바이트 + 썸네일로 압축 보관 (원본 배열을 들고 있으면 4K에서 메모리 수십 GB)"""
    ci, t, sh, img, gray = best
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ci, t, sh, jpg.tobytes(), thumb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="출력 이미지 폴더 (예: classroom/images)")
    ap.add_argument("--target", type=int, default=500, help="목표 이미지 장수 (blur/중복 제거 전 기준)")
    ap.add_argument("--candidate_fps", type=float, default=6.0, help="후보 프레임 디코딩 fps (window 내 최선 선택용)")
    ap.add_argument("--max_size", type=int, default=1920, help="긴 변 최대 픽셀. 0이면 원본 해상도 유지")
    ap.add_argument("--blur_ratio", type=float, default=0.35, help="선명도 < median*ratio 이면 흐림으로 제거. 0이면 비활성")
    ap.add_argument("--blur_window", type=int, default=15,
                    help="흐림 기준 median을 계산할 주변 선택 프레임 수 (15 ≈ 앞뒤 10초). 0이면 영상 전체 median")
    ap.add_argument("--dup_thresh", type=float, default=1.5, help="직전 프레임과 썸네일 평균 차이 < 값이면 중복 제거. 0이면 비활성")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 품질")
    ap.add_argument("--start", type=float, default=0.0, help="시작 시간(초) - 촬영 시작 직후 흔들림 제외용")
    ap.add_argument("--end", type=float, default=0.0, help="종료 시간(초), 0이면 끝까지")
    args = ap.parse_args()

    w, h, src_fps, dur, rot = probe(args.video)
    end = args.end if args.end > 0 else dur
    span = end - args.start
    print(f"[probe] {w}x{h} @ {src_fps:.2f}fps, duration {dur:.1f}s, rotation {rot}")
    print(f"[range] {args.start:.1f}s ~ {end:.1f}s ({span:.1f}s)")

    scale = 1.0
    if args.max_size > 0 and max(w, h) > args.max_size:
        scale = args.max_size / max(w, h)
    ow, oh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)

    cand_fps = min(args.candidate_fps, src_fps)
    n_cand = int(span * cand_fps)
    win = max(1, int(round(n_cand / args.target)))
    print(f"[plan] output {ow}x{oh}, candidates ~{n_cand} @ {cand_fps}fps, window {win} frames "
          f"({win / cand_fps:.2f}s) -> ~{n_cand // win} frames before filtering")
    if n_cand < args.target:
        print(f"[warn] 후보 프레임({n_cand})이 target({args.target})보다 적음. --candidate_fps를 올리세요.")

    # 10-bit(HDR) 영상도 8-bit BGR로 변환됨. 단, HDR 톤매핑은 하지 않으므로 촬영 시 HDR 비디오를 끄는 것을 권장.
    cmd = ["ffmpeg", "-v", "error"]
    if args.start > 0:
        cmd += ["-ss", str(args.start)]
    cmd += ["-i", args.video]
    if args.end > 0:
        cmd += ["-t", str(span)]
    cmd += ["-vf", f"fps={cand_fps},scale={ow}:{oh}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=ow * oh * 3 * 4)

    frame_bytes = ow * oh * 3
    picks = []  # (cand_idx, time, sharp, jpeg_bytes, thumb)
    best = None
    idx = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        img = np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 선명도는 고정 크기(긴 변 960)에서 측정 -> 해상도와 무관하게 비교 가능
        s_small = 960 / max(ow, oh)
        g_s = cv2.resize(gray, None, fx=s_small, fy=s_small, interpolation=cv2.INTER_AREA) if s_small < 1 else gray
        sh = sharpness(g_s)
        if best is None or sh > best[2]:
            best = (idx, args.start + idx / cand_fps, sh, img, gray)
        idx += 1
        if idx % win == 0:
            picks.append(pack(best, args.quality))
            best = None
        if idx % 200 == 0:
            print(f"  decoded {idx}/{n_cand} candidates, picked {len(picks)}", end="\r", flush=True)
    if best is not None:
        picks.append(pack(best, args.quality))
    proc.wait()
    print()
    if proc.returncode != 0 or not picks:
        sys.exit(f"[error] ffmpeg 디코딩 실패 (returncode {proc.returncode}, picks {len(picks)})")

    sharp_all = np.array([p[2] for p in picks])
    med = float(np.median(sharp_all))
    # 흐림 기준 = 주변 window의 선명도 median * ratio
    # (전역 median을 쓰면 무늬 없는 배경의 근접샷 구간이 선명해도 통째로 blur로 제거됨)
    if args.blur_window > 0:
        h = args.blur_window // 2
        blur_th_all = np.array([np.median(sharp_all[max(0, i - h):i + h + 1]) for i in range(len(sharp_all))]) * args.blur_ratio
    else:
        blur_th_all = np.full(len(sharp_all), med * args.blur_ratio)
    print(f"[sharpness] median {med:.1f}, min {sharp_all.min():.1f}, max {sharp_all.max():.1f}, "
          f"blur threshold {blur_th_all.min():.1f}~{blur_th_all.max():.1f} "
          f"({'local window ' + str(args.blur_window) if args.blur_window > 0 else 'global'})")

    os.makedirs(args.out, exist_ok=True)
    if any(f.endswith(".jpg") for f in os.listdir(args.out)):
        sys.exit(f"[error] {args.out} 에 이미 jpg가 있습니다. 비우고 다시 실행하세요.")

    log_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "extraction_log.csv")
    n_keep = n_blur = n_dup = 0
    last_thumb = None
    with open(log_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["candidate_idx", "time_sec", "sharpness", "status", "filename"])
        for k, (ci, t, sh, jpg, thumb) in enumerate(picks):
            if args.blur_ratio > 0 and sh < blur_th_all[k]:
                n_blur += 1
                wr.writerow([ci, f"{t:.2f}", f"{sh:.1f}", "blur", ""])
                continue
            if args.dup_thresh > 0 and last_thumb is not None and np.abs(thumb - last_thumb).mean() < args.dup_thresh:
                n_dup += 1
                wr.writerow([ci, f"{t:.2f}", f"{sh:.1f}", "duplicate", ""])
                continue
            n_keep += 1
            name = f"frame_{n_keep:05d}.jpg"
            with open(os.path.join(args.out, name), "wb") as fo:
                fo.write(jpg)
            wr.writerow([ci, f"{t:.2f}", f"{sh:.1f}", "keep", name])
            last_thumb = thumb

    print(f"[done] kept {n_keep}, removed blur {n_blur}, removed duplicate {n_dup}")
    print(f"[out] {args.out}  ({ow}x{oh})")
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
