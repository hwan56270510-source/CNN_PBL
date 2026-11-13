#!/usr/bin/env python3
# 추출 프레임 모션 블러 검출(analyze) -> 육안 검토 -> 제거 + 번호 재정리(apply). 사용법은 data_prep/README.md
import argparse
import csv
import os
import shutil

import cv2
import numpy as np

PROFILES = {
    "gs": dict(rel=0.45, pct=15, rel_or_pct=False, dup=0.0, dark=40, bright=225),
    # abs_rel: 절대 선명도(하위 pct%) 기준은 상대값도 abs_rel 미만일 때만 적용.
    #   무늬 없는 배경 위 물체(V4 회색 선반 위 클램프)는 선명해도 절대값이 낮음 → 상대값으로 보호
    "yolo": dict(rel=0.45, pct=4, rel_or_pct=True, abs_rel=0.55, dup=2.0, dark=40, bright=225),
}


def paths(images):
    images = os.path.abspath(os.path.expanduser(images.rstrip("/")))
    root = os.path.dirname(images)
    return images, root, os.path.join(root, "qa"), os.path.join(root, "extraction_log.csv")


def load_times(log):
    if not os.path.exists(log):
        return {}
    return {r["filename"].strip(): float(r["time_sec"]) for r in csv.DictReader(open(log)) if r["status"].strip() == "keep"}


def analyze(args):
    images, root, qa, log = paths(args.images)
    os.makedirs(qa, exist_ok=True)
    P = PROFILES[args.profile]
    t = load_times(log)
    names = sorted(f for f in os.listdir(images) if f.lower().endswith((".jpg", ".png")))
    orb = cv2.ORB_create(nfeatures=3000)
    rows, prev = [], None
    for i, n in enumerate(names):
        im = cv2.imread(os.path.join(images, n))
        s = 960 / max(im.shape[:2])
        g = cv2.cvtColor(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        th = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        rows.append(dict(name=n, time=t.get(n, ""), sharp=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                         orb=len(orb.detect(g, None)), bright=float(g.mean()),
                         dark=float((g < 15).mean()), sat=float((g > 250).mean()),
                         dup=float(np.abs(th - prev).mean()) if prev is not None else 99.0))
        prev = th
        if i % 50 == 0:
            print(f"  {i}/{len(names)}", end="\r", flush=True)
    print()
    sharp = np.array([r["sharp"] for r in rows])
    p = np.percentile(sharp, P["pct"])
    for i, r in enumerate(rows):
        lo, hi = max(0, i - 7), min(len(rows), i + 8)
        med = np.median(np.delete(sharp[lo:hi], i - lo)) if hi - lo > 1 else sharp[i]
        r["sharp_rel"] = r["sharp"] / (med + 1e-9)
        if P["rel_or_pct"]:
            blur = r["sharp_rel"] < P["rel"] or (r["sharp"] < p and r["sharp_rel"] < P.get("abs_rel", 9.0))
        else:
            blur = r["sharp_rel"] < P["rel"] and r["sharp"] < p
        expo = r["bright"] < P["dark"] or r["bright"] > P["bright"] or r["sat"] > 0.25
        dup = P["dup"] > 0 and r["dup"] < P["dup"]
        lowtex = r["orb"] < 150
        r["flags"] = "|".join(k for k, v in (("blur", blur), ("exposure", expo), ("dup", dup), ("lowtex", lowtex)) if v)
        r["remove"] = int(blur or expo or dup)  # lowtex는 표시만
    with open(os.path.join(qa, "qa_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "time", "sharp", "sharp_rel", "orb", "bright", "dark", "sat", "dup", "flags", "remove"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    rem = [r["name"] for r in rows if r["remove"]]
    with open(os.path.join(qa, "qa_remove_list.txt"), "w") as f:
        f.write("\n".join(rem) + ("\n" if rem else ""))

    # 확인용 그림: 제거 후보 + 텍스처 부족 표시 프레임
    def sheet(sel, tag):
        for k in range(0, len(sel), 40):
            tiles = []
            for r in sel[k:k + 40]:
                im = cv2.resize(cv2.imread(os.path.join(images, r["name"])), (320, 180), interpolation=cv2.INTER_AREA)
                tt = r["time"]
                ts = f"{int(tt // 60)}:{int(tt % 60):02d}" if tt != "" else ""
                cv2.rectangle(im, (0, 0), (320, 38), (0, 0, 0), -1)
                cv2.putText(im, f"{r['name'][6:11]} {ts} {r['flags']}", (3, 15), 0, 0.45, (0, 255, 255), 1)
                cv2.putText(im, f"sh{r['sharp']:.0f} rel{r['sharp_rel']:.2f} orb{r['orb']} dup{r['dup']:.1f}", (3, 32), 0, 0.42, (255, 255, 255), 1)
                tiles.append(im)
            while len(tiles) % 5:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(os.path.join(qa, f"qa_{tag}_{k // 40 + 1}.jpg"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
    for old in [f for f in os.listdir(qa) if f.startswith("qa_flagged_") or f.startswith("qa_lowtex_")]:
        os.remove(os.path.join(qa, old))
    sheet([r for r in rows if r["remove"]], "flagged")
    sheet([r for r in rows if "lowtex" in r["flags"] and not r["remove"]], "lowtex")

    from collections import Counter
    c = Counter(fl for r in rows for fl in r["flags"].split("|") if fl)
    print(f"[images] {len(rows)}  [remove candidates] {len(rem)}  flags: {dict(c)}")
    print(f"[sharp] median {np.median(sharp):.1f}  p{P['pct']} {p:.1f}")
    print(f"[out] {qa}/qa_report.csv, qa_remove_list.txt, qa_flagged_*.jpg, qa_lowtex_*.jpg")


def apply(args):
    images, root, qa, log = paths(args.images)
    rem = set(l.strip() for l in open(os.path.join(qa, "qa_remove_list.txt")) if l.strip())
    names = sorted(f for f in os.listdir(images) if f.lower().endswith((".jpg", ".png")))
    missing = rem - set(names)
    if missing:
        raise SystemExit(f"[error] 목록에 있는데 폴더에 없는 파일: {sorted(missing)[:5]} ... (이미 apply 했는지 확인)")
    t = load_times(log)
    rdir = images + "_qa_removed"
    os.makedirs(rdir, exist_ok=True)
    for n in sorted(rem):
        shutil.move(os.path.join(images, n), os.path.join(rdir, n))
    keep = [n for n in names if n not in rem]
    # 2단계 이름 변경 (충돌 방지)
    tmp = []
    for k, n in enumerate(keep, 1):
        ext = os.path.splitext(n)[1]
        tn = f"__tmp_{k:05d}{ext}"
        os.rename(os.path.join(images, n), os.path.join(images, tn))
        tmp.append((tn, f"frame_{k:05d}{ext}", n))
    for tn, new, _ in tmp:
        os.rename(os.path.join(images, tn), os.path.join(images, new))
    with open(os.path.join(qa, "qa_mapping.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "old_name", "time_sec"])
        for _, new, old in tmp:
            w.writerow([new, old, t.get(old, "")])
    if os.path.exists(log):
        shutil.copy(log, os.path.join(root, "extraction_log_before_qa.csv"))
        o2n = {old: new for _, new, old in tmp}
        rows = list(csv.DictReader(open(log)))
        with open(log, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                fn = r["filename"].strip()
                if r["status"].strip() == "keep":
                    if fn in o2n:
                        r["filename"] = o2n[fn]
                    else:
                        r["status"], r["filename"] = "qa_removed", fn + " (old)"
                w.writerow(r)
    print(f"[apply] removed {len(rem)} -> {rdir}")
    print(f"[apply] renumbered {len(keep)} images: frame_00001 ~ frame_{len(keep):05d}")
    print(f"[apply] mapping {qa}/qa_mapping.csv, log updated (backup: extraction_log_before_qa.csv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["analyze", "apply"])
    ap.add_argument("--images", required=True)
    ap.add_argument("--profile", choices=list(PROFILES), default="gs")
    args = ap.parse_args()
    analyze(args) if args.mode == "analyze" else apply(args)


if __name__ == "__main__":
    main()
