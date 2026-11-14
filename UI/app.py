import argparse
import os
import re
import sys
import threading
import time

import numpy as np
import viser
import viser.transforms as vtf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lift3d.lift_to_3d import read_colmap, qvec2rotmat  # noqa: E402
from src import utils  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ply", required=True, help="3DGS point_cloud.ply")
parser.add_argument("--colmap", required=True, help="COLMAP 모델 폴더 (gs_data/sparse/0, .bin 또는 .txt)")
parser.add_argument("--objects", required=True, help="lift3d가 만든 objects.json")
parser.add_argument("--location", default="1공학관 C302호_연구실창고", help="UI 상단에 표시할 장소 이름")
parser.add_argument("--home-image", default=None, help="'처음 시점'으로 쓸 이미지 파일명 (안 주면 제일 앞 번호 프레임)")
parser.add_argument("--max-points", type=int, default=400000, help="렌더링할 최대 가우시안 수 (성능용 서브샘플링)")
parser.add_argument("--min-opacity", type=float, default=0.05)
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()

SH_C0 = 0.28209479177387814
UP_HINT_AXIS = np.array([0.0, 0.0, 1.0])


# ----------------------------------------------------------------- 3DGS ply (xyz+회전+스케일+불투명도+색) 읽기
def read_gaussians_full(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                t, name = line.split()[1:3]
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1"}[t]))
        a = np.fromfile(f, dtype=np.dtype(props), count=n)
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    rot = np.stack([a["rot_0"], a["rot_1"], a["rot_2"], a["rot_3"]], 1).astype(np.float64)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9
    scale = np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-a["opacity"].astype(np.float64)))
    rgb = np.clip(0.5 + SH_C0 * np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1), 0, 1)
    return xyz, rot, scale, opacity, rgb


def quats_to_covariances(rot_wxyz, scale):
    # cov = R @ diag(scale^2) @ R^T, 가우시안마다 3x3
    n = rot_wxyz.shape[0]
    w, x, y, z = rot_wxyz[:, 0], rot_wxyz[:, 1], rot_wxyz[:, 2], rot_wxyz[:, 3]
    R = np.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y,
        2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x,
        2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y,
    ], axis=1).reshape(n, 3, 3)
    S2 = scale ** 2
    cov = np.einsum("nij,nj,nkj->nik", R, S2, R)
    return cov


# ----------------------------------------------------------------- 로드
print("[load] ply ...")
xyz, rot, scale, opacity, rgb = read_gaussians_full(args.ply)
keep = opacity >= args.min_opacity
xyz, rot, scale, opacity, rgb = xyz[keep], rot[keep], scale[keep], opacity[keep], rgb[keep]
if len(xyz) > args.max_points:
    idx = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
    xyz, rot, scale, opacity, rgb = xyz[idx], rot[idx], scale[idx], opacity[idx], rgb[idx]
covariances = quats_to_covariances(rot, scale)
print(f"[load] 가우시안 {len(xyz)}개 렌더링")

print("[load] colmap ...")
cams, imgs = read_colmap(args.colmap)


def frame_num(name):
    m = re.findall(r"(\d+)", name)
    return int(m[-1]) if m else 1 << 30


home_name = args.home_image or min(imgs, key=frame_num)
scene_up = np.mean([-qvec2rotmat(q).T[:, 1] for q, _, _ in imgs.values()], axis=0)
scene_up /= np.linalg.norm(scene_up)


def image_pose(name):
    q, t, _ = imgs[name]
    R = qvec2rotmat(q)
    C = -R.T @ t
    look_at = C + R.T @ np.array([0.0, 0.0, 1.0])
    return C, look_at


home_pos, home_lookat = image_pose(home_name)
print(f"[load] 처음 시점 = {home_name}")

print("[load] objects.json ...")
objects = utils.load_json(args.objects)["objects"]


def quat_from_axes(axes_rows):
    # axes_rows: 지역->세계 변환의 행벡터(axes.T 가 세계 기준 회전행렬)
    return vtf.SO3.from_matrix(np.asarray(axes_rows).T).wxyz


# ----------------------------------------------------------------- viser 서버
server = viser.ViserServer(port=args.port)
server.scene.add_gaussian_splats("/gaussians", centers=xyz.astype(np.float32), covariances=covariances.astype(np.float32),
                                  rgbs=rgb.astype(np.float32), opacities=opacity[:, None].astype(np.float32))

box_handles, label_handles = {}, {}
for key, obj in objects.items():
    obb = obj["obb"]
    box_handles[key] = server.scene.add_box(f"/obj_{key}/box", color=(255, 220, 0),
                                             dimensions=tuple(obb["extents"]), wireframe=True,
                                             position=tuple(obb["center"]), wxyz=quat_from_axes(obb["axes"]),
                                             visible=False)
    label_handles[key] = server.scene.add_label(f"/obj_{key}/label", obj.get("name_ko") or key,
                                                 position=tuple(np.array(obj["center"]) + scene_up * (obb["extents"][2] / 2 + 0.05)),
                                                 visible=False)

server.scene.set_up_direction(tuple(scene_up))
server.initial_camera.position = tuple(home_pos)
server.initial_camera.look_at = tuple(home_lookat)

# ----------------------------------------------------------------- GUI
server.gui.add_markdown(f"## {args.location}")
server.gui.add_divider()

key2label = {k: f"{objects[k].get('name_ko', k)} ({k})" for k in objects}
label2key = {v: k for k, v in key2label.items()}
dropdown = server.gui.add_dropdown("목록에서 찾기", list(key2label.values()))
search_text = server.gui.add_text("이름으로 검색", "")
search_btn = server.gui.add_button("검색")
home_btn = server.gui.add_button("처음 시점")
duration_slider = server.gui.add_slider("이동 시간(초)", min=1.0, max=10.0, step=0.5, initial_value=4.0)
sway_checkbox = server.gui.add_checkbox("도착 후 좌우로 살짝 둘러보기", True)
status_md = server.gui.add_markdown("")

# ----------------------------------------------------------------- 애니메이션 상태
state_lock = threading.Lock()
state = {"mode": "idle", "t0": 0.0, "dur": 4.0,
          "pos0": home_pos, "look0": home_lookat, "pos1": home_pos, "look1": home_lookat,
          "sway_center": None, "sway_offset": None, "sway_t0": 0.0, "active_key": None}
SWAY_DEG = 17.5
SWAY_PERIOD = 6.0


def ease(t):
    return t * t * (3 - 2 * t)


def start_move(pos1, look1, key=None):
    clients = list(server.get_clients().values())
    pos0 = clients[0].camera.position.copy() if clients else state["pos1"]
    look0 = clients[0].camera.look_at.copy() if clients else state["look1"]
    with state_lock:
        state.update(mode="moving", t0=time.time(), dur=duration_slider.value,
                      pos0=pos0, look0=look0, pos1=np.asarray(pos1, dtype=np.float64),
                      look1=np.asarray(look1, dtype=np.float64), active_key=key)
    for k, box in box_handles.items():
        box.visible = (k == key)
        label_handles[k].visible = (k == key)


def goto_object(key):
    obj = objects[key]
    c = obj["closeup"]
    start_move(c["position"], c["look_at"], key=key)
    status_md.content = f"**{key2label[key]}** 위치로 이동 중"


def goto_home():
    start_move(home_pos, home_lookat, key=None)
    status_md.content = "처음 시점으로 이동 중"


@dropdown.on_update
def _(_):
    goto_object(label2key[dropdown.value])


@search_btn.on_click
def _(_):
    q = search_text.value.strip().lower()
    hit = next((k for k in objects if q in k.lower() or q in str(objects[k].get("name_ko", "")).lower()
                or any(q in a.lower() for a in objects[k].get("aliases", []))), None)
    if hit:
        goto_object(hit)
    else:
        status_md.content = f"'{search_text.value}' 를 찾을 수 없음"


@home_btn.on_click
def _(_):
    goto_home()


@sway_checkbox.on_update
def _(_):
    if not sway_checkbox.value and state["mode"] == "sway":
        with state_lock:
            state["mode"] = "idle"


# ----------------------------------------------------------------- 메인 루프 (카메라 이동/둘러보기 연출)
def animate_loop():
    while True:
        time.sleep(1 / 30)
        with state_lock:
            mode, t0, dur = state["mode"], state["t0"], state["dur"]
            pos0, look0, pos1, look1 = state["pos0"], state["look0"], state["pos1"], state["look1"]
            active_key = state["active_key"]

        if mode == "moving":
            t = min(1.0, (time.time() - t0) / max(dur, 1e-6))
            e = ease(t)
            bump = 0.15 * np.linalg.norm(pos1 - pos0) * np.sin(np.pi * t)
            pos = pos0 + (pos1 - pos0) * e + scene_up * bump
            look = look0 + (look1 - look0) * e
            for c in server.get_clients().values():
                c.camera.position = pos
                c.camera.look_at = look
            if t >= 1.0:
                with state_lock:
                    if active_key is not None and sway_checkbox.value:
                        state["mode"] = "sway"
                        state["sway_center"] = objects[active_key]["closeup"]["look_at"]
                        state["sway_offset"] = pos1 - np.asarray(objects[active_key]["closeup"]["look_at"])
                        state["sway_t0"] = time.time()
                    else:
                        state["mode"] = "idle"
                if active_key is not None:
                    status_md.content = f"**{key2label[active_key]}** 도착"
                else:
                    status_md.content = "처음 시점 도착"

        elif mode == "sway":
            with state_lock:
                center, offset, t0s = state["sway_center"], state["sway_offset"], state["sway_t0"]
            ang = np.radians(SWAY_DEG) * np.sin(2 * np.pi * (time.time() - t0s) / SWAY_PERIOD)
            k = scene_up / (np.linalg.norm(scene_up) + 1e-9)
            v = offset
            v_rot = v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * (k @ v) * (1 - np.cos(ang))
            pos = np.asarray(center) + v_rot
            for c in server.get_clients().values():
                c.camera.position = pos
                c.camera.look_at = center


threading.Thread(target=animate_loop, daemon=True).start()
print(f"[viewer] http://localhost:{args.port} 에서 접속하세요")
server.sleep_forever()
