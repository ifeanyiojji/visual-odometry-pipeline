"""Synthetic monocular sequence with exact ground truth (no dataset download needed)."""
import argparse
from pathlib import Path

import cv2
import numpy as np

W, H, FX = 640, 480, 500.0
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])


def camera_poses(n):
    """World->camera poses (R, t) for a gently curving forward trajectory (camera: x right, y down, z fwd)."""
    poses = []
    for i in range(n):
        s = i * 0.35
        c = np.array([2.0 * np.sin(0.05 * s), 0.2 * np.sin(0.11 * s), s])
        yaw = 0.25 * np.sin(0.05 * s)
        R_cw = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0]  # camera->world
        R = R_cw.T
        poses.append((R, -R @ c))
    return poses


def render(points, gray_vals, radii, R, t):
    pc = points @ R.T + t
    img = np.full((H, W), 20, np.uint8)
    order = np.argsort(-pc[:, 2])  # far to near
    for i in order:
        z = pc[i, 2]
        if z < 1.5:
            continue
        u = int(FX * pc[i, 0] / z + W / 2)
        v = int(FX * pc[i, 1] / z + H / 2)
        if 0 <= u < W and 0 <= v < H:
            r = int(np.clip(radii[i] * FX / z, 2, 9))
            cv2.circle(img, (u, v), r, int(gray_vals[i]), -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


def generate(out_dir, n=120, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zmax = n * 0.35 + 40
    points = np.column_stack(
        [rng.uniform(-12, 12, 9000), rng.uniform(-4, 4, 9000), rng.uniform(0, zmax, 9000)]
    )
    gray = rng.integers(60, 255, len(points))
    radii = rng.uniform(0.02, 0.06, len(points))
    gt = []
    for i, (R, t) in enumerate(camera_poses(n)):
        cv2.imwrite(str(out / f"{i:06d}.png"), render(points, gray, radii, R, t))
        gt.append([i, *(-R.T @ t)])
    np.savetxt(out / "groundtruth.csv", gt, delimiter=",", header="frame,x,y,z", comments="",
               fmt=["%d", "%.9f", "%.9f", "%.9f"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.out_dir, a.frames, a.seed)
    print(f"fx=fy={FX} cx={W / 2} cy={H / 2}")
