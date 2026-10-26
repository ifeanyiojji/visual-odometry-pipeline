"""Python reference implementation of the C++ pipeline (same algorithm; scipy replaces Ceres)."""
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Tracks:
    pt: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    pt0: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    T0: list = field(default_factory=list)  # birth pose (3x4) per track
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    has3d: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def select(self, m):
        self.pt, self.pt0, self.X, self.has3d = self.pt[m], self.pt0[m], self.X[m], self.has3d[m]
        self.T0 = [p for p, k in zip(self.T0, m) if k]


def to34(R, t):
    return np.hstack([R, np.reshape(t, (3, 1))])


def project(K, P, X):
    pc = X @ P[:, :3].T + P[:, 3]
    depth = pc[:, 2:3]
    safe_depth = np.where(np.abs(depth) < 1e-9, np.where(depth < 0, -1e-9, 1e-9), depth)
    return (pc[:, :2] / safe_depth) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]], pc[:, 2]


def triangulate(K, P0, P1, p0, p1, thresh=2.0):
    """Returns (X, ok) for N correspondences; rejects points behind a camera or with high reprojection error."""
    h = cv2.triangulatePoints(K @ P0, K @ P1, p0.T.astype(np.float64), p1.T.astype(np.float64))
    valid_w = np.abs(h[3]) >= 1e-12
    w = np.where(valid_w, h[3], 1.0)
    X = (h[:3] / w).T
    u0, z0 = project(K, P0, X)
    u1, z1 = project(K, P1, X)
    ok = valid_w.ravel() & np.isfinite(X).all(axis=1) & (z0 > 0) & (z1 > 0)
    ok &= (np.linalg.norm(u0 - p0, axis=1) < thresh) & (np.linalg.norm(u1 - p1, axis=1) < thresh)
    return X, ok


def refine_pose(K, X, uv, P_init):
    """Motion-only BA: minimise Huber-robust reprojection error over [rvec, t]."""
    x0 = np.concatenate([cv2.Rodrigues(P_init[:, :3])[0].ravel(), P_init[:, 3]])

    def res(x):
        u, _ = project(K, to34(cv2.Rodrigues(x[:3])[0], x[3:]), X)
        return (u - uv).ravel()

    s = least_squares(res, x0, loss="huber", f_scale=1.0, max_nfev=30)
    return to34(cv2.Rodrigues(s.x[:3])[0], s.x[3:])


class MonoVO:
    def __init__(self, K, max_features=500, min_init_parallax=30.0, min_pnp=20):
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("K must be a finite 3x3 camera matrix with positive focal lengths")
        if max_features < 50 or min_init_parallax < 0 or min_pnp < 4:
            raise ValueError("max_features must be >= 50, min_init_parallax >= 0, and min_pnp >= 4")
        self.K, self.max_features, self.min_init_parallax, self.min_pnp = K, max_features, min_init_parallax, min_pnp
        self.tr = Tracks()
        self.prev = None
        self.pose = to34(np.eye(3), np.zeros(3))
        self.initialized = False

    def _detect(self, gray):
        need = self.max_features - len(self.tr.pt)
        if need < self.max_features // 3:
            return
        mask = np.full(gray.shape, 255, np.uint8)
        for p in self.tr.pt:
            cv2.circle(mask, (int(p[0]), int(p[1])), 15, 0, -1)
        pts = cv2.goodFeaturesToTrack(gray, need, 0.01, 15, mask=mask)
        if pts is None:
            return
        pts = pts.reshape(-1, 2).astype(np.float32)
        t = self.tr
        t.pt, t.pt0 = np.vstack([t.pt, pts]), np.vstack([t.pt0, pts])
        t.T0 += [self.pose.copy()] * len(pts)
        t.X = np.vstack([t.X, np.zeros((len(pts), 3))])
        t.has3d = np.concatenate([t.has3d, np.zeros(len(pts), bool)])

    def _klt(self, gray):
        t = self.tr
        if len(t.pt) == 0:
            return
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, t.pt, None, winSize=(21, 21), maxLevel=3)
        if p1 is None or st is None:
            self.tr = Tracks()
            return
        back, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, winSize=(21, 21), maxLevel=3)
        if back is None or stb is None:
            self.tr = Tracks()
            return
        h, w = gray.shape
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (np.linalg.norm(back - t.pt, axis=1) < 1.0)
        ok &= (p1[:, 0] >= 0) & (p1[:, 1] >= 0) & (p1[:, 0] < w) & (p1[:, 1] < h)
        t.pt = p1
        t.select(ok)

    def _try_init(self, f):
        t, K = self.tr, self.K
        if len(t.pt) < 50:  # init tracks lost: restart from this frame
            self.tr = Tracks()
            return False
        if np.median(np.linalg.norm(t.pt - t.pt0, axis=1)) < self.min_init_parallax:
            return False
        E, m = cv2.findEssentialMat(t.pt0, t.pt, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            return False
        n, R, tv, m = cv2.recoverPose(E, t.pt0, t.pt, K, mask=m)
        if n < 50:
            return False
        X, ok = triangulate(K, to34(np.eye(3), np.zeros(3)), to34(R, tv.ravel()), t.pt0, t.pt)
        ok &= m.ravel() > 0
        if ok.sum() < 40:
            return False
        t.X, t.has3d = X, ok.copy()
        t.select(ok)
        self.pose = to34(R, tv.ravel())
        self.initialized = True
        print(f"initialized at frame {f} with {len(t.pt)} points", file=sys.stderr)
        return True

    def _track_pose(self):
        t, K = self.tr, self.K
        idx = np.flatnonzero(t.has3d)
        if len(idx) < self.min_pnp:
            return 0
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            t.X[idx], t.pt[idx].astype(np.float64), K, None, iterationsCount=100, reprojectionError=2.0, confidence=0.99
        )
        if not ok or inl is None or len(inl) < self.min_pnp:
            return 0
        sel = idx[inl.ravel()]
        self.pose = refine_pose(K, t.X[sel], t.pt[sel], to34(cv2.Rodrigues(rvec)[0], tvec.ravel()))
        # prune 3D points that disagree with the refined pose
        u, z = project(K, self.pose, t.X)
        bad = t.has3d & ~np.isin(np.arange(len(t.pt)), sel) & ((z <= 0) | (np.linalg.norm(u - t.pt, axis=1) > 3.0))
        t.select(~bad)
        return len(sel)

    def _grow_map(self):
        t = self.tr
        for i in np.flatnonzero(~t.has3d & (np.linalg.norm(t.pt - t.pt0, axis=1) >= 15.0)):
            X, ok = triangulate(self.K, t.T0[i], self.pose, t.pt0[i : i + 1], t.pt[i : i + 1])
            if ok[0]:
                t.X[i], t.has3d[i] = X[0], True

    def process(self, f, gray):
        """Returns the world->camera 3x4 pose, or None until initialised."""
        if gray is None or gray.ndim != 2 or gray.size == 0:
            raise ValueError(f"frame {f} must be a non-empty grayscale image")
        if self.prev is not None and gray.shape != self.prev.shape:
            raise ValueError(f"frame {f} dimensions do not match the previous frame")
        if self.prev is None:
            self._detect(gray)
            self.prev = gray
            return None
        self._klt(gray)
        if not self.initialized:
            self._try_init(f)
            if not self.initialized and len(self.tr.pt) == 0:
                self._detect(gray)
            self.prev = gray
            return self.pose if self.initialized else None
        if self._track_pose() == 0:
            print(f"frame {f}: tracking failed, keeping previous pose", file=sys.stderr)
        self._grow_map()
        self._detect(gray)
        self.prev = gray
        return self.pose


def camera_center(P):
    return -P[:, :3].T @ P[:, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("fx", type=float)
    ap.add_argument("fy", type=float)
    ap.add_argument("cx", type=float)
    ap.add_argument("cy", type=float)
    ap.add_argument("-o", "--out", default="trajectory_py.csv")
    a = ap.parse_args()
    if not a.image_dir.is_dir():
        ap.error(f"image directory does not exist: {a.image_dir}")
    if not np.isfinite([a.fx, a.fy, a.cx, a.cy]).all() or a.fx <= 0 or a.fy <= 0:
        ap.error("camera intrinsics must be finite and fx/fy must be positive")
    files = sorted(p for p in a.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(files) < 2:
        ap.error(f"need at least 2 images in {a.image_dir}")
    try:
        fh = open(a.out, "w", encoding="utf-8")
    except OSError as exc:
        ap.error(f"could not open output CSV {a.out}: {exc}")
    vo = MonoVO(np.array([[a.fx, 0, a.cx], [0, a.fy, a.cy], [0, 0, 1.0]]))
    with fh:
        fh.write("frame,x,y,z\n")
        for f, p in enumerate(files):
            image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if image is None:
                ap.error(f"could not read image: {p}")
            pose = vo.process(f, image)
            if pose is not None:
                c = camera_center(pose)
                fh.write(f"{f},{c[0]:.9f},{c[1]:.9f},{c[2]:.9f}\n")
    if not vo.initialized:
        ap.error("could not initialize: sequence did not provide enough tracked parallax")


if __name__ == "__main__":
    main()
