"""Trajectory evaluation: Sim(3)-aligned ATE and RPE for monocular VO (scale is unobservable)."""
import argparse

import numpy as np


def load(path):
    d = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if d.dtype.names is None or not {"frame", "x", "y", "z"}.issubset(d.dtype.names):
        raise ValueError(f"{path}: expected CSV columns frame,x,y,z")
    frame_values = d["frame"].astype(float)
    if not np.isfinite(frame_values).all() or not np.equal(frame_values, np.floor(frame_values)).all():
        raise ValueError(f"{path}: frame numbers must be finite integers")
    frames = frame_values.astype(int)
    positions = np.column_stack([d["x"], d["y"], d["z"]]).astype(float)
    if len(frames) < 2 or not np.isfinite(positions).all():
        raise ValueError(f"{path}: expected at least two finite poses")
    if len(np.unique(frames)) != len(frames):
        raise ValueError(f"{path}: frame numbers must be unique")
    return frames, positions


def umeyama(src, dst):
    """Least-squares s, R, t such that dst ~= s * R @ src + t."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 2:
        raise ValueError("source and destination must be matching N x 3 arrays with N >= 2")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("trajectories must contain only finite values")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    variance = (xs**2).sum() / len(src)
    if variance <= np.finfo(float).eps:
        raise ValueError("cannot align a stationary estimated trajectory")
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((xs ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def associate(est_f, est, gt_f, gt):
    _, ie, ig = np.intersect1d(est_f, gt_f, return_indices=True)
    return est[ie], gt[ig]


def align(est, gt):
    s, R, t = umeyama(est, gt)
    return s * est @ R.T + t, s


def ate_rmse(est, gt):
    a, s = align(est, gt)
    return float(np.sqrt(((a - gt) ** 2).sum(1).mean())), s


def rpe_trans(est, gt, delta=10):
    """Relative translation error over `delta` frames after Sim(3) alignment."""
    if delta < 1 or delta >= len(est):
        raise ValueError(f"delta must be between 1 and {len(est) - 1}")
    a, _ = align(est, gt)
    d = (a[delta:] - a[:-delta]) - (gt[delta:] - gt[:-delta])
    return float(np.sqrt((d ** 2).sum(1).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("estimate")
    ap.add_argument("groundtruth")
    ap.add_argument("--delta", type=int, default=10, help="frame interval for relative pose error")
    ap.add_argument("--plot", help="save a top-down plot (png)")
    a = ap.parse_args()
    try:
        ef, e = load(a.estimate)
        gf, g = load(a.groundtruth)
        e, g = associate(ef, e, gf, g)
        if len(e) < 2:
            ap.error("estimate and ground truth must share at least two frame numbers")
        ate, s = ate_rmse(e, g)
        rpe = rpe_trans(e, g, a.delta)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    print(f"poses={len(e)}  ATE RMSE={ate:.4f} m  RPE({a.delta}f)={rpe:.4f} m  recovered scale={s:.3f}")
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        al, _ = align(e, g)
        plt.plot(g[:, 0], g[:, 2], label="ground truth")
        plt.plot(al[:, 0], al[:, 2], "--", label="estimate (Sim3-aligned)")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        plt.legend()
        plt.title(f"ATE RMSE {ate:.3f} m")
        plt.savefig(a.plot, dpi=120)
