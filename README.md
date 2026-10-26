# Monocular Visual Odometry

A compact, reproducible visual-odometry project in C++17 using OpenCV and Ceres, with a Python reference implementation and a synthetic benchmark. It estimates camera motion from a calibrated monocular image sequence and exports the camera-center trajectory as CSV.

This is a learning and portfolio project, not a production or safety-critical localization system. Monocular translation has unknown scale; the reported trajectory is only meaningful up to a global similarity transform unless an external source provides metric scale.

## What it does

1. Detects spatially distributed corners with GFTT and tracks them with forward/backward pyramidal KLT.
2. Waits for useful image parallax, then estimates the initial relative pose with a RANSAC essential matrix.
3. Triangulates candidate landmarks and rejects points with invalid depth or high reprojection error.
4. Estimates each next camera pose from 2D–3D correspondences with PnP-RANSAC.
5. Refines the pose with motion-only bundle adjustment: Ceres with a Huber loss in C++, or SciPy's robust least-squares solver in Python.
6. Extends the local map as new tracks gain parallax, and writes camera centers to CSV.

The C++ and Python programs mirror the same front end and geometric thresholds so they can be compared on the same inputs; the optimizer iteration limits are library-specific. The Python version is useful for experiments; the C++ implementation is the primary deliverable.

## Quick start: Python

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python python\vo_ref.py data\synth 500 500 320 240 --out trajectory_py.csv
.\.venv\Scripts\python tools\evaluate.py trajectory_py.csv data\synth\groundtruth.csv
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The checked-in sequence is synthetic and generated with `tools/synth.py`; its intrinsics are `fx=fy=500`, `cx=320`, `cy=240`. To generate another sequence:

```powershell
.\.venv\Scripts\python tools\synth.py data\my_sequence --frames 120 --seed 0
```

`tools/evaluate.py` associates poses by frame ID, performs a Sim(3) alignment, and reports aligned absolute trajectory error (ATE) and relative translation error (RPE). On the checked-in sequence, the Python reference currently produces 110 poses, about 0.145 m ATE RMSE and 0.103 m 10-frame RPE. These are reproducibility numbers for this synthetic sequence, not a claim of real-world or benchmark performance.

## Build and run: C++

Install CMake, the Visual Studio 2022 C++ build tools and Windows SDK, and vcpkg. In a Visual Studio 2022 Developer PowerShell:

```powershell
$env:VCPKG_ROOT = "$env:USERPROFILE\vcpkg"
& "$env:VCPKG_ROOT\vcpkg.exe" install --x-install-root="$env:VCPKG_ROOT\installed" --triplet=x64-windows
cmake -S . -B build-cpu -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_INSTALLED_DIR="$env:VCPKG_ROOT\installed"
cmake --build build-cpu --config Release
.\build-cpu\mono_vo.exe data\synth 500 500 320 240 trajectory.csv --no-gui
.\.venv\Scripts\python tools\evaluate.py trajectory.csv data\synth\groundtruth.csv
```

The vcpkg manifest installs only the OpenCV modules used by this project and disables the GPU-oriented and DNN defaults; the C++ visual-odometry pipeline runs on the CPU.

The executable accepts:

```text
mono_vo <image_dir> <fx> <fy> <cx> <cy> [out.csv] [--no-gui]
```

Images are read in lexicographic filename order. Use zero-padded filenames (as in the included data) so the order is chronological. With the GUI enabled, Esc exits the live track/trajectory viewer. `--no-gui` is intended for headless runs.

## Input and output contract

- Input: at least two readable, same-sized grayscale or color PNG/JPEG images, plus finite pinhole intrinsics (`fx`, `fy`, `cx`, `cy`) in pixels. Images are converted to grayscale.
- Output: `frame,x,y,z`, where `(x,y,z)` is the camera center in the initial camera coordinate frame. Frames before initialization are omitted.
- The essential matrix fixes the initial translation baseline to unit length. Metric scale is not observable from a monocular camera alone.
- Invalid calibration, unreadable or inconsistent images, invalid output paths, insufficient sequence length, and failed initialization produce explicit errors rather than a plausible-looking trajectory.

## Project layout

```text
src/main.cpp          C++ OpenCV + Ceres implementation
python/vo_ref.py      Python OpenCV + SciPy reference
tools/synth.py        deterministic synthetic image and ground-truth generator
tools/evaluate.py     Sim(3)-aligned ATE/RPE evaluator
data/synth/           checked-in synthetic sequence and ground truth
tests/                focused unit and end-to-end tests
```

## Engineering notes and limitations

- The tracker uses forward/backward consistency to remove many KLT failures; PnP-RANSAC limits the effect of mismatches.
- Triangulation checks cheirality and reprojection error. Pose refinement uses a robust Huber loss and falls back to the PnP estimate if Ceres cannot produce a usable solution.
- If frame-level PnP tracking fails, the implementation logs the failure and keeps the last valid pose. Long gaps, abrupt motion, blur, low texture, rolling shutter, and dynamic objects can still cause drift or tracking loss.
- This is a small local mapper, not a full SLAM system: it has no loop closure, global bundle adjustment, relocalization, learned features, or inertial sensor fusion. It does not model lens distortion; undistort images and update intrinsics before processing distorted-camera data.
- The synthetic renderer is useful for repeatable smoke tests, but its point-cloud imagery does not represent the full appearance or failure modes of real driving footage.

## Portfolio walkthrough

For a project review, explain the coordinate convention (world-to-camera poses internally, camera centers in the CSV), why monocular scale is ambiguous, how RANSAC and reprojection checks reject outliers, and how the synthetic ground truth makes regression tests measurable. A natural next step is to evaluate on a public sequence such as KITTI after undistortion and calibration, report results over multiple sequences, and characterize failure cases rather than tuning against one run.
