// Minimal monocular visual odometry: KLT tracking + essential-matrix init,
// PnP-RANSAC per frame, motion-only bundle adjustment with Ceres.
// Scale is arbitrary (fixed by the unit-length initial baseline).
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Pose = cv::Matx44d;  // world -> camera

struct Track {
  cv::Point2f pt;   // current pixel
  cv::Point2f pt0;  // pixel where the track was born
  Pose T0;          // camera pose when the track was born
  cv::Point3d X;    // world point (valid if has3d)
  bool has3d = false;
};

struct ReprojError {
  ReprojError(const cv::Point3d& X, const cv::Point2f& uv, const cv::Matx33d& K, int component)
      : X_(X), uv_(uv), fx_(K(0, 0)), fy_(K(1, 1)), cx_(K(0, 2)), cy_(K(1, 2)), component_(component) {}

  template <typename T>
  bool operator()(const T* pose, T* r) const {
    const T p[3] = {T(X_.x), T(X_.y), T(X_.z)};
    T pc[3];
    ceres::AngleAxisRotatePoint(pose, p, pc);
    pc[0] += pose[3];
    pc[1] += pose[4];
    pc[2] += pose[5];
    const T depth = pc[2] >= T(0)
                        ? (pc[2] < T(1e-9) ? T(1e-9) : pc[2])
                        : (pc[2] > T(-1e-9) ? T(-1e-9) : pc[2]);
    const T rx = T(fx_) * pc[0] / depth + T(cx_) - T(uv_.x);
    const T ry = T(fy_) * pc[1] / depth + T(cy_) - T(uv_.y);
    r[0] = component_ == 0 ? rx : ry;
    return true;
  }

  cv::Point3d X_;
  cv::Point2f uv_;
  double fx_, fy_, cx_, cy_;
  int component_;
};

static Pose toPose(const cv::Mat& R, const cv::Mat& t) {
  Pose T = Pose::eye();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) T(i, j) = R.at<double>(i, j);
    T(i, 3) = t.at<double>(i);
  }
  return T;
}

static cv::Matx33d rot(const Pose& T) {
  return cv::Matx33d(T(0, 0), T(0, 1), T(0, 2), T(1, 0), T(1, 1), T(1, 2), T(2, 0), T(2, 1), T(2, 2));
}

static cv::Matx34d top(const Pose& T) {
  return cv::Matx34d(T(0, 0), T(0, 1), T(0, 2), T(0, 3), T(1, 0), T(1, 1), T(1, 2), T(1, 3),
                     T(2, 0), T(2, 1), T(2, 2), T(2, 3));
}

static bool project(const cv::Matx33d& K, const Pose& T, const cv::Point3d& X, cv::Point2d& uv) {
  cv::Vec3d pc = top(T) * cv::Vec4d(X.x, X.y, X.z, 1.0);
  if (pc[2] <= 1e-6) return false;
  uv = {K(0, 0) * pc[0] / pc[2] + K(0, 2), K(1, 1) * pc[1] / pc[2] + K(1, 2)};
  return true;
}

// Triangulate with two poses; rejects points behind a camera or with large reprojection error.
static bool triangulate(const cv::Matx33d& K, const Pose& T0, const Pose& T1, const cv::Point2f& p0,
                        const cv::Point2f& p1, cv::Point3d& X) {
  cv::Mat P0(K * top(T0)), P1(K * top(T1));
  P0.convertTo(P0, CV_64F);
  P1.convertTo(P1, CV_64F);
  cv::Mat a(2, 1, CV_64F), b(2, 1, CV_64F);
  a.at<double>(0, 0) = p0.x;
  a.at<double>(1, 0) = p0.y;
  b.at<double>(0, 0) = p1.x;
  b.at<double>(1, 0) = p1.y;
  cv::Mat h;
  cv::triangulatePoints(P0, P1, a, b, h);
  h.convertTo(h, CV_64F);
  double w = h.at<double>(3);
  if (std::abs(w) < 1e-12) return false;
  X = {h.at<double>(0) / w, h.at<double>(1) / w, h.at<double>(2) / w};
  cv::Point2d u0, u1;
  if (!project(K, T0, X, u0) || !project(K, T1, X, u1)) return false;
  return cv::norm(u0 - cv::Point2d(p0)) < 2.0 && cv::norm(u1 - cv::Point2d(p1)) < 2.0;
}

// Motion-only BA: refine camera pose with Huber-robustified reprojection error.
static Pose refinePose(const cv::Matx33d& K, const std::vector<const Track*>& obs, const Pose& init) {
  double pose[6];
  cv::Vec3d rvec;
  cv::Rodrigues(rot(init), rvec);
  for (int i = 0; i < 3; ++i) {
    pose[i] = rvec[i];
    pose[3 + i] = init(i, 3);
  }
  ceres::Problem problem;
  for (const Track* t : obs) {
    for (int component = 0; component < 2; ++component) {
      auto* cost = new ceres::AutoDiffCostFunction<ReprojError, 1, 6>(new ReprojError(t->X, t->pt, K, component));
      problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose);
    }
  }
  ceres::Solver::Options opt;
  opt.linear_solver_type = ceres::DENSE_QR;
  opt.max_num_iterations = 30;
  ceres::Solver::Summary summary;
  ceres::Solve(opt, &problem, &summary);
  if (!summary.IsSolutionUsable() ||
      !std::all_of(pose, pose + 6, [](double v) { return std::isfinite(v); })) {
    std::cerr << "pose refinement failed; using the PnP estimate\n";
    return init;
  }

  cv::Vec3d rv(pose[0], pose[1], pose[2]);
  cv::Matx33d Rn;
  cv::Rodrigues(rv, Rn);
  Pose T = Pose::eye();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) T(i, j) = Rn(i, j);
    T(i, 3) = pose[3 + i];
  }
  return T;
}

static void detect(const cv::Mat& gray, std::vector<Track>& tracks, const Pose& T, int maxTotal) {
  int need = maxTotal - static_cast<int>(tracks.size());
  if (need < maxTotal / 3) return;
  cv::Mat mask(gray.size(), CV_8U, cv::Scalar(255));
  for (const auto& t : tracks) cv::circle(mask, t.pt, 15, cv::Scalar(0), -1);
  std::vector<cv::Point2f> pts;
  cv::goodFeaturesToTrack(gray, pts, need, 0.01, 15, mask);
  for (const auto& p : pts) {
    Track t;
    t.pt = t.pt0 = p;
    t.T0 = T;
    tracks.push_back(t);
  }
}

// Forward-backward KLT; drops failed, inconsistent, or out-of-image tracks.
static void trackKLT(const cv::Mat& prev, const cv::Mat& cur, std::vector<Track>& tracks) {
  if (tracks.empty()) return;
  std::vector<cv::Point2f> p0, p1, back;
  for (const auto& t : tracks) p0.push_back(t.pt);
  std::vector<uchar> st, stb;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prev, cur, p0, p1, st, err, {21, 21}, 3);
  cv::calcOpticalFlowPyrLK(cur, prev, p1, back, stb, err, {21, 21}, 3);
  std::vector<Track> kept;
  for (size_t i = 0; i < tracks.size(); ++i) {
    bool ok = st[i] && stb[i] && cv::norm(back[i] - p0[i]) < 1.0f && p1[i].x >= 0 && p1[i].y >= 0 &&
              p1[i].x < cur.cols && p1[i].y < cur.rows;
    if (!ok) continue;
    tracks[i].pt = p1[i];
    kept.push_back(tracks[i]);
  }
  tracks.swap(kept);
}

static cv::Point3d cameraCenter(const Pose& T) {
  cv::Vec3d c = -(rot(T).t() * cv::Vec3d(T(0, 3), T(1, 3), T(2, 3)));
  return {c[0], c[1], c[2]};
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "usage: mono_vo <image_dir> <fx> <fy> <cx> <cy> [out.csv] [--no-gui]\n";
    return 0;
  }
  if (argc < 6) {
    std::cerr << "usage: mono_vo <image_dir> <fx> <fy> <cx> <cy> [out.csv] [--no-gui]\n";
    return 1;
  }
  std::string dir = argv[1];
  double fx, fy, cx, cy;
  try {
    auto parseNumber = [](const char* text) {
      size_t parsed = 0;
      const double value = std::stod(text, &parsed);
      if (parsed != std::string(text).size()) throw std::invalid_argument("unexpected trailing characters");
      return value;
    };
    fx = parseNumber(argv[2]);
    fy = parseNumber(argv[3]);
    cx = parseNumber(argv[4]);
    cy = parseNumber(argv[5]);
  } catch (const std::exception& e) {
    std::cerr << "invalid camera intrinsics: " << e.what() << "\n";
    return 1;
  }
  if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) || !std::isfinite(cy) || fx <= 0 || fy <= 0) {
    std::cerr << "camera intrinsics must be finite and fx/fy must be positive\n";
    return 1;
  }
  const cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
  std::string outCsv = "trajectory.csv";
  bool gui = true;
  bool outputSet = false;
  for (int i = 6; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--no-gui")
      gui = false;
    else if (!arg.empty() && arg[0] == '-') {
      std::cerr << "unknown option: " << arg << "\n";
      return 1;
    } else if (!outputSet) {
      outCsv = arg;
      outputSet = true;
    } else {
      std::cerr << "only one output CSV path may be specified\n";
      return 1;
    }
  }

  std::vector<std::string> files;
  for (const char* ext : {"png", "jpg", "jpeg"}) {
    std::vector<std::string> f;
    cv::glob(dir + "/*." + ext, f, false);
    files.insert(files.end(), f.begin(), f.end());
  }
  std::sort(files.begin(), files.end());
  if (files.size() < 2) {
    std::cerr << "need at least 2 images in " << dir << "\n";
    return 1;
  }

  const int kMaxFeatures = 500, kMinPnP = 20;
  cv::Mat prev = cv::imread(files[0], cv::IMREAD_GRAYSCALE), cur;
  if (prev.empty()) {
    std::cerr << "could not read image: " << files[0] << "\n";
    return 1;
  }
  std::vector<Track> tracks;
  Pose pose = Pose::eye();
  detect(prev, tracks, pose, kMaxFeatures);

  std::ofstream csv(outCsv);
  if (!csv) {
    std::cerr << "could not open output CSV: " << outCsv << "\n";
    return 1;
  }
  csv << std::setprecision(9);
  csv << "frame,x,y,z\n";
  int written = 0;
  bool initialized = false;
  cv::Mat canvas(700, 700, CV_8UC3, cv::Scalar::all(255));
  const double kViewScale = 2.0;

  for (size_t f = 1; f < files.size(); ++f) {
    cur = cv::imread(files[f], cv::IMREAD_GRAYSCALE);
    if (cur.empty()) {
      std::cerr << "could not read image: " << files[f] << "\n";
      return 1;
    }
    if (cur.size() != prev.size()) {
      std::cerr << "image dimensions changed at frame " << f << "\n";
      return 1;
    }
    trackKLT(prev, cur, tracks);

    if (!initialized) {
      // Wait for enough parallax, then bootstrap from the essential matrix.
      if (tracks.size() < 50) {  // init tracks lost: restart from this frame
        tracks.clear();
        detect(cur, tracks, Pose::eye(), kMaxFeatures);
        prev = cur;
        continue;
      }
      std::vector<double> disp;
      for (const auto& t : tracks) disp.push_back(cv::norm(t.pt - t.pt0));
      std::nth_element(disp.begin(), disp.begin() + disp.size() / 2, disp.end());
      if (disp[disp.size() / 2] >= 30.0) {
        std::vector<cv::Point2f> a, b;
        for (const auto& t : tracks) {
          a.push_back(t.pt0);
          b.push_back(t.pt);
        }
        cv::Mat mask, R, tvec;
        cv::Mat E = cv::findEssentialMat(a, b, cv::Mat(K), cv::RANSAC, 0.999, 1.0, mask);
        int n = E.empty() ? 0 : cv::recoverPose(E, a, b, cv::Mat(K), R, tvec, mask);
        if (n >= 50) {
          Pose T1 = toPose(R, tvec);
          std::vector<Track> kept;
          for (size_t i = 0; i < tracks.size(); ++i) {
            if (!mask.at<uchar>(static_cast<int>(i))) continue;
            if (triangulate(K, Pose::eye(), T1, tracks[i].pt0, tracks[i].pt, tracks[i].X)) {
              tracks[i].has3d = true;
              kept.push_back(tracks[i]);
            }
          }
          if (kept.size() >= 40) {
            tracks.swap(kept);
            pose = T1;
            initialized = true;
            std::cout << "initialized at frame " << f << " with " << tracks.size() << " points\n";
            const cv::Point3d c = cameraCenter(pose);
            csv << f << "," << c.x << "," << c.y << "," << c.z << "\n";
            ++written;
          }
        }
      }
      prev = cur;
      continue;
    }

    // Pose from 2D-3D correspondences, then Ceres refinement on inliers.
    std::vector<cv::Point3d> obj;
    std::vector<cv::Point2f> img;
    std::vector<int> idx;
    for (size_t i = 0; i < tracks.size(); ++i) {
      if (!tracks[i].has3d) continue;
      obj.emplace_back(tracks[i].X);
      img.push_back(tracks[i].pt);
      idx.push_back(static_cast<int>(i));
    }
    int inliers = 0;
    if (obj.size() >= static_cast<size_t>(kMinPnP)) {
      cv::Mat rvec, tvec;
      std::vector<int> inl;
      bool ok = cv::solvePnPRansac(obj, img, cv::Mat(K), cv::noArray(), rvec, tvec, false, 100, 2.0, 0.99, inl);
      if (ok && inl.size() >= static_cast<size_t>(kMinPnP)) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        std::vector<const Track*> good;
        std::vector<char> isIn(tracks.size(), 0);
        for (int k : inl) {
          good.push_back(&tracks[idx[k]]);
          isIn[idx[k]] = 1;
        }
        pose = refinePose(K, good, toPose(R, tvec));
        inliers = static_cast<int>(inl.size());
        // Drop 3D points that disagree with the refined pose.
        std::vector<Track> kept;
        for (size_t i = 0; i < tracks.size(); ++i) {
          if (tracks[i].has3d && !isIn[i]) {
            cv::Point2d uv;
            if (!project(K, pose, tracks[i].X, uv) || cv::norm(uv - cv::Point2d(tracks[i].pt)) > 3.0) continue;
          }
          kept.push_back(tracks[i]);
        }
        tracks.swap(kept);
      }
    }
    if (inliers == 0) std::cerr << "frame " << f << ": tracking failed, keeping previous pose\n";

    // Triangulate tracks that now have enough parallax relative to where they were born.
    for (auto& t : tracks) {
      if (t.has3d || cv::norm(t.pt - t.pt0) < 15.0) continue;
      if (triangulate(K, t.T0, pose, t.pt0, t.pt, t.X)) t.has3d = true;
    }
    detect(cur, tracks, pose, kMaxFeatures);

    cv::Point3d c = cameraCenter(pose);
    csv << f << "," << c.x << "," << c.y << "," << c.z << "\n";
    ++written;

    if (gui) {
      cv::Point p(350 + static_cast<int>(c.x * kViewScale), 600 - static_cast<int>(c.z * kViewScale));
      cv::circle(canvas, p, 1, cv::Scalar(0, 0, 255), -1);
      cv::Mat vis;
      cv::cvtColor(cur, vis, cv::COLOR_GRAY2BGR);
      for (const auto& t : tracks)
        cv::circle(vis, t.pt, 2, t.has3d ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255), -1);
      cv::putText(vis, "frame " + std::to_string(f) + " inliers " + std::to_string(inliers), {10, 20},
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 2);
      cv::imshow("tracks", vis);
      cv::imshow("trajectory (x right, z up)", canvas);
      if (cv::waitKey(1) == 27) break;
    }
    prev = cur;
  }
  if (!initialized) {
    std::cerr << "could not initialize: sequence did not provide enough tracked parallax\n";
    return 1;
  }
  if (!csv) {
    std::cerr << "failed while writing output CSV: " << outCsv << "\n";
    return 1;
  }
  std::cout << "wrote " << written << " poses to " << outCsv << "\n";
  return 0;
}
