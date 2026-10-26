import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate import align, associate, rpe_trans


ROOT = Path(__file__).resolve().parents[1]


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_similarity_alignment_recovers_scale_and_translation(self):
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 2.0, 3.0]]
        )
        angle = 0.4
        rotation = np.array(
            [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
        )
        target = 2.5 * source @ rotation.T + [4.0, -2.0, 7.0]

        aligned, scale = align(source, target)

        self.assertAlmostEqual(scale, 2.5, places=10)
        np.testing.assert_allclose(aligned, target, atol=1e-10)

    def test_association_uses_matching_frame_ids(self):
        estimate = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        ground_truth = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        actual, expected = associate([2, 4], estimate, [1, 2, 4], ground_truth)

        np.testing.assert_array_equal(actual, estimate)
        np.testing.assert_array_equal(expected, ground_truth[1:])

    def test_rpe_rejects_an_interval_without_pairs(self):
        with self.assertRaisesRegex(ValueError, "delta"):
            rpe_trans(np.zeros((3, 3)), np.zeros((3, 3)), delta=3)


class SyntheticPipelineTests(unittest.TestCase):
    def test_python_pipeline_tracks_checked_in_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "trajectory.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "python" / "vo_ref.py"),
                    str(ROOT / "data" / "synth"),
                    "500",
                    "500",
                    "320",
                    "240",
                    "--out",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output.is_file())

            evaluation = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "evaluate.py"),
                    str(output),
                    str(ROOT / "data" / "synth" / "groundtruth.csv"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(evaluation.returncode, 0, msg=evaluation.stderr)
            poses = re.search(r"poses=(\d+)", evaluation.stdout)
            ate = re.search(r"ATE RMSE=([0-9.]+) m", evaluation.stdout)
            rpe = re.search(r"RPE\(10f\)=([0-9.]+) m", evaluation.stdout)
            self.assertIsNotNone(poses, msg=evaluation.stdout)
            self.assertIsNotNone(ate, msg=evaluation.stdout)
            self.assertIsNotNone(rpe, msg=evaluation.stdout)
            self.assertGreaterEqual(int(poses.group(1)), 100)
            self.assertLess(float(ate.group(1)), 0.20)
            self.assertLess(float(rpe.group(1)), 0.15)


if __name__ == "__main__":
    unittest.main()
