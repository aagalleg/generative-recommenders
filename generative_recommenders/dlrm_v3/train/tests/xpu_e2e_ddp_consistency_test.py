# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only regression tests for end-to-end consistency comparisons."""

import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import torch

from generative_recommenders.dlrm_v3.train.tests import xpu_e2e_ddp_consistency as consistency


def _run_data(value: float = 0.0) -> Dict[str, Any]:
    return {
        "metrics": {("train", 1, "loss_ne/window"): value,
                    ("train", 1, "num_samples"): 10.0,
                    ("train", 1, "accuracy"): value},
        "ranks": [{"dense": {"weight": torch.tensor([value, 0.0], dtype=torch.float64)},
                   "emb": {"table[0]": [10.0 + value, 20.0 + value]}}],
    }


class XpuE2eComparisonTest(unittest.TestCase):
    def _main(
        self, data: Dict[str, Any], options: Optional[List[str]] = None
    ) -> Tuple[int, Optional[Dict[str, Any]], str]:
        with tempfile.TemporaryDirectory() as out, contextlib.ExitStack() as stack:
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch("sys.argv", ["consistency", "--out-dir", out] + (options or [])))
            run = stack.enter_context(patch.object(
                consistency, "_run", side_effect=lambda name, *args: (True, name)))
            stack.enter_context(patch.object(
                consistency, "_load", side_effect=lambda name, world_size: data[name]))
            rc = consistency.main()
            path = os.path.join(out, "summary.json")
            summary = None
            if os.path.exists(path):
                with open(path) as f:
                    summary = json.load(f)
            if rc != 2:
                self.assertEqual(run.call_count, len(data))
            return rc, summary, output.getvalue()

    def test_distribution_diagnostics_and_worst_locations(self) -> None:
        a, b = _run_data(), _run_data()
        a["metrics"][("train", 1, "loss_ne/window")] = 1.0
        for step, error in ((2, 2.0), (3, 3.0)):
            a["metrics"][("eval", step, "loss_ne/window")] = error
            b["metrics"][("eval", step, "loss_ne/window")] = 0.0
        a["ranks"][0]["dense"]["weight"] = torch.tensor([3.0, 4.0], dtype=torch.float64)
        a["ranks"][0]["dense"]["other"] = torch.tensor([0.0], dtype=torch.float64)
        b["ranks"][0]["dense"]["other"] = torch.tensor([0.0], dtype=torch.float64)
        a["ranks"][0]["emb"]["table[0]"] = [20.0, 20.0]
        diff = consistency._diff(a, b)
        self.assertEqual(diff["metrics_ne"], 3.0)
        self.assertEqual(diff["dense_weights"], 4.0)
        self.assertEqual(diff["embedding_checksums"], 0.5)
        diagnostics = diff["diagnostics"]
        self.assertEqual(diagnostics["metrics_ne"]["median"], 2.0)
        self.assertAlmostEqual(diagnostics["metrics_ne"]["p95"], 2.9)
        self.assertEqual(diagnostics["metrics_ne"]["worst_key"], ("eval", 3, "loss_ne/window"))
        self.assertAlmostEqual(diagnostics["dense_weights"]["rms"], math.sqrt(25.0 / 3))
        self.assertEqual(diagnostics["dense_weights"]["worst"]["parameter"], "weight")
        self.assertEqual(diagnostics["dense_weights"]["worst"]["flat_index"], 1)
        self.assertEqual(diagnostics["embedding_checksums"]["worst"]["checksum"], "abs_sum")
        self.assertEqual(len(diagnostics["embedding_checksums"]["entries"]), 2)
        json.dumps(diff, allow_nan=False)

    def test_all_pairs_and_default_five_references(self) -> None:
        data = {f"reference_{i + 1}": _run_data(value)
                for i, value in enumerate((0.0, 0.0, 0.0, -1.0, 1.0))}
        data["new"] = _run_data(3.0)
        rc, summary, output = self._main(data)
        self.assertEqual(rc, 0)
        self.assertEqual(len(summary["calibration"]["pairs"]), 10)
        self.assertEqual(summary["results"]["dense_weights"]["noise"], 2.0)
        self.assertEqual(summary["results"]["dense_weights"]["tolerance"], 4.0)
        self.assertEqual(summary["calibration"]["statistics"]["dense_weights"],
                         {"min": 0.0, "median": 1.0, "max": 2.0, "outlier_sensitive": False})
        self.assertIn("10 pairs", output)
        self.assertIn("diagnostics", summary)

    def test_outlier_sensitive_calibration_warns(self) -> None:
        data = {f"reference_{i}": _run_data() for i in range(1, 6)}
        data["reference_5"] = _run_data(1.0)
        data["new"] = _run_data()
        rc, summary, output = self._main(data)
        self.assertEqual(rc, 0)
        self.assertTrue(summary["calibration"]["statistics"]["dense_weights"]["outlier_sensitive"])
        self.assertTrue(summary["calibration"]["warnings"])
        self.assertIn("WARNING: dense_weights:", output)

    def test_maximum_still_fails_despite_small_rms(self) -> None:
        data = {"reference_1": _run_data(), "reference_2": _run_data(), "new": _run_data()}
        for run in data.values():
            run["ranks"][0]["dense"]["weight"] = torch.zeros(10000, dtype=torch.float64)
        data["new"]["ranks"][0]["dense"]["weight"][42] = 2e-5
        rc, summary, _ = self._main(data, ["--reference-runs", "2"])
        self.assertEqual(rc, 1)
        self.assertFalse(summary["results"]["dense_weights"]["ok"])
        self.assertAlmostEqual(summary["diagnostics"]["dense_weights"]["rms"], 2e-7)

    def test_single_reference_uses_absolute_floors(self) -> None:
        data = {"reference_1": _run_data(), "new": _run_data(1e-6)}
        rc, summary, output = self._main(data, ["--reference-runs", "1"])
        self.assertEqual(rc, 0)
        self.assertIsNone(summary["results"]["dense_weights"]["noise"])
        self.assertEqual(summary["results"]["dense_weights"]["tolerance"], 1e-5)
        self.assertEqual(summary["calibration"]["pairs"], [])
        self.assertIn("no noise calibration", output)

    def test_two_reference_tolerance_boundary_and_informational_accuracy(self) -> None:
        for value, expected_rc in ((0.25, 0), (math.nextafter(0.25, math.inf), 1)):
            with self.subTest(value=value):
                data = {name: _run_data() for name in ("reference_1", "reference_2", "new")}
                data["reference_2"]["ranks"][0]["dense"]["weight"][0] = 0.125
                data["new"]["ranks"][0]["dense"]["weight"][0] = value
                data["new"]["metrics"][("train", 1, "accuracy")] = 1.0
                rc, summary, _ = self._main(data, ["--reference-runs", "2"])
                self.assertEqual(rc, expected_rc)
                self.assertEqual(summary["results"]["dense_weights"]["noise"], 0.125)
                self.assertEqual(summary["results"]["dense_weights"]["tolerance"], 0.25)
                self.assertIsNone(summary["results"]["metrics_accuracy_gauc"]["ok"])

    def test_reference_mismatches_invalidate_calibration(self) -> None:
        for mismatch in ("count", "keys"):
            with self.subTest(mismatch=mismatch):
                data = {name: _run_data() for name in ("reference_1", "reference_2", "new")}
                if mismatch == "count":
                    data["reference_2"]["metrics"][("train", 1, "num_samples")] = 9
                else:
                    data["reference_2"]["metrics"][("eval", 2, "loss_ne/window")] = 0
                rc, summary, output = self._main(data, ["--reference-runs", "2"])
                self.assertEqual(rc, 2)
                self.assertIsNone(summary)
                self.assertIn("invalid calibration", output)

    def test_new_metric_mismatch_fails_exact_gate(self) -> None:
        data = {"reference_1": _run_data(), "new": _run_data()}
        data["new"]["metrics"][("eval", 2, "loss_ne/window")] = 0
        rc, summary, _ = self._main(data, ["--reference-runs", "1"])
        self.assertEqual(rc, 1)
        self.assertEqual(summary["results"]["unmatched_metric_records"]["tolerance"], 0)
        self.assertFalse(summary["results"]["unmatched_metric_records"]["ok"])

    def test_nonfinite_values_rejected(self) -> None:
        for field in ("metric", "dense", "emb"):
            with self.subTest(field=field):
                a, b = _run_data(), _run_data()
                if field == "metric":
                    a["metrics"][("train", 1, "loss_ne/window")] = float("nan")
                elif field == "dense":
                    a["ranks"][0]["dense"]["weight"][0] = float("inf")
                else:
                    a["ranks"][0]["emb"]["table[0]"][0] = float("nan")
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    consistency._diff(a, b)

    def test_invalid_calibration_options_rejected_before_running(self) -> None:
        for options in (["--reference-runs", "0"], ["--factor", "nan"],
                        ["--factor", "0.5"], ["--min-tol-dense", "-1"]):
            with self.subTest(options=options), patch("sys.argv", ["consistency"] + options):
                with patch.object(consistency, "_run") as run, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        consistency.main()
                    self.assertEqual(error.exception.code, 2)
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
