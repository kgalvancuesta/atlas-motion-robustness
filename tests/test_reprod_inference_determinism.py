from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    import torch
    from monai.inferers import sliding_window_inference

    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False

from reproducibility.core import configure_strict_determinism  # noqa: E402
from reproducibility.inference import build_model, segmentation_metrics  # noqa: E402
from reproducibility.mrart import false_positive_metrics  # noqa: E402


@unittest.skipUnless(RUNTIME_AVAILABLE, "PyTorch and MONAI are required")
class LocalRepeatInferenceTest(unittest.TestCase):
    def test_same_checkpoint_input_and_configuration_repeat_exactly_on_cpu(self) -> None:
        configure_strict_determinism(314159)
        model = build_model("base_cnn").cpu().eval()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            torch.save({"model": model.state_dict()}, checkpoint)
            reloaded = build_model("base_cnn").cpu().eval()
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            reloaded.load_state_dict(state["model"])
            generator = torch.Generator().manual_seed(271828)
            value = torch.randn((1, 1, 32, 32, 32), generator=generator)

            def predict() -> np.ndarray:
                configure_strict_determinism(314159)
                with torch.inference_mode():
                    logits = sliding_window_inference(value, (32, 32, 32), 1, reloaded)
                    self.assertTrue(bool(torch.isfinite(logits).all().item()))
                    return torch.sigmoid(logits).ge(0.5).to(torch.uint8).numpy()[0, 0]

            first = predict()
            second = predict()
            self.assertTrue(np.array_equal(first, second))
            target = np.zeros(first.shape, dtype=bool)
            first_segmentation = segmentation_metrics(first, target, 1.0)
            second_segmentation = segmentation_metrics(second, target, 1.0)
            self.assertEqual(first_segmentation, second_segmentation)
            self.assertEqual(false_positive_metrics(first, 1.0), false_positive_metrics(second, 1.0))


if __name__ == "__main__":
    unittest.main()
