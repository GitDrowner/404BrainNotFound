from __future__ import annotations

import unittest

import torch

from aigc_detector.losses import UncertaintyWeightedLoss


def config() -> dict:
    return {
        "mode": "uncertainty",
        "classification_anchor": 1.0,
        "min_log_variance": -3.0,
        "max_log_variance": 3.0,
        "groups": {
            "robustness": {
                "terms": ["clean_classification", "consistency"],
                "initial_weight": 0.4,
            },
            "local": {"terms": ["tile_auxiliary"], "initial_weight": 0.1},
        },
    }


class UncertaintyWeightedLossTest(unittest.TestCase):
    def test_initial_weights_and_exact_objective(self) -> None:
        module = UncertaintyWeightedLoss(config())
        terms = {
            "classification": torch.tensor(2.0),
            "clean_classification": torch.tensor(1.0),
            "consistency": torch.tensor(0.5),
            "tile_auxiliary": torch.tensor(0.25),
        }
        loss, stats = module(terms, learn=False)
        expected = (
            2.0
            + 0.4 * 1.5
            - torch.log(torch.tensor(0.4)).item()
            + 0.1 * 0.25
            - torch.log(torch.tensor(0.1)).item()
        )
        self.assertAlmostEqual(float(loss), expected, places=5)
        self.assertAlmostEqual(stats["uncertainty/robustness/weight"], 0.4, places=5)
        self.assertAlmostEqual(stats["uncertainty/local/weight"], 0.1, places=5)

    def test_learning_can_be_disabled_during_warmup(self) -> None:
        module = UncertaintyWeightedLoss(config())
        terms = {
            "classification": torch.tensor(1.0, requires_grad=True),
            "clean_classification": torch.tensor(1.0, requires_grad=True),
            "consistency": torch.tensor(1.0, requires_grad=True),
            "tile_auxiliary": torch.tensor(1.0, requires_grad=True),
        }
        loss, _ = module(terms, learn=False)
        loss.backward()
        self.assertIsNone(module.log_variances.grad)

    def test_weights_receive_gradients_after_warmup(self) -> None:
        module = UncertaintyWeightedLoss(config())
        terms = {
            "classification": torch.tensor(1.0, requires_grad=True),
            "clean_classification": torch.tensor(1.0, requires_grad=True),
            "consistency": torch.tensor(1.0, requires_grad=True),
            "tile_auxiliary": torch.tensor(1.0, requires_grad=True),
        }
        loss, _ = module(terms, learn=True)
        loss.backward()
        self.assertIsNotNone(module.log_variances.grad)
        self.assertEqual(tuple(module.log_variances.grad.shape), (2,))


if __name__ == "__main__":
    unittest.main()
