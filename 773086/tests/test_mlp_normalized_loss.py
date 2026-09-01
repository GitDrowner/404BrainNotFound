from __future__ import annotations

import unittest

import torch

from aigc_detector.losses import NormalizedMLPLoss


def config() -> dict:
    return {
        "mode": "mlp_normalized",
        "classification_anchor": 1.0,
        "auxiliary_budget": 1.0,
        "ema_decay": 0.9,
        "min_weight": 0.05,
        "hidden_dim": 8,
        "groups": {
            "a": {"terms": ["a"]},
            "b": {"terms": ["b"]},
            "c": {"terms": ["c"]},
        },
    }


def terms(values: tuple[float, float, float]) -> dict[str, torch.Tensor]:
    return {
        "classification": torch.tensor(0.7, requires_grad=True),
        "a": torch.tensor(values[0], requires_grad=True),
        "b": torch.tensor(values[1], requires_grad=True),
        "c": torch.tensor(values[2], requires_grad=True),
    }


class NormalizedMLPLossTest(unittest.TestCase):
    def test_uniform_warmup_and_simplex(self) -> None:
        module = NormalizedMLPLoss(config())
        loss, stats = module(terms((1.0, 2.0, 4.0)), learn=False)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(stats["mlp/weight_sum"], 1.0, places=6)
        for name in ("a", "b", "c"):
            self.assertAlmostEqual(stats[f"mlp/{name}/weight"], 1.0 / 3.0, places=6)

    def test_controller_learns_nonuniform_positive_weights(self) -> None:
        module = NormalizedMLPLoss(config())
        optimizer = torch.optim.SGD(module.parameters(), lr=0.2)
        module(terms((1.0, 1.0, 1.0)), learn=False)[0].backward()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = module(terms((0.5, 2.0, 5.0)), learn=True)
        loss.backward()
        self.assertIsNotNone(module.scorer[-1].weight.grad)
        self.assertGreater(float(module.scorer[-1].weight.grad.abs().sum()), 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _, stats = module(terms((0.5, 2.0, 5.0)), learn=True)
        weights = [stats[f"mlp/{name}/weight"] for name in ("a", "b", "c")]
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertTrue(all(weight >= 0.05 for weight in weights))
        self.assertGreater(max(weights) - min(weights), 1e-5)

    def test_detector_terms_receive_gradients_but_not_weight_shortcut(self) -> None:
        module = NormalizedMLPLoss(config())
        batch_terms = terms((0.5, 2.0, 5.0))
        loss, _ = module(batch_terms, learn=True)
        loss.backward()
        self.assertIsNotNone(batch_terms["classification"].grad)
        for name in ("a", "b", "c"):
            self.assertIsNotNone(batch_terms[name].grad)
            self.assertGreater(float(batch_terms[name].grad), 0.0)


if __name__ == "__main__":
    unittest.main()
