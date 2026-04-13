from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.domain import (
    ConditionalDomainAdversarialLoss,
    MinimumClassConfusionLoss,
    MultipleKernelMaximumMeanDiscrepancy,
    entropy,
)


class FirstFeatureDiscriminator(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]


class DomainLossTests(unittest.TestCase):
    def test_linear_mkmmd_matches_tllib_pairwise_estimator(self) -> None:
        class DotProductKernel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ x.t()

        source = torch.tensor(
            [[1.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[0.0, 2.0], [1.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        )

        loss_module = MultipleKernelMaximumMeanDiscrepancy([DotProductKernel()], linear=True)
        loss_value = loss_module(source, target)

        expected = 0.0
        batch_size = source.shape[0]
        for index in range(batch_size):
            source_i = source[index]
            source_j = source[(index + 1) % batch_size]
            target_i = target[index]
            target_j = target[(index + 1) % batch_size]
            expected += torch.dot(source_i, source_j)
            expected += torch.dot(target_i, target_j)
            expected -= torch.dot(source_i, target_j)
            expected -= torch.dot(source_j, target_i)
        expected = expected / batch_size

        self.assertTrue(torch.isclose(loss_value, expected))

    def test_mcc_matches_tllib_definition(self) -> None:
        logits = torch.tensor(
            [[2.0, 0.0, -1.0], [0.5, 1.0, 0.0], [-0.2, 0.3, 1.2]],
            dtype=torch.float32,
        )
        temperature = 2.0
        loss_module = MinimumClassConfusionLoss(temperature=temperature)
        loss_value = loss_module(logits)

        probabilities = F.softmax(logits / temperature, dim=1)
        entropy_weight = 1.0 + torch.exp(-entropy(probabilities).detach())
        entropy_weight = (logits.shape[0] * entropy_weight / entropy_weight.sum()).unsqueeze(dim=1)
        class_confusion = (probabilities * entropy_weight).transpose(0, 1).mm(probabilities)
        class_confusion = class_confusion / class_confusion.sum(dim=1, keepdim=True)
        expected = (class_confusion.sum() - torch.trace(class_confusion)) / logits.shape[1]

        self.assertTrue(torch.isclose(loss_value, expected))

    def test_cdan_entropy_conditioning_uses_one_combined_weighted_bce(self) -> None:
        loss_module = ConditionalDomainAdversarialLoss(
            domain_discriminator=FirstFeatureDiscriminator(),
            feature_dim=2,
            num_classes=2,
            entropy_conditioning=True,
            randomized=False,
            grl=nn.Identity(),
        )

        logits_source = torch.tensor([[4.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
        features_source = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        logits_target = torch.tensor([[0.2, 0.0], [0.1, 0.0]], dtype=torch.float32)
        features_target = torch.tensor([[3.0, 0.0], [4.0, 0.0]], dtype=torch.float32)

        loss_value, acc_value = loss_module(
            logits_source,
            features_source,
            logits_target,
            features_target,
        )

        probabilities_source = F.softmax(logits_source, dim=1).detach()
        probabilities_target = F.softmax(logits_target, dim=1).detach()
        conditioned_source = loss_module.map(features_source, probabilities_source)
        conditioned_target = loss_module.map(features_target, probabilities_target)
        conditioned = torch.cat([conditioned_source, conditioned_target], dim=0)
        logits_domain = loss_module.domain_discriminator(conditioned)
        labels_domain = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

        probabilities = torch.cat([probabilities_source, probabilities_target], dim=0)
        combined_weights = 1.0 + torch.exp(-entropy(probabilities.detach()))
        combined_weights = combined_weights / combined_weights.sum() * combined_weights.shape[0]
        expected_loss = F.binary_cross_entropy_with_logits(
            logits_domain,
            labels_domain,
            weight=combined_weights,
        )
        expected_acc = (((torch.sigmoid(logits_domain) >= 0.5).float() == labels_domain).float().mean()).item()

        source_weights = 1.0 + torch.exp(-entropy(probabilities_source.detach()))
        source_weights = source_weights / source_weights.sum() * source_weights.shape[0]
        target_weights = 1.0 + torch.exp(-entropy(probabilities_target.detach()))
        target_weights = target_weights / target_weights.sum() * target_weights.shape[0]
        old_split_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(logits_domain[:2], labels_domain[:2], weight=source_weights)
            + F.binary_cross_entropy_with_logits(logits_domain[2:], labels_domain[2:], weight=target_weights)
        )

        self.assertTrue(torch.isclose(loss_value, expected_loss))
        self.assertAlmostEqual(acc_value, expected_acc)
        self.assertFalse(torch.isclose(expected_loss, old_split_loss))


if __name__ == "__main__":
    unittest.main()
