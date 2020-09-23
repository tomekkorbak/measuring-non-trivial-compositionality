import math

import pytest
import numpy as np
from scipy.spatial.distance import hamming
import editdistance

from metrics.topographic_similarity import TopographicSimilarity
from metrics.context_independence import ContextIndependence
from metrics.disentanglement import PositionalDisentanglement, BagOfWordsDisentanglement
from metrics.tre import TreeReconstructionError, AdditiveComposition
from protocols import get_trivially_compositional_protocol


@pytest.mark.parametrize(
    'metric,expected_score',
    [(TopographicSimilarity(input_metric=hamming, messages_metric=editdistance.eval), 1),
     (ContextIndependence(10), 0.25),
     (PositionalDisentanglement(2, 2), 1),
     (BagOfWordsDisentanglement(2, 2), 1)])
def test_metric_for_fully_compositional_protocol(
        metric,
        expected_score,
):
    protocol = get_trivially_compositional_protocol(5, 5)
    score = metric.measure(protocol)
    np.testing.assert_almost_equal(score, expected_score)


@pytest.mark.slow
def test_tre():
    protocol = get_trivially_compositional_protocol(5, 5)
    tre = TreeReconstructionError(10, 2, AdditiveComposition)
    score = tre.measure(protocol)
    np.testing.assert_almost_equal(score, 0, decimal=1)


def test_disentanglement_handles_constant_protocol():
    constant_protocol = {
        ('color=0', 'shape=0'): 'ba',
        ('color=0', 'shape=1'): 'ba',
        ('color=1', 'shape=0'): 'ba',
        ('color=1', 'shape=1'): 'ba',
    }
    positional_disentanglement = PositionalDisentanglement(2, 2)
    bow_disentanglement = BagOfWordsDisentanglement(2, 2)
    assert math.isnan(positional_disentanglement.measure(constant_protocol))
    assert math.isnan(bow_disentanglement.measure(constant_protocol))
