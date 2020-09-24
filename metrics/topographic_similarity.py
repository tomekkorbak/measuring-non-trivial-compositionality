from typing import Callable, List

from scipy.stats import spearmanr

from metrics.base import Metric, Protocol
from metrics.utils import flatten_derivation


class TopographicSimilarity(Metric):

    def __init__(self, input_metric: Callable, messages_metric: Callable):
        self.input_metric = input_metric
        self.messages_metric = messages_metric

    def measure(self, protocol: Protocol) -> float:
        distance_messages = self._compute_distances(
            sequence=list(protocol.values()),
            metric=self.messages_metric)
        distance_inputs = self._compute_distances(
            sequence=[flatten_derivation(derivation) for derivation in protocol.keys()],
            metric=self.input_metric)
        return spearmanr(distance_messages, distance_inputs).correlation

    def _compute_distances(self, sequence: List[str], metric: Callable) -> List[float]:
        distances = []
        for i, element_1 in enumerate(sequence):
            for j, element_2 in enumerate(sequence[i+1:]):
                distances.append(metric(element_1, element_2))
        return distances
