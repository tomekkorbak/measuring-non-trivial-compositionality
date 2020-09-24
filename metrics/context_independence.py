from collections import defaultdict
from typing import Dict

import numpy as np

from metrics.base import Metric, Protocol
from metrics.utils import flatten_derivation


class ContextIndependence(Metric):

    def __init__(self, num_concepts: int):
        self.num_concepts = num_concepts

    def measure(self, protocol: Protocol) -> float:
        character_set = set(c for message in protocol.values() for c in message)
        vocab = {char: idx for idx, char in enumerate(character_set)}
        concept_set = set(concept for concepts in protocol.keys()
                          for concept in flatten_derivation(concepts))
        concepts = {concept: idx for idx, concept in enumerate(concept_set)}

        concept_symbol_matrix = self._compute_concept_symbol_matrix(protocol, vocab, concepts)
        v_cs = concept_symbol_matrix.argmax(axis=1)
        context_independence_scores = np.zeros(len(concept_set))
        for concept in range(len(concept_set)):
            v_c = v_cs[concept]
            p_vc_c = concept_symbol_matrix[concept, v_c] / concept_symbol_matrix[concept, :].sum(axis=0)
            p_c_vc = concept_symbol_matrix[concept, v_c] / concept_symbol_matrix[:, v_c].sum(axis=0)
            context_independence_scores[concept] = p_vc_c * p_c_vc
        return context_independence_scores.mean(axis=0)

    def _compute_concept_symbol_matrix(
            self,
            protocol: Protocol,
            vocab: Dict[str, int],
            concepts: Dict[str, int],
            epsilon: float = 10e-8
    ) -> np.ndarray:
        concept_to_message = defaultdict(list)
        for derivation, message in protocol.items():
            for concept in flatten_derivation(derivation):
                concept_to_message[concept] += list(message)
        concept_symbol_matrix = np.ndarray((self.num_concepts, len(vocab)))
        concept_symbol_matrix.fill(epsilon)
        for concept, symbols in concept_to_message.items():
            for symbol in symbols:
                concept_symbol_matrix[concepts[concept], vocab[symbol]] += 1
        return concept_symbol_matrix
