from typing import Dict

import torch

from protocols import Derivation, Protocol


def flatten_derivation(derivation: Derivation):
    if isinstance(derivation, str):
        return [derivation]
    if isinstance(derivation, tuple):
        return flatten_derivation(derivation[0]) + flatten_derivation(derivation[1])
    else:
        raise TypeError('Invalid derivation')


def derivation_to_tensor(derivation: Derivation, concepts: Dict[str, int]):
    if isinstance(derivation, str):
        return torch.LongTensor([concepts[derivation]])
    if isinstance(derivation, tuple):
        return (derivation_to_tensor(derivation[0], concepts),
                derivation_to_tensor(derivation[1], concepts))
    else:
        raise TypeError('Invalid derivation')


def get_vocab_from_protocol(protocol: Protocol) -> Dict[str, int]:
    character_set = set(c for message in protocol.values() for c in message)
    return {char: idx for idx, char in enumerate(character_set)}
