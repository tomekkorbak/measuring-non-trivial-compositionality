from typing import Tuple, Iterable, Dict, Type

import torch

from metrics.base import Metric
from metrics.utils import flatten_derivation, derivation_to_tensor, get_vocab_from_protocol
from protocols import Protocol


class CompositionFunction(torch.nn.Module):

    def __init__(self, representation_size: int):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplemented


class AdditiveComposition(CompositionFunction):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class LinearComposition(CompositionFunction):
    def __init__(self, representation_size: int):
        super().__init__(representation_size)
        self.linear = torch.nn.Linear(representation_size * 2, representation_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((x, y), dim=1))


class MLPComposition(CompositionFunction):
    def __init__(self, representation_size: int):
        super().__init__(representation_size)
        self.linear_1 = torch.nn.Linear(representation_size * 2, 50)
        self.linear_2 = torch.nn.Linear(50, representation_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.tanh(self.linear_1(torch.cat((x, y), dim=1))))


class LinearMultiplicationComposition(CompositionFunction):
    def __init__(self, representation_size: int):
        super().__init__(representation_size)
        self.linear_1 = torch.nn.Linear(representation_size, representation_size)
        self.linear_2 = torch.nn.Linear(representation_size, representation_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.linear_1(x) * self.linear_2(y)


class MultiplicativeComposition(CompositionFunction):
    def __init__(self, representation_size: int):
        super().__init__(representation_size)
        self.bilinear = torch.nn.Bilinear(
            in1_features=representation_size,
            in2_features=representation_size,
            out_features=representation_size
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bilinear(x, y)


class MultipleCrossEntropyLoss(torch.nn.Module):

    def __init__(self, representation_size: int, message_length: int):
        super().__init__()
        self.representation_size = representation_size
        self.message_length = message_length

    def forward(self, reconstruction, message):
        assert self.representation_size % self.message_length == 0
        width_of_single_symbol = self.representation_size//self.message_length
        loss = 0
        for i in range(self.message_length):
            start = width_of_single_symbol * i
            end = width_of_single_symbol * (i+1)
            loss += torch.nn.functional.cross_entropy(
                reconstruction[:, start:end],
                message[start:end].argmax(dim=0).reshape(1)
            )
        return loss


class Objective(torch.nn.Module):
    def __init__(
            self,
            num_concepts: int,
            vocab_size: int,
            message_length: int,
            composition_fn: torch.nn.Module,
            loss_fn: torch.nn.Module,
            zero_init=False
    ):
        super().__init__()
        self.composition_fn = composition_fn
        self.loss_fn = loss_fn
        self.emb = torch.nn.Embedding(num_concepts, message_length * vocab_size)
        if zero_init:
            self.emb.weight.data.zero_()

    def compose(self, derivations):
        if isinstance(derivations, tuple):
            args = (self.compose(node) for node in derivations)
            return self.composition_fn(*args)
        else:
            return self.emb(derivations)

    def forward(self, messages, derivations):
        return self.loss_fn(self.compose(derivations), messages)


class TreeReconstructionError(Metric):

    def __init__(
            self,
            num_concepts: int,
            message_length: int,
            composition_fn: Type[CompositionFunction],
            weight_decay=1e-5,
    ):
        self.num_concepts = num_concepts
        self.message_length = message_length
        self.composition_fn = composition_fn
        self.weight_decay = weight_decay

    def measure(self, protocol: Protocol) -> float:
        tensorised_protocol = self._protocol_to_tensor(protocol)
        vocab = get_vocab_from_protocol(protocol)
        objective = Objective(
            num_concepts=self.num_concepts,
            vocab_size=len(vocab),
            message_length=self.message_length,
            composition_fn=self.composition_fn(representation_size=self.message_length * len(vocab)),
            loss_fn=MultipleCrossEntropyLoss(representation_size=self.message_length * len(vocab), message_length=self.message_length)
        )
        reconstruction_error = self._train_model(
            messages=tensorised_protocol.values(),
            derivations=tensorised_protocol.keys(),
            objective=objective,
            optimizer=torch.optim.Adam(objective.parameters(), lr=1e-1, weight_decay=self.weight_decay),
            n_epochs=1_000
        )
        return reconstruction_error

    def _train_model(
            self,
            messages: Iterable[torch.Tensor],
            derivations: Iterable[torch.Tensor],
            objective: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            n_epochs: int,
            quiet: bool = True
    ) -> float:
        for t in range(n_epochs):
            optimizer.zero_grad()
            errors = [objective(message, derivation) for message, derivation in zip(messages, derivations)]
            loss = sum(errors)
            loss.backward()
            if not quiet and t % 1000 == 0:
                print(f'Training loss at epoch {t} is {loss.item():.4f}')
            optimizer.step()
        return loss.item()

    def _protocol_to_tensor(self, protocol: Protocol) -> Dict[Tuple[torch.LongTensor, torch.LongTensor], torch.LongTensor]:
        vocab = get_vocab_from_protocol(protocol)
        concept_set = set(concept for derivation in protocol.keys() for concept in flatten_derivation(derivation))
        concepts = {concept: idx for idx, concept in enumerate(concept_set)}
        tensorized_protocol = {}
        for derivation, message in protocol.items():
            derivation = derivation_to_tensor(derivation, concepts)
            message = torch.LongTensor([vocab[char] for char in message])
            tensorized_protocol[derivation] = torch.nn.functional.one_hot(
                message, num_classes=len(vocab)).reshape(-1)
        return tensorized_protocol
