from typing import Dict, List, Tuple
from random import shuffle

import torch
from egg.core import RnnReceiverDeterministic, SenderReceiverRnnReinforce

from metrics.base import Metric, Protocol
from metrics.utils import get_vocab_from_protocol, flatten_derivation, derivation_to_tensor
from protocols import Derivation


NN_CONFIG = {
    'receiver_hidden': 100,
    'receiver_cell': 'lstm',
    'receiver_emb': 50,
    'cell_layers': 1,
    'num_features': 50,
    'learning_rate': 1e-2,
    'weight_decay': 1e-6
}


def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels):
    nll_1 = torch.nn.functional.cross_entropy(receiver_output[0], labels[0], reduction="none")
    nll_2 = torch.nn.functional.cross_entropy(receiver_output[1], labels[1], reduction="none")
    acc_1 = (labels[0] == receiver_output[0].argmax(dim=1)).float().mean()
    acc_2 = (labels[1] == receiver_output[1].argmax(dim=1)).float().mean()
    return nll_1 + nll_2, {'acc': (acc_1 * acc_2).mean(dim=0),
                           'partial_acc': ((acc_1 + acc_2)/2).mean(dim=0)}


class FixedProtocolSender(torch.nn.Module):
    def __init__(self, protocol: Protocol, vocab: Dict[str, int]):
        super().__init__()
        self.protocol = protocol
        self.vocab = vocab
        self.training = False

    def forward(self, derivation: Derivation) -> torch.Tensor:
        message = [self.vocab[symbol] for symbol in self.protocol[derivation]]
        zeros = torch.zeros(len(message))
        return torch.LongTensor(message).unsqueeze(dim=0) + 1, zeros.unsqueeze(dim=0), zeros.unsqueeze(dim=0)


class Receiver(torch.nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Receiver, self).__init__()
        self.fc1 = torch.nn.Linear(n_hidden, n_features*4)
        self.fc2_1 = torch.nn.Linear(n_features*4, n_features)
        self.fc2_2 = torch.nn.Linear(n_features*4, n_features)

    def forward(self, input, _):
        hidden = torch.nn.functional.leaky_relu(self.fc1(input))
        return torch.stack([self.fc2_1(hidden), self.fc2_2(hidden)])


class Generalisation(Metric):

    def __init__(self, context_sensitive: bool):
        self.context_sensitive = context_sensitive

    def measure(self, protocol: Protocol) -> float:
        vocab = get_vocab_from_protocol(protocol)
        sender = FixedProtocolSender(protocol, vocab)
        receiver = RnnReceiverDeterministic(
            agent=Receiver(NN_CONFIG['receiver_hidden'], NN_CONFIG['num_features']),
            vocab_size=len(vocab) + 1,
            embed_dim=NN_CONFIG['receiver_emb'],
            hidden_size=NN_CONFIG['receiver_hidden'],
            cell=NN_CONFIG['receiver_cell'],
            num_layers=NN_CONFIG['cell_layers']
        )
        game = SenderReceiverRnnReinforce(sender, receiver, loss_nll, sender_entropy_coeff=0, receiver_entropy_coeff=0.05)
        if self.context_sensitive:
            concept_set = set(concept for derivation in protocol.keys() for concept in flatten_derivation(derivation)[1:])
        else:
            concept_set = set(concept for derivation in protocol.keys() for concept in flatten_derivation(derivation))
        concepts = {concept: idx for idx, concept in enumerate(concept_set)}
        if self.context_sensitive:
            derivations = [(derivation, derivation_to_tensor(derivation[1], concepts)) for derivation in protocol.keys()]
        else:
            derivations = [(derivation, derivation_to_tensor(derivation, concepts)) for derivation in protocol.keys()]
        shuffle(derivations)
        split_idx = int(len(derivations)*0.8)
        train_derivations, test_derivations = derivations[:split_idx], derivations[split_idx:]
        test_accuracy = self._train_and_test(game, train_derivations, test_derivations)
        return test_accuracy

    def _train_and_test(
            self,
            game: torch.nn.Module,
            train_derivations: List[Tuple[Derivation, Tuple[torch.Tensor, torch.Tensor]]],
            test_derivations: List[Tuple[Derivation, Tuple[torch.Tensor, torch.Tensor]]]
    ) -> float:
        optimiser = torch.optim.Adam(
            game.parameters(),
            lr=NN_CONFIG['learning_rate'],
            weight_decay=NN_CONFIG['weight_decay']
        )
        for t in range(100):
            train_accurarcies = []
            shuffle(train_derivations)
            game.train()
            optimiser.zero_grad()
            for derivation, label in train_derivations:
                loss, interaction = game(derivation, label)
                loss.backward()
                train_accurarcies.append(interaction.aux['acc'].item())
            optimiser.step()
            train_accuracy = sum(train_accurarcies)/len(train_accurarcies)
            if train_accuracy == 1:
                break  # early stopping
        else:
            print('Failed to converge')
        test_accuracies = []
        for derivation, label in test_derivations:
            loss, interaction = game(derivation, label)
            test_accuracies.append(interaction.aux['acc'].item())
        return sum(test_accuracies)/len(test_accuracies)
