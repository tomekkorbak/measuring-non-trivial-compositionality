import itertools
import collections

from metrics.base import Metric, Protocol


class ConflictCount(Metric):

    def __init__(self, max_length: int):
        self.max_length = max_length

    def measure(self, protocol: Protocol) -> float:
        all_conflicts = []
        # for all mappings of symbol to features
        for p in itertools.permutations(range(self.max_length)):
            meanings = [collections.defaultdict(collections.Counter)
                        for i in range(self.max_length)]
            for features, msg in protocol.items():
                for i in range(self.max_length):
                    meanings[i][msg[i]].update([features[p[i]]])

            # count conflicts
            conflicts = 0
            for meaning in meanings:
                for symbol in meaning.values():
                    conflicts += sum(v for c, v in symbol.most_common()[1:])
            all_conflicts += [conflicts]
        return min(all_conflicts)
