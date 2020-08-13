from abc import ABC, abstractmethod

from protocols import Protocol


class Metric(ABC):

    @abstractmethod
    def measure(self, protocol: Protocol) -> float:
        raise NotImplemented()
