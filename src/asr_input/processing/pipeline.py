"""Text processing pipeline — chain multiple processors."""

from abc import ABC, abstractmethod


class TextProcessor(ABC):
    """Base class for a text processing step."""

    @abstractmethod
    def process(self, text: str) -> str: ...


class ProcessingPipeline:
    """Run text through a chain of processors in order."""

    def __init__(self) -> None:
        self._steps: list[TextProcessor] = []

    def add(self, step: TextProcessor) -> "ProcessingPipeline":
        self._steps.append(step)
        return self

    def run(self, text: str) -> str:
        for step in self._steps:
            text = step.process(text)
        return text
