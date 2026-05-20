from abc import ABC, abstractmethod
from typing import Any
import pathlib, json

class AbstractMemoryWriter(ABC):

  def __init__(self, location: str):
    super().__init__()
    self.location = location

  @abstractmethod
  def write(self, value: Any):
    pass

  @abstractmethod
  def read(self) -> Any:
    pass

class SimpleWriter(AbstractMemoryWriter):

  def __init__(self, location: str):
    super().__init__(location)
    self.path = pathlib.Path(self.location)

  def write(self, value: Any):
    self.path.write_text(json.dumps(value))

  def read(self) -> Any:
    return json.loads(self.path.read_text())


class MemoryOffloader:

  def __setitem__(self, key: str, value: Any):
    SimpleWriter(key).write(value)

  def __getitem__(self, key: str) -> Any:
    return SimpleWriter(key).read()
