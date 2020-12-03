from typing import Iterable, List, Callable, Tuple, TypeVar

from donkeycar.pipeline.types import TubRecord

X = TypeVar('X', covariant=True)
Y = TypeVar('Y', covariant=True)
XOut = TypeVar('XOut', covariant=True)
YOut = TypeVar('YOut', covariant=True)


class TubSequence(object):
    def __init__(self, records: List[TubRecord]) -> None:
        self.records = records

    def __iter__(self) -> Iterable[TubRecord]:
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def build_pipeline(self,
                       x_transform: Callable[[TubRecord], X],
                       y_transform: Callable[[TubRecord], Y]) -> Iterable[Tuple[X, Y]]:

        iterator = self.__iter__()
        for record in iterator:
            x = x_transform(record)
            y = y_transform(record)
            yield x, y

    @classmethod
    def map_pipeline(
            cls,
            x_transform: Callable[[X], XOut],
            y_transform: Callable[[Y], YOut],
            pipeline: Iterable[Tuple[X, Y]]) -> Iterable[Tuple[XOut, YOut]]:

        for record in iter(pipeline):
            x, y = record
            yield x_transform(x), y_transform(y)

    @classmethod
    def map_pipeline_factory(
            cls,
            x_transform: Callable[[X], XOut],
            y_transform: Callable[[Y], YOut],
            factory: Callable[[], Iterable[Tuple[X, Y]]]) -> Iterable[Tuple[XOut, YOut]]:

        pipeline = factory()
        return cls.map_pipeline(pipeline=pipeline, x_transform=x_transform, y_transform=y_transform)
