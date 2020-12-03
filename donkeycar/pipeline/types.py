import os
from typing import Any, List, Optional, cast

from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import load_image_arr, normalize_image
from typing_extensions import TypedDict

TubRecordDict = TypedDict(
    'TubRecordDict',
    {
        'cam/image_array': str,
        'user/angle': float,
        'user/throttle': float,
        'user/mode': str,
        'imu/acl_x': Optional[float],
        'imu/acl_y': Optional[float],
        'imu/acl_z': Optional[float],
        'imu/gyr_x': Optional[float],
        'imu/gyr_y': Optional[float],
        'imu/gyr_z': Optional[float],
    }
)


class TubRecord(object):
    def __init__(self, config: Any, base_path: str, underlying: TubRecordDict) -> None:
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._image: Optional[Any] = None

    def image(self, cached=True, normalize=False) -> Any:
        if not self._image:
            image_path = self.underlying['cam/image_array']
            full_path = os.path.join(self.base_path, image_path)
            _image = load_image_arr(full_path, cfg=self.config)
            if normalize:
                _image = normalize_image(_image)
            if cached:
                self._image = _image
            return _image
        else:
            return self._image


class TubDataset(object):
    def __init__(self, paths: List[str], config: Any) -> None:
        self.paths = paths
        self.config = config
        self.tubs = [Tub(path) for path in self.paths]
        self.records: List[TubRecord] = list()

    def load_records(self) -> List[TubRecord]:
        self.records.clear()
        for tub in self.tubs:
            for record in tub:
                underlying = cast(TubRecordDict, record)
                tub_record = TubRecord(self.config, tub.base_path, underlying)
                self.records.append(tub_record)

        return self.records
