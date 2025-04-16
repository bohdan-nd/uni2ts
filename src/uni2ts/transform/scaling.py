from dataclasses import dataclass
from typing import Any
from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import MapFuncMixin


@dataclass
class MinMaxScaler(MapFuncMixin, Transformation):
    fields: tuple[str, ...] = tuple()
    optional_fields: tuple[str, ...] = tuple()
    feature_range: tuple[float, float] = (0.0, 1.0)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(self._scale_feature, data_entry, self.fields, self.optional_fields)

        return data_entry

    def _scale_feature(self, data_entry: dict[str, Any], field: str) -> list[UnivarTimeSeries]:
        field_arr: list[UnivarTimeSeries] = data_entry[field]
        scaled_arr: list[UnivarTimeSeries] = []

        for arr in field_arr:
            min_value, max_value = arr[0], arr[-1]
            arr_minmax = (arr - min_value) / (max_value - min_value)
            arr_minmax_scaled = arr_minmax * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

            scaled_arr.append(arr_minmax_scaled)

        return scaled_arr
