from dataclasses import dataclass
from typing import Any
from uni2ts.common.typing import UnivarTimeSeries
import numpy as np
from scipy import stats

from ._base import Transformation
from ._mixin import MapFuncMixin


@dataclass
class MagnitudeNormalizer(Transformation):
    mag_field: str = "mag"
    magerror_field: str = "magerr"
    band_field: str = "bands"

    def __call__(self, data_entry):
        mag_arr: list[UnivarTimeSeries] = data_entry[self.mag_field]
        magerror_arr: list[UnivarTimeSeries] = data_entry[self.magerror_field]
        bands_arr: list[UnivarTimeSeries] = data_entry[self.band_field]

        for index in range(len(mag_arr)):
            mag = mag_arr[index]
            magerror = magerror_arr[index]
            bands = bands_arr[index]

            for unuqie_band in np.unique(bands):
                band_mask = bands == unuqie_band
                band_mag_mean = mag[band_mask].mean()
                band_mag_mad = stats.median_abs_deviation(mag[band_mask])

                mag[band_mask] = (mag[band_mask] - band_mag_mean) / band_mag_mad
                magerror[band_mask] = magerror[band_mask] / band_mag_mad

        data_entry[self.mag_field] = mag_arr
        data_entry[self.magerror_field] = magerror_arr

        return data_entry


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
