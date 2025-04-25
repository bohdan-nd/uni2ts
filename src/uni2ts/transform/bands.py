from dataclasses import dataclass
from typing import Any
from uni2ts.common.typing import UnivarTimeSeries
import numpy as np
from scipy import stats
from functools import partial
from ._base import Transformation
from ._mixin import MapFuncMixin


@dataclass
class MagAndMagErrorNormalizer(Transformation):
    mag_field: str = "mag"
    magerror_field: str = "magerr"
    band_field: str = "bands"

    def __call__(self, data_entry):
        mag_arr: list[UnivarTimeSeries] = data_entry[self.mag_field]
        magerror_arr: list[UnivarTimeSeries] = data_entry[self.magerror_field]
        bands_arr: list[UnivarTimeSeries] = data_entry[self.band_field]

        for index in range(len(mag_arr)):
            mag = mag_arr[index].copy()
            magerror = magerror_arr[index].copy()
            bands = bands_arr[index]

            for unique_band in np.unique(bands):
                band_mask = bands == unique_band
                band_mag_mean = mag[band_mask].mean()
                band_mag_mad = stats.median_abs_deviation(mag[band_mask])

                mag[band_mask] = (mag[band_mask] - band_mag_mean) / band_mag_mad
                magerror[band_mask] = magerror[band_mask] / band_mag_mad

            mag_arr[index] = mag
            magerror_arr[index] = magerror

        data_entry[self.mag_field] = mag_arr
        data_entry[self.magerror_field] = magerror_arr

        return data_entry


@dataclass
class FilterBands(MapFuncMixin, Transformation):
    fields: tuple[str, ...] = tuple()
    optional_fields: tuple[str, ...] = tuple()
    bands_field: str = "bands"
    index_key: str = "index"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        index = data_entry[self.index_key]
        random_generator = np.random.default_rng(index)

        bands: UnivarTimeSeries = data_entry["bands"][0]
        unique_bands = bands.unique()
        band_number_to_select = random_generator.integers(1, len(unique_bands) + 1)
        bands_to_select = random_generator.choice(unique_bands, band_number_to_select)

        filter_indices = np.isin(bands, bands_to_select)

        self.map_func(
            partial(self.filter_by_indices, indices=filter_indices), data_entry, self.fields, self.optional_fields
        )

        return data_entry

    def filter_by_indices(self, data_entry: dict[str, Any], field: str, indices: np.ndarray) -> list[UnivarTimeSeries]:
        arr: list[UnivarTimeSeries] = data_entry[field]
        filtered_arr = [timeseries[indices] for timeseries in arr]

        return filtered_arr


@dataclass
class BandRemapper(Transformation):
    bands_field: str = "bands"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        band_arr: list[UnivarTimeSeries] = data_entry[self.bands_field]
        remapped_band_arr: list[UnivarTimeSeries] = []

        for index in range(len(band_arr)):
            bands = band_arr[index].copy()

            _, indices = np.unique(bands, return_index=True)
            ordered_unique_bands = bands[np.sort(indices)]
            already_remapped = np.zeros_like(bands)

            for new_value, old_value in enumerate(ordered_unique_bands):
                to_change_mask = (already_remapped == 0) & (bands == old_value)
                bands[to_change_mask] = new_value
                already_remapped[to_change_mask] = 1

            remapped_band_arr.append(bands)

        data_entry[self.bands_field] = remapped_band_arr

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
