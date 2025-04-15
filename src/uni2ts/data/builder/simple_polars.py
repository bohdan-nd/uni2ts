import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List, Union, Tuple

import polars as pl

import datasets
from datasets import Features, Sequence, Value
from torch.utils.data import Dataset

from uni2ts.common.env import env
from uni2ts.data.dataset import EvalDataset, SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Transformation
import os
from functools import partial
from tqdm import tqdm

from ._base import DatasetBuilder
from concurrent.futures import ProcessPoolExecutor

ID_COLUMN = "item_id"
FREQ_COLUMN = "freq"
START_COLUMN = "start"
MAG_COLUMN = "mag"
MAGERROR_COLUMN = "magerr"
BAND_COLUMN = "bands"

RAW_ID_COLUMN = "objectid"
TIMESTEMP_COLUMN = "mjd"
BANDS = ["r", "g", "i"]
BAND_TO_INTEGER = {"r": 0, "g": 1, "i": 2}


def mjd_to_unix_timestemp(timestemp):
    return (timestemp - 40587) * 86400 * 1e3


def _concat_non_null_columns_expr(columns: list[str]) -> pl.Expr:
    return pl.concat_list(
        [pl.when(pl.col(column).is_not_null()).then(pl.col(column)).otherwise(pl.lit([])) for column in columns]
    )


def _sort_columns_by_columns_expr(columns_to_sort: list[str], ref_column: str) -> pl.Expr:
    return pl.col(columns_to_sort).list.gather(pl.col(ref_column).list.eval(pl.element().arg_sort()))


def _transform_polars(lf: pl.LazyFrame, offset: float, end: float, freq: str) -> pl.LazyFrame:
    rows = lf.select(pl.len()).collect().item()
    offset_int = int(rows * offset)
    length = int(rows * end) - offset_int

    lf = lf.slice(offset_int, length)

    select_id_expr = pl.col(RAW_ID_COLUMN)

    mag_columns = [f"{MAG_COLUMN}_{band}" for band in BANDS]
    concat_mag_expr = _concat_non_null_columns_expr(mag_columns)

    magerror_columns = [f"{MAGERROR_COLUMN}_{band}" for band in BANDS]
    concat_magerror_expr = _concat_non_null_columns_expr(magerror_columns)

    timestemp_columns = [f"{TIMESTEMP_COLUMN}_{band}" for band in BANDS]
    concat_timestemp_expr = _concat_non_null_columns_expr(timestemp_columns)

    timestemp_columns_to_band = [(f"{TIMESTEMP_COLUMN}_{band}", band) for band in BANDS]

    generate_bands_expr = [
        pl.when(pl.col(col_name).is_not_null()).then(
            pl.lit(BAND_TO_INTEGER[band]).repeat_by(pl.col(col_name).list.len())
        )
        for col_name, band in timestemp_columns_to_band
    ]
    concat_bands = pl.concat_list(generate_bands_expr)

    lf = lf.select(
        select_id_expr.alias(ID_COLUMN),
        concat_mag_expr.alias(MAG_COLUMN),
        concat_magerror_expr.alias(MAGERROR_COLUMN),
        concat_timestemp_expr.alias(TIMESTEMP_COLUMN),
        concat_bands.alias(BAND_COLUMN),
    )

    sort_column_by_timestemp = _sort_columns_by_columns_expr(
        [MAG_COLUMN, MAGERROR_COLUMN, TIMESTEMP_COLUMN, BAND_COLUMN], TIMESTEMP_COLUMN
    )

    first_timestemp_exprs = mjd_to_unix_timestemp(pl.col(TIMESTEMP_COLUMN).list.first()).cast(pl.Datetime("ms"))

    generate_freq_expr = pl.lit(freq)

    lf = lf.with_columns(
        sort_column_by_timestemp, first_timestemp_exprs.alias(START_COLUMN), generate_freq_expr.alias(FREQ_COLUMN)
    )

    features = Features(
        {
            ID_COLUMN: Value("string"),
            START_COLUMN: Value("timestamp[ms]"),
            FREQ_COLUMN: Value("string"),
            TIMESTEMP_COLUMN: Sequence(Value("float32")),
            MAG_COLUMN: Sequence(Value("float32")),
            MAGERROR_COLUMN: Sequence(Value("float32")),
            BAND_COLUMN: Sequence(Value("uint8")),
        }
    )

    return lf, features


def _load_and_transform_polars(path, offset: float, end: float, freq: str) -> datasets.Dataset:
    lf = pl.scan_parquet(path)
    lf, features = _transform_polars(lf, offset, end, freq)
    df = lf.collect()

    dataset = datasets.Dataset.from_polars(df, features=features)

    return dataset


def _create_hf_dataset_from_polars(
    files: List[str], offset: float, end: float, freq: str = "H", max_workers: Optional[int] = None
) -> datasets.Dataset:
    polars_transform_fun = partial(_load_and_transform_polars, offset=offset, end=end, freq=freq)

    if not max_workers:
        max_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        dataset = list(tqdm(executor.map(polars_transform_fun, files), total=len(files)))

    dataset = datasets.concatenate_datasets(dataset)

    return dataset


def _select_parquet_files(folder_path: Union[str, Path]):
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    return [
        path_object
        for path_object in folder_path.iterdir()
        if path_object.is_file() and path_object.suffix == ".parquet"
    ]


# The option how to transform the column obtained from Polars into a 2D array
# HF Dataset Indexer uses a custom method to extract data, thus it's no longer needed

# def _transform_to_uni2ts_format(batch):
#     timeseries = batch[HF_TIMESERIES_COLUMN]
#     non_null_counter = batch.pop(HF_NON_NULL_COUNTER)

#     transformed_timeseries = []

#     for data, non_null_count in zip(timeseries, non_null_counter):
#         transformed_data = np.array(data).reshape((non_null_count, -1))
#         transformed_timeseries.append(transformed_data)

#     batch[HF_TIMESERIES_COLUMN] = transformed_timeseries

#     return batch


@dataclass
class SimplePolarsDatasetBuilder(DatasetBuilder):
    dataset: str
    weight: float = 1.0
    sample_time_series: Optional[SampleTimeSeriesType] = SampleTimeSeriesType.NONE
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(self, folder_path: Path, offset: float, end: float, freq: str):
        polars_files = _select_parquet_files(folder_path)
        hf_dataset = _create_hf_dataset_from_polars(polars_files, offset, end, freq)

        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.info.description = {"band_to_integer": BAND_TO_INTEGER}
        hf_dataset.save_to_disk(str(self.storage_path / self.dataset))

    def load_dataset(self, transform_map: dict[str, Callable[..., Transformation]]) -> Dataset:
        hf_dataset = datasets.load_from_disk(str(self.storage_path / self.dataset))

        # HuggingFaceDatasetIndexer uses custom method to extract values from the dataset, so the 'set_transform' is no longer relevant
        # hf_dataset.set_transform(_transform_to_uni2ts_format)

        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(hf_dataset),
            transform=transform_map[self.dataset](),
            dataset_weight=self.weight,
            sample_time_series=self.sample_time_series,
        )


@dataclass
class SimpleEvalDatasetBuilder(DatasetBuilder):
    dataset: str
    offset: Optional[int]
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(self, folder_path: Path, offset: float, end: float, freq: str):
        polars_files = _select_parquet_files(folder_path)
        hf_dataset = _create_hf_dataset_from_polars(polars_files, offset, end, freq)

        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(str(self.storage_path / self.dataset))

    def load_dataset(self, transform_map: dict[str, Callable[..., Transformation]]) -> Dataset:
        hf_dataset = datasets.load_from_disk(str(self.storage_path / self.dataset))

        # HuggingFaceDatasetIndexer uses custom method to extract values from the dataset, so the 'set_transform' is no longer relevant
        # hf_dataset.set_transform(_transform_to_uni2ts_format)

        return EvalDataset(
            self.windows,
            HuggingFaceDatasetIndexer(hf_dataset),
            transform=transform_map[self.dataset](
                offset=self.offset,
                distance=self.distance,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
            ),
        )


def build_datasets(args):
    dataset_builder = SimplePolarsDatasetBuilder(dataset=args.dataset_name)
    dataset_builder.build_dataset(args.folder_path, offset=0.0, end=args.split_ratio, freq=args.freq)

    eval_dataset_builder = SimpleEvalDatasetBuilder(
        dataset=f"{args.dataset_name}_eval",
        offset=None,
        windows=None,
        distance=None,
        prediction_length=None,
        context_length=None,
        patch_size=None,
    )
    eval_dataset_builder.build_dataset(
        folder_path=Path(args.folder_path), offset=args.split_ratio, end=1.0, freq=args.freq
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("folder_path", type=str)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    # Define the `freq` argument with a default value. Use this value as 'freq' if 'freq' is None.
    parser.add_argument(
        "--freq",
        default="H",  # Set the default value
        help="The user specified frequency",
    )

    args = parser.parse_args()
    build_datasets(args)
