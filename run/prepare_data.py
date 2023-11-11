import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from astral.geocoder import database, lookup
from astral.location import Location
from astral.sun import sun
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "hour_sin",
    "hour_cos",
    # "day_sin",
    # "day_cos",
    # "month_sin",
    # "month_cos",
    # "minute_sin",
    # "minute_cos",
    # "week_sin",
    # "week_cos",
    # "sun_event",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829

city_name = 'New York'
city = lookup(city_name, database())
city = Location(city)

def get_sun_events(series_df: pl.DataFrame):
    """
    For each data add times of dawn, sunrise, noon, sunset and dusk
    """
    dates = series_df.with_columns(
        pl.col('timestamp').dt.date()
        ).select('timestamp').unique().to_series().to_list()
    
    sun_df = pl.DataFrame() 
    
    for date in dates:
        s = sun(city.observer, date=date, tzinfo=city.timezone)
        tmp = pl.from_dict(s).transpose(include_header=True, column_names=['timestamp'])
        tmp = tmp.with_columns(
            ((pl.col('column').cast(pl.Categorical).cast(pl.Int8)+1)/5).cast(pl.Float32),
            pl.col('timestamp').dt.round('5s')
            ).select(['column', 'timestamp'])
        tmp = tmp.rename({'column': 'sun_event'})
        sun_df = pl.concat([sun_df, tmp])

    series_df = series_df.join(sun_df, on='timestamp', how='left')
    series_df = series_df.with_columns(pl.col('sun_event').forward_fill())
    series_df = series_df.with_columns(pl.col('sun_event').backward_fill().alias('event_back'))
    series_df = series_df.with_columns(pl.when(pl.col('sun_event').is_null()).then(pl.col('event_back')-0.2).otherwise(pl.col('sun_event')).alias('sun_event'))
    series_df = series_df.drop('event_back')
    series_df = series_df.fill_null(0)
    return series_df
 

def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()
    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.day(), 366, "day"),
        *to_coord(pl.col("timestamp").dt.week(), 53, "week"),
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
        *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
    ).select("series_id", *FEATURE_NAMES)
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # delete the directory if it exists
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select([pl.col("series_id"), pl.col("anglez"), pl.col("enmo"), pl.col("timestamp")])
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # add features
            # this_series_df = get_sun_events(this_series_df)
            this_series_df = add_feature(this_series_df)

            # save each feature in .npy
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()
