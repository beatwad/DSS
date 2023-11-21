import numpy as np
import polars as pl
from scipy.signal import find_peaks


def post_process_for_seg(
    keys: list[str], preds: np.ndarray, score_th: float = 0.01, distance: int = 5000
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        # this_series_sleeps = preds[series_idx][:, :, [0, 0]]
        this_series_preds = preds[series_idx][:, :, [1, 2]].reshape(-1, 2)

        # sleep_diffs = np.diff(this_series_sleeps[:, :, 0], prepend=1).flatten()
        # sleep_diffs = np.clip(sleep_diffs, -0.001, 0.0001)
        # mask = sleep_diffs > 0  # & (sleep_diffs < 0.0001)
        # this_series_preds[mask, 0] += sleep_diffs[mask]
        # mask = sleep_diffs < 0  # & (sleep_diffs > -0.001)
        # this_series_preds[mask, 1] -= sleep_diffs[mask]

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # If there is no prediction, insert dummy
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    # sub_df = sub_df.with_columns(
    #     pl.when(pl.col("score") < 0).then(0).otherwise(pl.col("score")).alias("score"))
    # sub_df = sub_df.with_columns(
    #     pl.when(pl.col("score") > 1).then(1).otherwise(pl.col("score")).alias("score"))
    return sub_df