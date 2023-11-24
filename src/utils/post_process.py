import numpy as np
import polars as pl
from scipy.signal import find_peaks


def post_process_for_seg(
    keys: list[str], preds: np.ndarray, score_th: float = 0.01, distance: int = 5000, 
    offset: int = 0) -> pl.DataFrame:
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
        this_series_preds = preds[series_idx].reshape(-1, 2)

        onset_event_preds = this_series_preds[:, 0]
        onset_steps = find_peaks(onset_event_preds, height=score_th, distance=distance)[0]
        onset_scores = onset_event_preds[onset_steps]
        min_onset_step = min(onset_steps)

        wakeup_event_preds = this_series_preds[:, 1]
        wakeup_steps = find_peaks(wakeup_event_preds, height=score_th, distance=distance)[0]
        wakeup_scores = wakeup_event_preds[wakeup_steps]
        max_wakeup_step = max(wakeup_steps)

        for step, score in zip(onset_steps, onset_scores):
            # select only wakeups than has at least one onset before
            if (step >= max_wakeup_step 
                or step <= 720 
                or step >= len(this_series_preds) - 720 * offset
                ):
                continue

            records.append(
                {
                    "series_id": series_id,
                    "step": step,
                    "event": "onset",
                    "score": score,
                }
            )

        for step, score in zip(wakeup_steps, wakeup_scores):
            # select only onsets than has at least one wakeup after
            if (step <= min_onset_step 
                or step <= 720 * offset 
                or step >= len(this_series_preds) - 720
                ):
                continue

            records.append(
                {
                    "series_id": series_id,
                    "step": step,
                    "event": "wakeup",
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

    return sub_df