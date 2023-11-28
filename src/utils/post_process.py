import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def clean_weak_events(sub_df, event_threshold):
    # delete onset and wakeup events that are too far from the same events with high score
    sub_df["prev_onset_step"] = np.nan
    sub_df["next_onset_step"] = np.nan
    sub_df["prev_wakeup_step"] = np.nan
    sub_df["next_wakeup_step"] = np.nan
    sub_df["to_del"] = 0

    # onset
    mask = (sub_df["event"] == "onset") & (sub_df["score"] >= 0.1)
    sub_df.loc[mask, "prev_high_score_onset_step"] = sub_df.loc[mask, "step"]
    sub_df.loc[mask, "next_high_score_onset_step"] = sub_df.loc[mask, "step"]
    sub_df["prev_high_score_onset_step"] = sub_df["prev_high_score_onset_step"].ffill()
    sub_df["next_high_score_onset_step"] = sub_df["next_high_score_onset_step"].bfill()

    mask = (
        (sub_df["event"] == "onset")
        & (sub_df["step"] - sub_df["prev_high_score_onset_step"] >= event_threshold)
        & (sub_df["next_high_score_onset_step"] - sub_df["step"] >= event_threshold)
    )
    sub_df.loc[mask, "to_del"] = 1

    # wakeup
    mask = (sub_df["event"] == "wakeup") & (sub_df["score"] >= 0.1)
    sub_df.loc[mask, "prev_high_score_wakeup_step"] = sub_df.loc[mask, "step"]
    sub_df.loc[mask, "next_high_score_wakeup_step"] = sub_df.loc[mask, "step"]
    sub_df["prev_high_score_wakeup_step"] = sub_df["prev_high_score_wakeup_step"].ffill()
    sub_df["next_high_score_wakeup_step"] = sub_df["next_high_score_wakeup_step"].bfill()

    mask = (
        (sub_df["event"] == "wakeup")
        & (sub_df["step"] - sub_df["prev_high_score_wakeup_step"] >= event_threshold)
        & (sub_df["next_high_score_wakeup_step"] - sub_df["step"] >= event_threshold)
    )
    sub_df.loc[mask, "to_del"] = 1

    # remove these events
    sub_df = sub_df[sub_df["to_del"] == 0]

    return sub_df


def clean_too_far_events(sub_df, event_threshold):
    # delete onset and wakeup events that are too far from next wakeup / previous onset event
    sub_df["prev_onset_step"] = np.nan
    sub_df["next_onset_step"] = np.nan
    sub_df["prev_wakeup_step"] = np.nan
    sub_df["next_wakeup_step"] = np.nan
    sub_df["to_del"] = 0

    # onset
    mask = sub_df["event"] == "onset"
    sub_df.loc[mask, "prev_onset_step"] = sub_df.loc[mask, "step"]
    sub_df.loc[mask, "next_onset_step"] = sub_df.loc[mask, "step"]
    sub_df["prev_onset_step"] = sub_df["prev_onset_step"].ffill()
    sub_df["next_onset_step"] = sub_df["next_onset_step"].bfill()

    mask = (sub_df["event"] == "wakeup") & (
        sub_df["step"] - sub_df["prev_onset_step"] >= event_threshold
    )
    sub_df.loc[mask, "to_del"] = 1

    # wakeup
    mask = sub_df["event"] == "wakeup"
    sub_df.loc[mask, "prev_wakeup_step"] = sub_df.loc[mask, "step"]
    sub_df.loc[mask, "next_wakeup_step"] = sub_df.loc[mask, "step"]
    sub_df["prev_wakeup_step"] = sub_df["prev_wakeup_step"].ffill()
    sub_df["next_wakeup_step"] = sub_df["next_wakeup_step"].bfill()

    mask = (sub_df["event"] == "onset") & (
        sub_df["next_wakeup_step"] - sub_df["step"] >= event_threshold
    )
    sub_df.loc[mask, "to_del"] = 1

    # remove these events
    sub_df = sub_df[sub_df["to_del"] == 0]

    return sub_df


def delete_alone_events(sub_df, alone_threshold):
    # detect alone onset event that are between two other onset events and delete them
    sub_df["to_del"] = 0
    
    # onset
    mask_1 = (
        sub_df["step"] - sub_df["prev_onset_step"].shift(1)
        < sub_df["step"] - sub_df["prev_wakeup_step"]
    )
    mask_2 = sub_df["step"] - sub_df["prev_onset_step"].shift(1) >= alone_threshold
    mask_3 = (
        sub_df["next_onset_step"].shift(-1) - sub_df["step"]
        < sub_df["next_wakeup_step"] - sub_df["step"]
    )
    mask_4 = sub_df["next_onset_step"].shift(-1) - sub_df["step"] >= alone_threshold
    mask_5 = sub_df["event"] == "onset"
    mask_6 = sub_df["score"] < 0.1
    sub_df.loc[mask_1 & mask_2 & mask_3 & mask_4 & mask_5 & mask_6, "to_del"] = 1

    # wakeup
    mask_1 = (
        sub_df["step"] - sub_df["prev_wakeup_step"].shift(1)
        < sub_df["step"] - sub_df["prev_onset_step"]
    )
    mask_2 = sub_df["step"] - sub_df["prev_wakeup_step"].shift(1) >= alone_threshold
    mask_3 = (
        sub_df["next_wakeup_step"].shift(-1) - sub_df["step"]
        < sub_df["next_onset_step"] - sub_df["step"]
    )
    mask_4 = sub_df["next_wakeup_step"].shift(-1) - sub_df["step"] >= alone_threshold
    mask_5 = sub_df["event"] == "wakeup"
    mask_6 = sub_df["score"] < 0.1
    sub_df.loc[mask_1 & mask_2 & mask_3 & mask_4 & mask_5 & mask_6, "to_del"] = 1

    # remove these events
    sub_df = sub_df[sub_df["to_del"] == 0]

    return sub_df

    
def delete_events_among_differnet_evnets(sub_df, close_threshold):
    # delete events that are among many different events
    sub_df["to_del"] = 0
    
    # onset
    mask_1 = (
        (sub_df["event"].shift(1) == "wakeup")
        & (sub_df["event"].shift(2) == "wakeup")
        & (sub_df["event"].shift(3) == "wakeup")
    )
    mask_2 = (
        (sub_df["event"].shift(-1) == "wakeup")
        & (sub_df["event"].shift(-2) == "wakeup")
        & (sub_df["event"].shift(-3) == "wakeup")
    )
    mask_3 = sub_df["event"] == "onset"
    mask_4 = sub_df["step"] - sub_df["prev_wakeup_step"] <= close_threshold
    mask_5 = sub_df["next_wakeup_step"] - sub_df["step"] <= close_threshold
    mask_6 = sub_df["score"] < 0.1

    sub_df.loc[mask_1 & mask_2 & mask_3 & mask_4 & mask_5 & mask_6, "to_del"] = 1

    # wakeup
    mask_1 = (
        (sub_df["event"].shift(1) == "onset")
        & (sub_df["event"].shift(2) == "onset")
        & (sub_df["event"].shift(3) == "onset")
    )
    mask_2 = (
        (sub_df["event"].shift(-1) == "onset")
        & (sub_df["event"].shift(-2) == "onset")
        & (sub_df["event"].shift(-3) == "onset")
    )
    mask_3 = sub_df["event"] == "wakeup"
    mask_4 = sub_df["step"] - sub_df["prev_onset_step"] <= 360
    mask_5 = sub_df["next_onset_step"] - sub_df["step"] <= 360
    mask_6 = sub_df["score"] < 0.1
    sub_df.loc[mask_1 & mask_2 & mask_3 & mask_4 & mask_5 & mask_6, "to_del"] = 1

    # remove these events
    sub_df = sub_df[sub_df["to_del"] == 0]

    return sub_df


def post_process_for_seg(
    keys: list[str],
    preds: np.ndarray,
    series_lens: dict,
    score_th: float = 0.01,
    distance: int = 5000,
    offset: int = 0,
) -> pd.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pd.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        max_step = series_lens[series_id]

        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        onset_event_preds = this_series_preds[:, 0]
        onset_steps = find_peaks(onset_event_preds, height=score_th, distance=distance)[0]
        onset_scores = onset_event_preds[onset_steps]
        min_onset_step = min(onset_steps) if len(onset_steps) > 0 else 0

        wakeup_event_preds = this_series_preds[:, 1]
        wakeup_steps = find_peaks(wakeup_event_preds, height=score_th, distance=distance)[0]
        wakeup_scores = wakeup_event_preds[wakeup_steps]
        max_wakeup_step = max(wakeup_steps) if len(wakeup_steps) > 0 else 0

        for step, score in zip(onset_steps, onset_scores):
            # select only wakeups than has at least one onset before
            # and not too close to series borders
            if step >= max_wakeup_step or step <= 720 or step >= max_step - 720 * offset:
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
            # and not too close to series borders
            if step <= min_onset_step or step <= 720 * offset or step >= max_step - 180:
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

    sub_df = pd.DataFrame(records).sort_values(["series_id", "step"]).reset_index(drop=True)
    sub_df["row_id"] = np.arange(len(sub_df))

    # delete onset and wakeup events that are too far from the same events with high score
    # high_score_event_threshold = 17280 * 1.01 # 24 hours
    # sub_df = clean_weak_events(sub_df, high_score_event_threshold)
    
    # # delete onset and wakeup events that are too far from next wakeup / previous onset event
    # event_threshold = 14400 # 20 hours
    # sub_df = clean_too_far_events(sub_df, event_threshold)

    # # detect alone onset event that are between two other onset events and delete them
    # alone_threshold = 2880 # 2 hours
    # sub_df = delete_alone_events(sub_df, alone_threshold)

    # # # delete events that are among many different events
    # close_threshold = 360 # 30 minutes
    # sub_df = delete_events_among_differnet_evnets(sub_df, close_threshold)

    sub_df = sub_df[["row_id", "series_id", "step", "event", "score"]]

    return sub_df