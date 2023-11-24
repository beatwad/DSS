import glob
import json
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.conf import InferenceConfig
from src.datamodule import load_chunk_features
from src.dataset.common import get_test_ds
from src.models.base import BaseModel
from src.models.common import get_model
from src.utils.common import nearest_valid_size, trace
from src.utils.post_process import post_process_for_seg


def get_weight_paths(cfg: InferenceConfig):
    # load weights
    if cfg.weight is not None:
        if cfg.best_model == 'ensemble':
            weight_path = f'{cfg.dir.model_dir}/{cfg.weight["exp_name"]}/*.pth'
        elif cfg.best_model == 'loss':
            weight_path = f'{cfg.dir.model_dir}/{cfg.weight["exp_name"]}/*_loss.pth'
        elif cfg.best_model == 'score':
            weight_path = f'{cfg.dir.model_dir}/{cfg.weight["exp_name"]}/*_score.pth'
        else:
            weight_path = f'{cfg.dir.model_dir}/{cfg.weight["exp_name"]}/*_epoch.pth'
        
        if cfg.best_model == 'ensemble':
            weight_paths = glob.glob(weight_path)
        else:
            weight_paths = [glob.glob(weight_path)[0]]
        return weight_paths



def load_model(cfg: InferenceConfig, weight_paths: list) -> BaseModel:
    num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    models = list()
    
    for weight_path in weight_paths:
    
        model = get_model(
            cfg,
            feature_dim=len(cfg.features),
            n_classes=len(cfg.labels),
            num_timesteps=num_timesteps // cfg.downsample_rate,
            test=True,
        )

        model.load_state_dict(torch.load(weight_path))
        models.append(model)
        print(f'load weight from {weight_path}')
    
    return models


def get_test_dataloader(cfg: InferenceConfig) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    feature_dir = Path(cfg.dir.processed_dir) / cfg.phase
    series_ids = [x.name for x in feature_dir.glob("*")]
    chunk_features = load_chunk_features(
        duration=cfg.duration,
        feature_names=cfg.features,
        series_ids=series_ids,
        processed_dir=Path(cfg.dir.processed_dir),
        phase=cfg.phase,
    )
    test_dataset = get_test_ds(cfg, chunk_features=chunk_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def inference(
    duration: int, loader: DataLoader, model: BaseModel, device: torch.device, use_amp
) -> tuple[list[str], np.ndarray]:
    model = model.to(device)
    model.eval()

    preds = []
    keys = []
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = batch["feature"].to(device)
                output = model.predict(
                    x,
                    org_duration=duration,
                )
            if output.preds is None:
                raise ValueError("output.preds is None")
            else:
                key = batch["key"]
                preds.append(output.preds.detach().cpu().numpy())
                keys.extend(key)

    preds = np.concatenate(preds)

    return keys, preds  # type: ignore


def make_submission(
    keys: list[str], preds: np.ndarray, series_lens: np.ndarray, score_th, distance, offset
) -> pl.DataFrame:
    sub_df = post_process_for_seg(
        keys,
        preds,  # type: ignore
        series_lens,
        score_th=score_th,
        distance=distance,  # type: ignore
        offset=offset
    )
    return sub_df


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with trace("load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)
    with trace("load model"):
        weights = get_weight_paths(cfg)
        models = load_model(cfg, weights)
    
    keys, preds = None, None
    
    for model in models:
        with trace("inference"):
            tmp_keys, tmp_preds = inference(cfg.duration, test_dataloader, model, device, 
                                            use_amp=cfg.use_amp)
            if keys is None:
                keys = tmp_keys
                preds = tmp_preds
            else:
                preds += tmp_preds

    preds /= len(models)
    
    np.save(Path(cfg.dir.sub_dir) / "keys.npy", keys)
    np.save(Path(cfg.dir.sub_dir) / "preds.npy", preds)
        
    with open (Path(cfg.dir.processed_dir) / cfg.phase / 'series_lens.json') as f:
        series_lens = json.load(f)
    
    with trace("make submission"):
        sub_df = make_submission(
            keys,
            preds,
            series_lens,
            score_th=cfg.pp.score_th,
            distance=cfg.pp.distance,
            offset=cfg.pp.offset
        )
    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    main()
