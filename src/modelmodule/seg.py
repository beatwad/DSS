from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import SWALR, AveragedModel
from torchvision.transforms.functional import resize
from transformers import get_cosine_schedule_with_warmup

from src.datamodule.seg import nearest_valid_size
from src.models.common import get_model
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg


# class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
#     """
#     Cosine anneal scheduler that decays after every cycle
#     """
#     def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
#         super(CosineAnnealingWarmRestartsDecay, self).__init__(optimizer, T_0, T_mult, eta_min, 
#                                                                last_epoch)
#         self.T_0 = T_0
#         self.optimizer = optimizer

#     def step(self, step=None):
#         if step is not None and step > 0 and step % self.T_0 == 0: # step >= self.min_epoch
#             self.base_lrs = [lrs * self.decay for lrs in self.base_lrs]  
#         super().step(step)       


class SegModel(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        val_event_df: pl.DataFrame,
        feature_dim: int,
        num_classes: int,
        duration: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.lr = self.cfg.optimizer.lr
        self.val_event_df = val_event_df
        num_timesteps = nearest_valid_size(int(duration * cfg.upsample_rate), cfg.downsample_rate)
        self.model = get_model(
            cfg,
            feature_dim=feature_dim,
            n_classes=num_classes,
            num_timesteps=num_timesteps // cfg.downsample_rate,
        )
        self.duration = duration
        self.validation_step_outputs: list = []
        self.__best_loss = np.inf
        self.__best_score = 0

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Optional[torch.Tensor]]:
        return self.model(x, labels)

    def training_step(self, batch, batch_idx):
        return self.__share_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.__share_step(batch, "val")

    def __share_step(self, batch, mode: str) -> torch.Tensor:
        if mode == "train":
            do_mixup = np.random.rand() < self.cfg.augmentation.mixup_prob
            do_cutmix = np.random.rand() < self.cfg.augmentation.cutmix_prob
        elif mode == "val":
            do_mixup = False
            do_cutmix = False

        output = self.model(batch["feature"], batch["label"], do_mixup, do_cutmix)
        loss: torch.Tensor = output["loss"]
        logits = output["logits"]  # (batch_size, n_timesteps, n_classes)

        if mode == "train":
            self.log(
                f"{mode}_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        elif mode == "val":
            # stretch array to required size using bilinear interpolation
            resized_logits = resize(
                logits.sigmoid().detach().cpu(),
                size=[self.duration, logits.shape[2]],
                antialias=False,
            )
            
            resized_labels = resize(
                batch["label"].detach().cpu(),
                size=[self.duration, logits.shape[2]],
                antialias=False,
            )
            
            self.validation_step_outputs.append(
                (
                    batch["key"],
                    resized_labels.numpy(),
                    resized_logits.numpy(),
                    loss.detach().item(),
                )
            )
            self.log(
                f"{mode}_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        return loss

    def on_validation_epoch_end(self):
        keys = []
        for x in self.validation_step_outputs:
            keys.extend(x[0])
        labels = np.concatenate([x[1] for x in self.validation_step_outputs])
        preds = np.concatenate([x[2] for x in self.validation_step_outputs])
        losses = np.array([x[3] for x in self.validation_step_outputs])
        loss = losses.mean()

        val_pred_df = post_process_for_seg(
            keys=keys,
            preds=preds[:, :, [1, 2]],
            score_th=self.cfg.post_process.score_th,
            distance=self.cfg.post_process.distance,
        )
        score = event_detection_ap(self.val_event_df.to_pandas(), val_pred_df.to_pandas())
        self.log("val_score", score, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        if score > self.__best_score:
            np.save("keys.npy", np.array(keys))
            np.save("labels_score.npy", labels)
            np.save("preds_score.npy", preds)
            val_pred_df.write_csv("val_pred_score_df.csv")
            torch.save(self.model.state_dict(), f"best_model_score.pth")
            print(f"Save best score model {self.__best_score} -> {score}, epoch {self.current_epoch}")
            self.__best_score = score
        if loss < self.__best_loss:
            np.save("keys.npy", np.array(keys))
            np.save("labels_loss.npy", labels)
            np.save("preds_loss.npy", preds)
            val_pred_df.write_csv("val_pred_loss_df.csv")
            torch.save(self.model.state_dict(), f"best_model_loss.pth")
            print(f"Save best loss model {self.__best_loss} -> {loss}, epoch {self.current_epoch}")
        if self.current_epoch > 30 and self.current_epoch % 5 == 0:
            np.save("keys.npy", np.array(keys))
            np.save(f"labels_ep_{self.current_epoch}.npy", labels)
            np.save(f"preds_loss_ep_{self.current_epoch}.npy", preds)
            val_pred_df.write_csv(f"val_pred_df_ep_{self.current_epoch}.csv")
            torch.save(self.model.state_dict(), f"model_ep_{self.current_epoch}.pth")
            print(f"Save model, epoch {self.current_epoch}")
        
        self.__best_loss = min(self.__best_loss, loss)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # if self.cfg.scheduler.type == 'min_lr':
        scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                T_0=self.trainer.max_steps // 4, 
                                                T_mult=1, 
                                                eta_min=0, 
                                                last_epoch=-1)
        # else:
        #     scheduler = get_cosine_schedule_with_warmup(
        #         optimizer, num_training_steps=self.trainer.max_steps, 
        #         num_warmup_steps=self.cfg.scheduler.num_warmup_steps
        #     )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        # if self.cfg.scheduler.type == 'min_lr':
            # scheduler.step()
            # scheduler.step(step=self.global_step)
        # else:
        scheduler.step()