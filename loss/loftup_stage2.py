import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from ema_pytorch import EMA
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from src.utilities.train_utils import StepsCounter

from ..upsamplers.upsamplers import (
    CATransformer,
    ChannelNorm,
    UpsamplerwithChannelNorm,
    get_upsampler,
    load_upsampler_weights,
)
from .loftup_stage1 import LoftUpStage1Loss
from .utils import (
    AttentionDownsampler,
    TVLoss,
    adjust_features_with_masks,
    apply_jitter,
    compute_affinity_matrix_batch,
    create_random_projection,
    entropy,
    mask_feature_similarity_loss,
    project,
    sample_transform,
)


@dataclass
class LoftUpStage2ModelOutput(ModelOutput):
    total_loss: Tensor | None = None
    consistency_loss: Tensor | None = None
    recon_loss: Tensor | None = None
    tv_loss: Tensor | None = None
    entropy_loss: Tensor | None = None
    global_img: Tensor | None = None
    global_masks: Tensor | None = None
    lr_feats: Tensor | None = None
    hr_feats: Tensor | None = None
    ct_cropped_up_feats_resized: Tensor | None = None
    ct_cropped_hr_feats: Tensor | None = None
    lr_aug_feats: Tensor | None = None
    hr_aug_feats: Tensor | None = None
    proj_hr_aug_feats: Tensor | None = None
    proj_downsampled_hr_feats: Tensor | None = None
    proj_lr_aug_feats: Tensor | None = None


class LoftUpStage2Loss(LoftUpStage1Loss):
    def __init__(
        self,
        featurizer: nn.Module,
        upsampler: nn.Module | UpsamplerwithChannelNorm,
        ema_upsampler: EMA | nn.Module | None,
        downsampler: nn.Module,
        dim: int,
        uncertainty_net: nn.Module | None = None,
        multi_upsample_size: bool = False,
        upsample_size: int = 224,
        resampled_img_size: int = 224,
        use_random_aug_size: bool = True,  # default to True
        project_dim: int = 64,
        clamp_featup: bool = True,
        kernel_entropy_weight: float = 0.0,
        tv_weight: float = 0.0,
        # Feature size specific
        upsampler_type: str = "loftup",
        final_feat_size=None,
        affinity_loss_type: str = "l1",
        # Augmentation
        n_augs: int = 4,
        augmentation_kwargs: dict = {},
        # High-resolution supervision
        hr_res: int = 224,
        consistency_weight: float = 0.0,
        consistency_method: str = "bilinear",
        n_freqs: int = 1,
        use_crop_upsampler: bool = False,
        # EMA specifics (if ema upsampler is not provided)
        # See ema_pytorch.EMA for details
        ema_kwargs: dict = {
            "beta": 0.999,
            "update_every": 10,
            "update_after_step": 0,
            "use_foreach": False,
        },
        # SAM specifics
        sam_model: nn.Module | None = None,
        sam_mask_alpha: float = 0.0,
        sam_mask_reg: float = 0.0,
    ):
        super().__init__(
            featurizer=featurizer,
            upsampler=upsampler,
            downsampler=downsampler,
            dim=dim,
            uncertainty_net=uncertainty_net,
            multi_upsample_size=multi_upsample_size,
            upsample_size=upsample_size,
            resampled_img_size=resampled_img_size,
            project_dim=project_dim,
            clamp_featup=clamp_featup,
            kernel_entropy_weight=kernel_entropy_weight,
            tv_weight=tv_weight,
            n_augs=n_augs,
            augmentation_kwargs=augmentation_kwargs,
            sam_model=sam_model,
            sam_mask_alpha=sam_mask_alpha,
            sam_mask_reg=sam_mask_reg,
        )

        ########## Models ###########
        self.ema_upsampler: EMA = (
            ema_upsampler if ema_upsampler is not None else EMA(upsampler, **ema_kwargs)
        )
        self._ema_update_after_step = self.ema_upsampler.update_after_step

        # High-resolution loss
        self.hr_res = hr_res
        self.hr_weight = consistency_weight
        self.consistency_method = consistency_method

        # Image sizes
        self.augmentation_size = [224, 336, 448, 518]  # loftup aug_size specific
        self.featurizer_patch_size = 16
        self._fixed_img_size = 224
        self.affinity_loss_type = affinity_loss_type
        self.upsampler_type = upsampler_type
        self.final_feat_size = final_feat_size
        self.use_random_aug_size = use_random_aug_size

        # Training state
        # May raise if not initialized in the trainer at first
        self._step_counter = StepsCounter(["train"])

        # Assertions
        assert affinity_loss_type in [None, "l1", "l2"], (
            f"Unsupported affinity_loss_type: {affinity_loss_type}"
        )

        self._use_affine_loss = self.affinity_loss_type is not None

    @property
    def _global_step(self):
        return self._step_counter["train"]

    @torch.no_grad()
    def _forward_ema_upsampler(self, feats: Tensor, img: Tensor):
        return self.ema_upsampler(feats, img)

    def _ensure_img_patchable(
        self, img: Tensor, mask: Tensor | None = None, patchable_type: str = "crop"
    ):
        """Ensure the image size is divsible by model's patch size"""
        hw = torch.tensor(img.shape[-2:])
        if patchable_type == "crop":
            # Crop to the least size that is divisible by patch size
            patchable_size = (
                hw // self.featurizer_patch_size
            ) * self.featurizer_patch_size
            img = img[:, :, : patchable_size[0], : patchable_size[1]]
            if mask is not None:
                mask = mask[:, :, : patchable_size[0], : patchable_size[1]]

        elif patchable_type == "resize":
            # Resize to the least size that is divisible by patch size
            patchable_size = (
                (hw + self.featurizer_patch_size - 1) // self.featurizer_patch_size
            ) * self.featurizer_patch_size
            img = F.interpolate(img, size=patchable_size.tolist(), mode="bilinear")
            if mask is not None:
                mask = F.interpolate(mask, size=patchable_size.tolist(), mode="nearest")

        else:
            raise ValueError(f"Unsupported patchable_type: {patchable_type}")

        return img, mask

    def _random_high_res_img_resize(
        self, img: Tensor, binary_masks: Tensor | None = None
    ):
        if self.use_random_aug_size:
            sz = random.choice(self.augmentation_size)
        else:
            sz = self._fixed_img_size

        hr_factor = self.hr_res / sz

        img = F.interpolate(img, size=sz, mode="bilinear", align_corners=False)
        if binary_masks is not None:
            binary_masks = F.interpolate(binary_masks, size=sz, mode="nearest")

        return img, binary_masks, hr_factor, sz

    def _random_crop_views_per_augmentation(
        self,
        orig_img: Tensor,
        orig_masks: Tensor | None,
        global_img_sz: int,
    ):
        """Random crop views for each augmentation step
        Says global_img_sz=224, the original image size is 512, then the cropped image/masks
        are also sized as 224 which cropped from the original image/masks.
        """
        H, W = orig_img.shape[2], orig_img.shape[3]
        crop_x, crop_y = (
            random.randint(0, H - global_img_sz),
            random.randint(0, W - global_img_sz),
        )
        crop_x, crop_y = (
            crop_x - crop_x % self.featurizer_patch_size,
            crop_y - crop_y % self.featurizer_patch_size,
        )

        # Crop the image and masks
        cropped_img = orig_img[
            :, :, crop_x : crop_x + global_img_sz, crop_y : crop_y + global_img_sz
        ]
        cropped_masks = None
        if orig_masks is not None:
            cropped_masks = orig_masks[
                :, :, crop_x : crop_x + global_img_sz, crop_y : crop_y + global_img_sz
            ]

        crop_params = edict({"x": crop_x, "y": crop_y, "size": global_img_sz})

        return cropped_img, cropped_masks, crop_params

    def _forward_self_consistency_loss(
        self,
        global_img: Tensor,
        orig_img: Tensor,
        global_masks: Tensor | None,
        orig_masks: Tensor | None,
        lr_feats: Tensor,
        hr_factor: float = 1.0,
        aug_index: int = 0,
    ):
        """Per-augmentation step consistency loss"""

        # Random crop the image and masks (crop out views)
        global_img_sz = global_img.shape[-1]
        cropped_img, cropped_masks, crop_params = (
            self._random_crop_views_per_augmentation(
                orig_img, orig_masks, global_img_sz
            )
        )

        # Upsample the LR features to HR features
        # FIXME: this needs do upsample per-augmentation?
        hr_feats = self._forward_upsampler(lr_feats, global_img)

        # Init all losses
        device = orig_img.device
        total_loss = torch.tensor(0.0, device=device)

        ############### Compute the consistency loss ################

        # Intermidiates
        cropped_feats = None
        cropped_up_feats = None
        cropped_feats_mask_adj = None
        cropped_hr_feats = None
        cropped_up_feats_resized = None

        if self._global_step > self._ema_update_after_step:
            # Only do the ema upsampler loss after the ema update starts
            with torch.no_grad():
                cropped_feats = self._forward_featurizer(cropped_img)
                cropped_up_feats = self._forward_ema_upsampler(
                    cropped_feats, cropped_img
                )
                cropped_up_feats = self._ensure_equal_size(
                    cropped_up_feats, cropped_img
                )

                # Mask adjustment
                if self.sam_mask_alpha > 0 and cropped_masks is not None:
                    cropped_up_masks = self._ensure_equal_size(
                        cropped_masks, cropped_up_feats, mode="nearest"
                    )
                    cropped_feats_mask_adj = adjust_features_with_masks(
                        cropped_up_feats, cropped_up_masks, alpha=self.sam_mask_alpha
                    )

            # Feature region in global image
            if self.upsampler_type == "loftup":
                feat_final_size = global_img_sz
            else:
                assert self.final_feat_size is not None, (
                    "final_feat_size must be specified for non-loftup upsampler"
                )
                feat_final_size = self.final_feat_size * self.featurizer_patch_size

            ########### Find the corresponding HR feature region ############

            cx, cy = crop_params.x, crop_params.y
            # If crop_size is 224, then
            # 224 / (fixed_hr_size / 224) * feat_final_size / glb_img_sz (from random)
            # if the feat_final_size = global_img_sz, then
            # 224 / (fixed_hr_size / 224)
            _cropped_hr_feat_loc_x = int(
                cx / hr_factor * feat_final_size / global_img_sz
            )
            _cropped_hr_feat_loc_y = int(
                cy / hr_factor * feat_final_size / global_img_sz
            )
            # glb_img_sz / (fixed_hr_size / 224) * feat_final_size / glb_img_sz
            # = 224 * feat_final_size / fixed_hr_size
            _cropped_hr_feat_sz = int(
                global_img_sz / hr_factor * feat_final_size / global_img_sz
            )

            # Crop the HR feature is corresponding region that aligns the same cropped image
            cropped_hr_feats = hr_feats[
                :,
                :,
                _cropped_hr_feat_loc_x : _cropped_hr_feat_loc_x + _cropped_hr_feat_sz,
                _cropped_hr_feat_loc_y : _cropped_hr_feat_loc_y + _cropped_hr_feat_sz,
            ]
            # Resize cropped HR feature to the same size as HR feature
            # ema_upsampler(crop(img)) ~= crop(upsampler(lr_feat))
            cropped_up_feats_resized = self._ensure_equal_size(
                cropped_up_feats, cropped_hr_feats
            )

            if self._use_affine_loss:
                # Affinity matrix loss
                affine_loss_fn = (
                    F.mse_loss if self.affinity_loss_type == "mse" else F.l1_loss
                )
                aff_matrix_hr = compute_affinity_matrix_batch(cropped_hr_feats)
                aff_matrix_up = compute_affinity_matrix_batch(cropped_up_feats_resized)
                # Replace the MSE self-consistency loss (in feature space) into token-level consistency loss
                consistency_loss = affine_loss_fn(aff_matrix_up, aff_matrix_hr)
            else:
                # Direct feature space loss
                consistency_loss = F.mse_loss(
                    cropped_up_feats_resized, cropped_hr_feats
                )
            total_loss += consistency_loss * self.hr_weight
        else:
            # If not start ema update, set the consistency to zero.
            consistency_loss = self._zero
            total_loss += consistency_loss * self.hr_weight

        ############ Augmentation loss ############
        # Same to stage1 loss
        aug_ret = self._forward_augmentation_loss(
            lr_feats,
            hr_feats,
            orig_img,
            global_img,
            global_masks,  # not used
            aug_index,
        )
        total_loss += aug_ret.augmentation_loss

        ret = {
            # Losses
            "total_loss": total_loss,
            # Consistency loss
            "consistency_loss": consistency_loss.detach(),
            # Augmentation losses
            "recon_loss": aug_ret.recon_loss,
            "tv_loss": aug_ret.tv_loss,
            "entropy_loss": aug_ret.entropy_loss,
            # Features
            "hr_feats": hr_feats,
            # Consistency features
            "ct_cropped_feats": cropped_feats,
            "ct_cropped_up_feats": cropped_up_feats,
            "ct_croppped_mask_adj_feats": cropped_feats_mask_adj,
            "ct_cropped_hr_feats": cropped_hr_feats,
            "ct_cropped_up_feats_resized": cropped_up_feats_resized,
            # Augmentation features
            "lr_aug_feats": aug_ret.lr_aug_feats,
            "hr_aug_feats": aug_ret.hr_aug_feats,
            "proj_hr_aug_feats": aug_ret.proj_hr_aug_feats,
            "proj_downsampled_hr_feats": aug_ret.proj_downsampled_hr_feats,
            "proj_lr_aug_feats": aug_ret.proj_lr_aug_feats,
        }

        return edict(ret)

    def forward(self, img: Tensor, binary_masks: Tensor | None = None):
        orig_img = img.clone()
        orig_masks = binary_masks.clone() if binary_masks is not None else None

        # Random high-resolution image resize
        global_img, global_masks, hr_factor, inp_size = (
            self._random_high_res_img_resize(img, binary_masks)
        )

        # LR features for the pretrained VFM
        lr_feats = self.featurizer(global_img)

        # Losses breakdowns
        total_loss = torch.tensor(0.0, device=img.device)
        total_ct_loss = self._zero
        total_recon_loss = self._zero
        total_tv_loss = self._zero

        # Augmentation loop
        for aug_index in range(self.n_augs):
            aug_loss_dict = self._forward_self_consistency_loss(
                global_img,
                orig_img,
                global_masks,
                orig_masks,
                lr_feats,
                hr_factor,
                aug_index,
            )
            total_loss = total_loss + aug_loss_dict.total_loss / self.n_augs
            # Logs
            total_ct_loss = total_ct_loss + aug_loss_dict.consistency_loss / self.n_augs
            total_recon_loss = total_recon_loss + aug_loss_dict.recon_loss / self.n_augs
            total_tv_loss = total_tv_loss + aug_loss_dict.tv_loss / self.n_augs

        ret = {
            "total_loss": total_loss,
            # Losses breakdowns
            "consistency_loss": total_ct_loss,
            "recon_loss": total_recon_loss,
            "tv_loss": total_tv_loss,
            # Images
            "global_img": global_img,
            "global_masks": global_masks,
            # Features
            "lr_feats": lr_feats,
            "hr_feats": aug_loss_dict.hr_feats,
            # Loss computation intermediates
            "ct_cropped_up_feats_resized": aug_loss_dict.ct_cropped_up_feats_resized,
            "ct_cropped_hr_feats": aug_loss_dict.ct_cropped_hr_feats,
            "lr_aug_feats": aug_loss_dict.lr_aug_feats,
            "hr_aug_feats": aug_loss_dict.hr_aug_feats,
            "proj_hr_aug_feats": aug_loss_dict.proj_hr_aug_feats,
            "proj_downsampled_hr_feats": aug_loss_dict.proj_downsampled_hr_feats,
            "proj_lr_aug_feats": aug_loss_dict.proj_lr_aug_feats,
        }

        return LoftUpStage2ModelOutput(**ret)
