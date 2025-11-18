import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from src.utilities.config_utils import function_config_to_easy_dict

from ..upsamplers.upsamplers import (
    CATransformer,
    ChannelNorm,
    UpsamplerwithChannelNorm,
    get_upsampler,
    load_upsampler_weights,
)
from .utils import (
    AttentionDownsampler,
    TVLoss,
    adjust_features_with_masks,
    apply_jitter,
    create_random_projection,
    entropy,
    mask_feature_similarity_loss,
    project,
    sample_transform,
)


@dataclass
class LoftUpStage1Output(ModelOutput):
    total_loss: Tensor | None = None
    # Augmentation losses
    augmentation_loss: Tensor | None = None
    recon_loss: Tensor | None = None
    tv_loss: Tensor | None = None
    entropy_loss: Tensor | None = None
    # SAM adjustment losses
    sam_adjust_loss: Tensor | None = None
    mask_recon_loss: Tensor | None = None
    mask_reg_loss: Tensor | None = None
    # Features
    adjusted_up_lr_feats: Tensor | None = None
    hr_feats: Tensor | None = None
    lr_aug_feats: Tensor | None = None
    hr_aug_feats: Tensor | None = None
    # Projected features
    proj_hr_aug_feats: Tensor | None = None
    proj_downsampled_hr_feats: Tensor | None = None
    proj_lr_aug_feats: Tensor | None = None


class _AugmentationCollections:
    _supported_augmentations = ["loftup"]

    @classmethod
    def __class_getitem__(cls, item):
        assert item in cls._supported_augmentations, f"Unsupported augmentation: {item}"
        return getattr(cls, item)

    @classmethod
    def loftup(
        cls,
        img: Tensor,
        max_zoom=1.2,
        max_rotate=0.0,
        max_pad=0,
        *,
        params: tuple | None = None,
    ):
        # Sample the augmentation parameters
        if params is None:
            params = sample_transform(
                False,
                0,
                max_zoom,
                h=img.shape[2],
                w=img.shape[3],
                max_rotation=max_rotate,
            )
        # Apply the augmentations
        img = apply_jitter(img, max_pad, params)
        return img, params


class LoftUpStage1Loss(nn.Module):
    @function_config_to_easy_dict
    def __init__(
        self,
        featurizer: nn.Module,
        upsampler: nn.Module,
        downsampler: nn.Module,
        dim: int,
        uncertainty_net: nn.Module | None = None,
        multi_upsample_size: bool = False,
        upsample_size: int = 224,
        resampled_img_size: int = 224,
        project_dim: int = 64,
        clamp_featup: bool = True,
        kernel_entropy_weight: float = 0.0,
        tv_weight: float = 0.0,
        # Augmentation
        n_augs: int = 4,
        augmentation_kwargs: dict = {},
        # SAM specific
        sam_model: nn.Module | None = None,
        sam_mask_alpha: float = 0.8,
        sam_mask_reg: float = 0.0,
    ):
        super().__init__()

        ############## Models ################
        self.featurizer = featurizer
        self.upsampler = upsampler
        self.downsampler = downsampler
        self.uncertainty_net = uncertainty_net
        self.sam_model = sam_model

        # Function
        self.tv_loss = TVLoss()

        self.upsample_size = upsample_size
        self.resampled_img_size = resampled_img_size
        self.multi_upsample_size = multi_upsample_size
        self.random_proj_dim = project_dim
        self.clamp_featup = clamp_featup
        self.kernel_entropy_weight = kernel_entropy_weight
        self.tv_weight = tv_weight

        # Augmentations
        self.n_augs = n_augs
        self.augmentation_kwargs = augmentation_kwargs
        self.augment_fn = _AugmentationCollections["loftup"]

        # SAM specific
        self.sam_mask_alpha = sam_mask_alpha
        self.sam_mask_reg = sam_mask_reg
        self._sam_model_online = sam_model is not None

        self._pred_uncertainty = uncertainty_net is not None
        self._use_dim_proj = project_dim is not None and project_dim > 0
        self._use_entropy_loss = kernel_entropy_weight > 0.0
        self._zero = nn.Buffer(torch.zeros(1))

    def _forward_upsampler(self, feat, x):
        return self.upsampler(feat, x)

    @torch.no_grad()
    def _forward_featurizer(self, x) -> Tensor:
        return self.featurizer(x)

    def _resample_global_img_and_masks(self, img, binary_masks: Tensor | None = None):
        if self.multi_upsample_size:
            sample_size = random.choice(
                [self.upsample_size // 4, self.upsample_size // 2, self.upsample_size]
            )
        else:
            sample_size = self.upsample_size

        # Resample the image and masks
        global_img = F.interpolate(
            img, size=(sample_size, sample_size), mode="bilinear", align_corners=False
        )
        if binary_masks is not None:
            binary_masks = F.interpolate(
                binary_masks, size=(sample_size, sample_size), mode="nearest"
            )

        return global_img, binary_masks

    def _resample_inputs(self, img: Tensor, binary_masks: Tensor | None = None):
        """Resample inputs to desired upsample size."""
        input_img_size: int = self.resampled_img_size  # TODO: this is desired?

        global_img, binary_masks = self._resample_global_img_and_masks(
            img, binary_masks
        )

        # Resample image
        img = F.interpolate(
            img,
            size=(input_img_size, input_img_size),
            mode="bilinear",
            align_corners=False,
        )
        global_img = F.interpolate(
            global_img,
            size=(input_img_size, input_img_size),
            mode="bilinear",
            align_corners=False,
        )
        if binary_masks is not None:
            binary_masks = F.interpolate(
                binary_masks, size=(input_img_size, input_img_size), mode="nearest"
            )
        return img, global_img, binary_masks

    def _get_lr_feature(self, img: Tensor):
        lr_feat = self._forward_featurizer(img)
        return lr_feat

    def _ensure_equal_size(self, interp_img, ref_img, mode="bilinear"):
        if interp_img.shape[2] != ref_img.shape[2]:
            interp_img = F.interpolate(
                interp_img,
                size=ref_img.shape[2:],
                mode=mode,
                align_corners=False if mode != "nearest" else None,
            )

        return interp_img

    def _forward_sam_adjust_loss(
        self,
        lr_feats: Tensor,
        hr_feats: Tensor,
        binary_masks: Tensor,
        img: Tensor,
        guidance_img: Tensor,
    ):
        if self.sam_mask_alpha <= 0.0:
            return 0.0

        # Upsample the lr feature
        up_lr_feats = F.interpolate(
            lr_feats, size=guidance_img.shape[-2:], mode="bilinear"
        )

        # Adjust lr features with binary masks
        adjusted_up_feats = adjust_features_with_masks(
            up_lr_feats, binary_masks, alpha=self.sam_mask_alpha
        )

        # Projection
        if self._use_dim_proj:
            proj_matrix = create_random_projection(lr_feats, self.random_proj_dim)
            dd_hr_feats = project(hr_feats, proj_matrix)
            dd_adjusted_up_feats = project(adjusted_up_feats, proj_matrix)
        else:
            dd_hr_feats, dd_adjusted_up_feats = hr_feats, adjusted_up_feats

        # Projected upsampled featues ~= Projected SAM mask adjusted features
        mask_up_recon_loss = ((dd_adjusted_up_feats - dd_hr_feats) ** 2).mean()
        sam_loss = mask_up_recon_loss

        # Mask regularization loss
        # The features in one mask class are encouraged to be similar to their mean
        mask_reg_loss = self._zero
        if self.sam_mask_reg > 0.0:
            mask_reg_loss = (
                mask_feature_similarity_loss(hr_feats, binary_masks) * self.sam_mask_reg
            )
            sam_loss += mask_reg_loss

        return edict(
            {
                "sam_adjust_loss": sam_loss,
                # Sub-losses
                "mask_recon_loss": mask_up_recon_loss.detach(),
                "mask_reg_loss": mask_reg_loss.detach(),
                # Features
                "adjusted_up_lr_feats": adjusted_up_feats,
            }
        )

    def _forward_augmentation_loss(
        self,
        lr_feats: Tensor,
        hr_feats: Tensor | None,
        orig_img: Tensor,
        global_img: Tensor,
        binary_masks: Tensor | None,
        aug_index: int,
    ):
        # Upsample the un-augmented feature
        if hr_feats is None:
            hr_feats = self._forward_upsampler(lr_feats, global_img)
            hr_feats = self._ensure_equal_size(hr_feats, orig_img)
        assert hr_feats.shape[2:] == orig_img.shape[2:], (
            "HR features and guidance image must have the same spatial size."
        )

        # Augment the image and then get augmented feature
        aug_img, aug_params_ = self.augment_fn(global_img, **self.augmentation_kwargs)
        aug_img = self._ensure_equal_size(aug_img, global_img)
        lr_aug_feats = self._get_lr_feature(aug_img)

        # Random feature projection
        # Apply the same the projection to the HR features
        proj_matrix = create_random_projection(lr_feats, self.random_proj_dim)
        hr_aug_feats, _ = self.augment_fn(
            hr_feats, params=aug_params_, **self.augmentation_kwargs
        )
        hr_aug_feats = self._ensure_equal_size(hr_aug_feats, global_img)

        # Projection features to some smaller dimensions
        # Downsampled(proj(HR)) ~= proj(LR)
        # TODO: this projection is necessary?
        proj_hr_aug_feats = project(hr_aug_feats, proj_matrix)
        dd_ds_hr_feats = self.downsampler(proj_hr_aug_feats, aug_img)
        dd_lr_aug_feats = project(lr_aug_feats, proj_matrix)
        multi_view_loss = (dd_ds_hr_feats - dd_lr_aug_feats) ** 2

        # Reconstruction loss
        if self._pred_uncertainty:
            # has uncertainty estimation
            assert self.uncertainty_net is not None, "Uncertainty net is not provided."
            uncentainty = self.uncertainty_net(lr_aug_feats)
            _eps = 1e-8
            uc_factor = 1 / ((2 * uncentainty**2) + _eps)
            uncentainty_loss = uncentainty.log()
            recon_loss = (multi_view_loss * uc_factor + uncentainty_loss).mean()
        else:
            # no uncertainty estimation
            recon_loss = multi_view_loss.mean()

        if self.clamp_featup:
            recon_loss = torch.clamp(recon_loss, 0.0)

        # Compute CRF loss?
        ...

        # Entropy loss
        entropy_loss = self._zero
        if self._use_entropy_loss:
            # TODO: find out this is used or not.
            entropy_loss = (
                entropy(self.downsampler.get_kernel()) * self.kernel_entropy_weight
            )

        # TV loss (the first augmentation)
        tv_loss = self._zero
        if aug_index == 0 and self.tv_weight > 0.0:
            tv_loss = self.tv_loss(lr_aug_feats) * self.tv_weight

        # Composite all losses
        aug_loss = recon_loss + tv_loss - entropy_loss

        ret = edict(
            {
                "augmentation_loss": aug_loss,
                # Sub-losses
                "recon_loss": recon_loss.detach(),
                "tv_loss": tv_loss.detach(),
                "entropy_loss": entropy_loss.detach(),
                # Features
                "hr_feats": hr_feats,
                "lr_aug_feats": lr_aug_feats,
                "hr_aug_feats": hr_aug_feats,
                # Projected features
                "proj_hr_aug_feats": proj_hr_aug_feats,  # Projected (down-dim) hr features
                "proj_downsampled_hr_feats": dd_ds_hr_feats,  # Projected downsampled hr features
                "proj_lr_aug_feats": dd_lr_aug_feats,  # Projected lr augmented features
            }
        )

        return ret

    def forward(self, img: Tensor, binary_masks: Tensor):
        total_loss: torch.Tensor = torch.tensor(0.0, device=img.device)

        # Resample inputs
        img, guidance_img, binary_masks = self._resample_inputs(img, binary_masks)

        # Get LR feature
        lr_feats = self._get_lr_feature(img)

        # For-loop the augmenation process
        for aug_index in range(self.n_augs):
            # Augmentation loss
            aug_loss_dict = self._forward_augmentation_loss(
                lr_feats,
                None,  # let the function compute hr_feats
                img,
                guidance_img,
                binary_masks,
                aug_index,
            )
            total_loss = total_loss + aug_loss_dict.augmentation_loss / self.n_augs

        # SAM adjustment loss
        if self._sam_model_online or binary_masks is None:
            assert self.sam_model is not None, (
                "SAM model must be provided if binary masks are not given."
            )
            binary_masks = self.sam_model.generate_masks(img).detach()
            binary_masks = F.interpolate(
                binary_masks,
                size=(self.resampled_img_size, self.resampled_img_size),
                mode="nearest",
            )

        sam_loss_dict = self._forward_sam_adjust_loss(
            lr_feats,
            aug_loss_dict.hr_feats,
            binary_masks,
            img,
            guidance_img,
        )
        total_loss = total_loss + sam_loss_dict.sam_adjust_loss

        # Merge all losses and feature parts
        ret = LoftUpStage1Output(
            total_loss=total_loss,
            # Augmentation losses
            augmentation_loss=aug_loss_dict.augmentation_loss,
            recon_loss=aug_loss_dict.recon_loss,
            tv_loss=aug_loss_dict.tv_loss,
            entropy_loss=aug_loss_dict.entropy_loss,
            # SAM adjustment losses
            sam_adjust_loss=sam_loss_dict.sam_adjust_loss,
            mask_recon_loss=sam_loss_dict.mask_recon_loss,
            mask_reg_loss=sam_loss_dict.mask_reg_loss,
            # Features
            adjusted_up_lr_feats=sam_loss_dict.adjusted_up_lr_feats,
            hr_feats=aug_loss_dict.hr_feats,
            lr_aug_feats=aug_loss_dict.lr_aug_feats,
            hr_aug_feats=aug_loss_dict.hr_aug_feats,
            # Projected features
            proj_hr_aug_feats=aug_loss_dict.proj_hr_aug_feats,
            proj_downsampled_hr_feats=aug_loss_dict.proj_downsampled_hr_feats,
            proj_lr_aug_feats=aug_loss_dict.proj_lr_aug_feats,
        )

        return ret
