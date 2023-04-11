# Modified from https://github.com/facebookresearch/Mask2Former
from typing import Dict

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from torch import nn

from ..pixel_decoder.fastinst_encoder import build_pixel_decoder
from ..transformer_decoder.utils import build_transformer_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class FastInstHead(nn.Module):

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            num_classes: int,
            pixel_decoder: nn.Module,
            # extra parameters
            transformer_predictor: nn.Module
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "transformer_predictor": build_transformer_decoder(
                cfg,
                cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                input_shape=input_shape
            ),
        }

    def forward(self, features, targets=None):
        return self.layers(features, targets)

    def layers(self, features, targets=None):
        mask_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, targets)
        return predictions
