# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_fastinst_config(cfg):
    """
    Add config for FastInst.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "fastinst_instance"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # add rand augmentation
    cfg.INPUT.CROP.RESIZE = CN()
    cfg.INPUT.CROP.RESIZE.ENABLED = False
    cfg.INPUT.CROP.RESIZE.MIN_SIZE = (800,)
    cfg.INPUT.CROP.RESIZE.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # fastinst model config
    cfg.MODEL.FASTINST = CN()

    # loss
    cfg.MODEL.FASTINST.DEEP_SUPERVISION = True
    cfg.MODEL.FASTINST.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.FASTINST.CLASS_WEIGHT = 2.0
    cfg.MODEL.FASTINST.DICE_WEIGHT = 5.0
    cfg.MODEL.FASTINST.MASK_WEIGHT = 5.0
    cfg.MODEL.FASTINST.LOCATION_WEIGHT = 1e3
    cfg.MODEL.FASTINST.PROPOSAL_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.FASTINST.NHEADS = 8
    cfg.MODEL.FASTINST.DROPOUT = 0.
    cfg.MODEL.FASTINST.DIM_FEEDFORWARD = 1024
    cfg.MODEL.FASTINST.DEC_LAYERS = 10
    cfg.MODEL.FASTINST.PRE_NORM = False
    #
    cfg.MODEL.FASTINST.HIDDEN_DIM = 256
    cfg.MODEL.FASTINST.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.FASTINST.NUM_AUX_QUERIES = 8

    # fastinst inference config
    cfg.MODEL.FASTINST.TEST = CN()
    cfg.MODEL.FASTINST.TEST.SEMANTIC_ON = False
    cfg.MODEL.FASTINST.TEST.INSTANCE_ON = True
    cfg.MODEL.FASTINST.TEST.PANOPTIC_ON = False
    cfg.MODEL.FASTINST.TEST.OBJECT_MASK_THRESHOLD = 0.8
    cfg.MODEL.FASTINST.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.FASTINST.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.FASTINST.SIZE_DIVISIBILITY = -1

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "FastInstEncoderV1"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.FASTINST.TRANSFORMER_DECODER_NAME = "FastInstDecoderV4"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.FASTINST.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.FASTINST.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.FASTINST.IMPORTANCE_SAMPLE_RATIO = 0.75
