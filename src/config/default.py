# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:24:15 2023

@author: knight
"""

from yacs.config import CfgNode as CN

_CN = CN()

##############  ↓  LGFCTR Pipeline  ↓  ##############
_CN.LGFCTR = CN()
_CN.LGFCTR.RESOLUTION = (8, 2)  # (8,2)
_CN.LGFCTR.ATTENTION = 'linear'
_CN.LGFCTR.FF_TYPE = 'mix'  # [cnn, mlp, mix]
_CN.LGFCTR.POSITIONAL_MODE = None  # ['add', 'cat', None]
_CN.LGFCTR.POSITIONAL_POS = 'resolution'  # [resolution, layer]
_CN.LGFCTR.NHEAD = 4
_CN.LGFCTR.TRANSFORMER_TYPE = 'multi'  # ['default', 'semi', 'pure', 'multi']

# 1. LGFCTR-encoder config
_CN.LGFCTR.ENCODER = CN()
_CN.LGFCTR.ENCODER.STEM_DIMS = [32] * 3
_CN.LGFCTR.ENCODER.STEM_DIMS2 = [64] * 3
_CN.LGFCTR.ENCODER.LAYER_NAMES = ['self'] * 3 + ['cross'] * 0
_CN.LGFCTR.ENCODER.LAYER_DIMS = [128, 192, 256]

# 2. LGFCTR-decoder config
_CN.LGFCTR.DECODER = CN()
_CN.LGFCTR.DECODER.LAYER_NAMES = ['self'] * 0 + ['cross'] * 3
_CN.LGFCTR.DECODER.LAYER_DIMS = [256] * 3
_CN.LGFCTR.DECODER.ENCODER_LAYER_DIMS_INV = _CN.LGFCTR.ENCODER.LAYER_DIMS[-2::-1]
_CN.LGFCTR.DECODER.IS_OUT_CONVS = False

# 3. Coarse-Matching config
_CN.LGFCTR.MATCH_COARSE = CN()
_CN.LGFCTR.MATCH_COARSE.THR = 0.2
_CN.LGFCTR.MATCH_COARSE.BORDER_RM = 2
_CN.LGFCTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LGFCTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.LGFCTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.LGFCTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
_CN.LGFCTR.MATCH_COARSE.SPARSE_SPVS = True

# 4. fine preprocess and matching config
_CN.LGFCTR.MATCH_FINE = CN()
_CN.LGFCTR.MATCH_FINE.FINE_WINDOW_SIZE = 5  # 5
_CN.LGFCTR.MATCH_FINE.FINE_CONCAT_COARSE_FEAT = True
_CN.LGFCTR.MATCH_FINE.D_MODEL_C = _CN.LGFCTR.DECODER.LAYER_DIMS[0]
_CN.LGFCTR.MATCH_FINE.D_MODEL_F = _CN.LGFCTR.DECODER.LAYER_DIMS[-1]
_CN.LGFCTR.MATCH_FINE.MATCHING_TYPE = 'feat_regressor_attn'  # options: ['feat_reg_cls', 'sim_expectation', 'feat_regressor', 'feat_regressor_attn']
_CN.LGFCTR.MATCH_FINE.COMPACT_DIMS = [256, 128]  # For feat_regressor
_CN.LGFCTR.MATCH_FINE.MLP_DIMS = [128, 32, 2]  # For feat_regressor, the last item must be 2
_CN.LGFCTR.MATCH_FINE.NHEAD = 8
_CN.LGFCTR.MATCH_FINE.FF_TYPE = 'mlp'

# 5. LGFCTR Losses
# -- # coarse-level
_CN.LGFCTR.LOSS = CN()
_CN.LGFCTR.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.LGFCTR.LOSS.COARSE_WEIGHT = 1.0
# -- - -- # focal loss (coarse)
_CN.LGFCTR.LOSS.FOCAL_ALPHA = 0.25
_CN.LGFCTR.LOSS.FOCAL_GAMMA = 2.0
_CN.LGFCTR.LOSS.POS_WEIGHT = 1.0
_CN.LGFCTR.LOSS.NEG_WEIGHT = 1.0

# -- # fine-level
_CN.LGFCTR.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2', 'l2 + cls']
_CN.LGFCTR.LOSS.FINE_WEIGHT = 1.0
_CN.LGFCTR.LOSS.FINE_WEIGHT_CLS = 1.0
_CN.LGFCTR.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)
_CN.LGFCTR.LOSS.POS_WEIGHT_CLS = 1.0  # for classification of fine-level
_CN.LGFCTR.LOSS.NEG_WEIGHT_CLS = 1.0

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None  # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None  # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 800  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_RESIZE_VAL = 1200
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 8e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.1
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'  # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = False
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 4  # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 100
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
