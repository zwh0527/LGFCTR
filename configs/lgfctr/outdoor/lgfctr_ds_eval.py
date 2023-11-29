from src.config.default import _CN as cfg

cfg.LGFCTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.LGFCTR.MATCH_COARSE.SPARSE_SPVS = True
cfg.LGFCTR.MATCH_COARSE.THR = 0.2

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.1
