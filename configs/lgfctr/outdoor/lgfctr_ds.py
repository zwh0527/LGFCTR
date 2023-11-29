from src.config.default import _CN as cfg

cfg.LGFCTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.LGFCTR.MATCH_COARSE.SPARSE_SPVS = True

cfg.TRAINER.WARMUP_STEP = 368 * 100 * 3 // 4 // (64//4)  # 3 epochs

cfg.LGFCTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.15
