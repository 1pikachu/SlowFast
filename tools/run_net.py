#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize
import torch


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)

        # OOB
        cfg.TRAIN.ENABLE = False
        cfg.LOG_MODEL_INFO = False
        cfg.NUM_GPUS = 0
        cfg.DATA.DECODING_BACKEND = 'pyav'
        cfg.DATA.PATH_PREFIX = args.dataset_dir
        cfg.DATA.PATH_LABEL_SEPARATOR = ','
        cfg.TEST.BATCH_SIZE = args.batch_size

        if args.device == "xpu" and args.ipex:
            import intel_extension_for_pytorch
        elif args.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False

        cfg.datatype = [torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float]
        # Perform training.
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)

        # Perform multi-clip testing.
        if cfg.TEST.ENABLE:
            if args.device == "cuda":
                with torch.cuda.amp.autocast(enabled=True, dtype=cfg.datatype[0]):
                    launch_job(cfg=cfg, init_method=args.init_method, func=test)
            elif args.device == "xpu":
                with torch.xpu.amp.autocast(enabled=True, dtype=cfg.datatype[0]):
                    launch_job(cfg=cfg, init_method=args.init_method, func=test)
            else:
                with torch.cpu.amp.autocast(enabled=True, dtype=cfg.datatype[0]):
                    launch_job(cfg=cfg, init_method=args.init_method, func=test)


        # Perform model visualization.
        if cfg.TENSORBOARD.ENABLE and (
            cfg.TENSORBOARD.MODEL_VIS.ENABLE
            or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
        ):
            launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

        # Run demo.
        if cfg.DEMO.ENABLE:
            demo(cfg)


if __name__ == "__main__":
    main()
