#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys
import torch

import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_files",
        help="Path to the config files",
        default=["configs/Kinetics/SLOWFAST_4x16_R50.yaml"],
        nargs="+",
    )
    parser.add_argument(
        "--opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float16"
    )
    parser.add_argument(
        "--channels_last",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=0
    )
    parser.add_argument('--profile', action='store_true', help='profile')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nvFuser')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--dataset_dir', type=str, default='/home2/pytorch-broad-models/pytorchvideo/tiny-Kinetics-400', help='dataset_dir')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--compile', action='store_true', default=False, help='compile model')
    parser.add_argument('--backend', default="inductor", type=str, help='backend')
    parser.add_argument('--ipex', default=False, action='store_true', help="ipex is not enabled now")
    parser.add_argument("--xpu_fallback", default=False, action="store_true", help="Whether to set xpu fallback")

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # OOB
    cfg.precision = args.precision
    cfg.channels_last = args.channels_last
    cfg.num_iter = args.num_iter
    cfg.num_warmup = args.num_warmup
    cfg.profile = args.profile
    cfg.jit = args.jit
    cfg.nv_fuser = args.nv_fuser
    cfg.device = args.device
    cfg.dataset_dir = args.dataset_dir
    cfg.batch_size = args.batch_size
    cfg.compile = args.compile
    cfg.backend = args.backend
    cfg.ipex = args.ipex

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg
