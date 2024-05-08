#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import time

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    def convert_device(inputs, video_idx=None, meta=None, device="cuda"):
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(cfg.device)
        else:
            inputs = inputs.to(cfg.device)
        if video_idx:
            video_idx = video_idx.to(cfg.device)
        if meta:
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].to(cfg.device)
                else:
                    meta[key] = val.to(cfg.device)
        return inputs, video_idx, meta


    if cfg.channels_last and cfg.device != "xpu":
        model = model.to(memory_format=torch.channels_last_3d)
        print("---- Use 3D NHWC model")
    if cfg.nv_fuser:
        fuser_mode = "fuser2"
    else:
        fuser_mode = "none"

    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    model = model.to(cfg.device)

    if cfg.device == "xpu":
        model = torch.xpu.optimize(model=model, dtype=cfg.datatype[0])
        print("---- xpu optimize")
    if cfg.jit:
        try:
            inputs = iter(test_loader).__next__()[0]
            inputs = convert_device(inputs, device=cfg.device)[0]
            inputs = [inputs]
            model = torch.jit.trace(model, inputs, check_trace=False)
            print("---- with JIT trace")
        except (RuntimeError, TypeError) as e:
            print("---- JIT trace disable.")
            print("failed to use PyTorch jit mode due to: ", e)
    if cfg.compile:
        print("----enable compiler")
        pipe = torch.compile(pipe, backend=cfg.backend, options={"freezing": True})

    total_time = 0.0
    total_sample = 0
    profile_iter = min(cfg.num_iter+cfg.num_warmup, len(test_loader)) // 2
    for cur_iter, (inputs, labels, video_idx, time_, meta) in enumerate(
        test_loader
    ):
        if cur_iter > cfg.num_iter and cfg.num_iter > 1: break
        if cfg.channels_last and cfg.device != "xpu":
            try:
                inputs = [i.to(memory_format=torch.channels_last_3d) for i in inputs]
            except:
                pass

        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time_)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time_)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        elif cfg.profile and cfg.device == "xpu":
            # Perform the forward pass.
            with torch.autograd.profiler_legacy.profile(enabled=True, use_xpu=True, record_shapes=False) as prof:
                tic = time.time()
                # Transfer the data to the current GPU device.
                inputs, video_idx, meta = convert_device(inputs, video_idx, meta, cfg.device)
                preds = model(inputs)
                torch.xpu.synchronize()
                toc = time.time()
            if cfg.profile and cur_iter == profile_iter:
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
                torch.save(prof.table(sort_by="id", row_limit=100000),
                    timeline_dir+'profile_detail_withId.pt')
                prof.export_chrome_trace(timeline_dir+"trace.json")
            elapsed = toc - tic
            print("Iteration: {}, inference time: {} sec.".format(cur_iter, elapsed), flush=True)
            if cur_iter >= cfg.num_warmup:
                total_time += elapsed
                total_sample += cfg.TEST.BATCH_SIZE
        elif cfg.profile and cfg.device != "xpu":
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                record_shapes=True,
            ) as p:
                tic = time.time()
                # Transfer the data to the current GPU device.
                inputs, video_idx, meta = convert_device(inputs, video_idx, meta, cfg.device)
                if cfg.device == "cuda":
                    with torch.jit.fuser(fuser_mode):
                        preds = model(inputs)
                    torch.cuda.synchronize()
                else:
                    preds = model(inputs)
                toc = time.time()
            elapsed = toc - tic
            print("Iteration: {}, inference time: {} sec.".format(cur_iter, elapsed), flush=True)
            if cur_iter >= cfg.num_warmup:
                total_time += elapsed
                total_sample += cfg.TEST.BATCH_SIZE
            if cur_iter == profile_iter:
                output = p.key_averages().table(sort_by="self_cpu_time_total")
                print(output)
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                            str(cur_iter) + '-' + str(os.getpid()) + '.json'
                p.export_chrome_trace(timeline_file)
        else:
            tic = time.time()
            # Transfer the data to the current GPU device.
            inputs, video_idx, meta = convert_device(inputs, video_idx, meta, cfg.device)
            # Perform the forward pass.
            if cfg.device == "cuda":
                with torch.jit.fuser(fuser_mode):
                    preds = model(inputs)
                torch.cuda.synchronize()
            elif cfg.device == "xpu":
                preds = model(inputs)
                torch.xpu.synchronize()
            else:
                preds = model(inputs)
            toc = time.time()
            elapsed = toc - tic
            print("Iteration: {}, inference time: {} sec.".format(cur_iter, elapsed), flush=True)
            if cur_iter >= cfg.num_warmup:
                total_time += elapsed
                total_sample += cfg.TEST.BATCH_SIZE

        # Gather all the predictions across all the devices to perform ensemble.
        #if cfg.NUM_GPUS > 1:
        #    preds, labels, video_idx = du.all_gather(
        #        [preds, labels, video_idx]
        #    )
        #if cfg.NUM_GPUS:
        #    preds = preds.cpu()
        #    labels = labels.cpu()
        #    video_idx = video_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        #test_meter.update_stats(
        #    preds.detach(), labels.detach(), video_idx.detach()
        #)
        #test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()
    print("\n", "-"*20, "Summary", "-"*20)
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))
    exit(0)

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    #du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    out_str_prefix = "lin" if cfg.MODEL.DETACH_FINAL_FC else ""

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        train_loader = loader.construct_loader(cfg, "train")
        out_str_prefix = "knn"
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES
            if not cfg.TASK == "ssl"
            else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
    result_string = (
        "_a{}{}{} Top1 Acc: {} Top5 Acc: {} MEM: {:.2f} dataset: {}{}"
        "".format(
            out_str_prefix,
            cfg.TEST.DATASET[0],
            test_meter.stats["top1_acc"],
            test_meter.stats["top1_acc"],
            test_meter.stats["top5_acc"],
            misc.gpu_mem_usage(),
            cfg.TEST.DATASET[0],
            cfg.MODEL.NUM_CLASSES,
        )
    )
    logger.info("testing done: {}".format(result_string))

    return result_string
