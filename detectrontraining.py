import os
import cv2
import pandas as pd
from tqdm import tqdm

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo

def load_cells_from_csv(images_dir, csv_path, *, xy_is_center=True):
    df = pd.read_csv(csv_path)
    grouped = df.groupby("image_filename")

    dataset_dicts = []
    for img_id, (img_name, rows) in enumerate(grouped):   # <-- img_id is int
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        record = {
            "file_name": img_path,
            "image_id": img_id,     # ✅ MUST BE INT for COCOEvaluator
            "height": h,
            "width": w,
            "annotations": []
        }

        for _, r in rows.iterrows():
            x = float(r["x"])
            y = float(r["y"])
            bw = float(r["width"])
            bh = float(r["height"])

            # If x,y are centers
            if xy_is_center:
                x0 = x - bw / 2.0
                y0 = y - bh / 2.0
                x1 = x + bw / 2.0
                y1 = y + bh / 2.0
            else:
                # If x,y are top-left
                x0 = x
                y0 = y
                x1 = x + bw
                y1 = y + bh

            # clip to image boundaries
            x0 = max(0.0, min(x0, w - 1.0))
            y0 = max(0.0, min(y0, h - 1.0))
            x1 = max(0.0, min(x1, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))

            record["annotations"].append({
                "bbox": [x0, y0, x1, y1],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            })

        dataset_dicts.append(record)

    return dataset_dicts


def register_riva_dataset(root_dir):
    # root_dir = "riva-cervical-cytology-challenge-track-b-isbi_B"
    ann_dir = os.path.join(root_dir, "annotations", "annotations")
    img_dir = os.path.join(root_dir, "images", "images")

    train_images = os.path.join(img_dir, "train")
    val_images   = os.path.join(img_dir, "val")

    train_csv = os.path.join(ann_dir, "train.csv")
    val_csv   = os.path.join(ann_dir, "val.csv")

    DatasetCatalog.register(
        "cells_train",
        lambda: load_cells_from_csv(train_images, train_csv, xy_is_center=True)
    )
    DatasetCatalog.register(
        "cells_val",
        lambda: load_cells_from_csv(val_images, val_csv, xy_is_center=True)
    )

    MetadataCatalog.get("cells_train").set(thing_classes=["CELL"])
    MetadataCatalog.get("cells_val").set(thing_classes=["CELL"])

root_dir = "riva-cervical-cytology-challenge-track-b-isbi_B"  # <-- change if needed
register_riva_dataset(root_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("cells_train",)
cfg.DATASETS.TEST  = ("cells_val",)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
cfg.INPUT.MAX_SIZE_TRAIN = 1024
cfg.INPUT.MIN_SIZE_TEST  = 1024
cfg.INPUT.MAX_SIZE_TEST  = 1024

cfg.DATALOADER.NUM_WORKERS = 2

# Start safe; increase if VRAM allows
# Default values (will be overridden by user input)
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (7000, 9000)
cfg.SOLVER.WARMUP_ITERS = 200
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.TEST.EVAL_PERIOD = 500

# Helpful for many objects per image
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 1000

from detectron2.utils.events import CommonMetricPrinter, JSONWriter


class TrainingProgressHook(HookBase):
    """Hook to display training progress with tqdm"""
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.pbar = None
        self.current_iter = 0
        
    def before_train(self):
        self.pbar = tqdm(total=self.max_iter, desc="Training", unit="iter", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.current_iter = 0
        
    def after_step(self):
        self.current_iter += 1
        if self.pbar is not None:
            self.pbar.update(1)
            # Update description with current iteration
            self.pbar.set_description(f"Training (iter {self.current_iter}/{self.max_iter})")
            
    def after_train(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class EvaluationProgressHook(HookBase):
    """Hook to display evaluation progress"""
    def before_eval(self):
        print("\n" + "="*60)
        print("Starting Evaluation...")
        print("="*60)
        
    def after_eval(self):
        print("="*60)
        print("Evaluation Completed!")
        print("="*60 + "\n")


class NoTensorboardTrainer(DefaultTrainer):
    # keep your no-tensorboard writers
    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]

    # ✅ THIS enables evaluation
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        # Add training progress hook
        hooks.insert(-1, TrainingProgressHook(self.cfg.SOLVER.MAX_ITER))
        # Add evaluation progress hook
        hooks.insert(-1, EvaluationProgressHook())
        return hooks


if __name__ == '__main__':
    cfg.OUTPUT_DIR = "./output_cells_frcnn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.DEVICE = "cpu"

    # Interactive input for training parameters
    print("\n" + "="*60)
    print("Detectron2 Training Configuration")
    print("="*60)

    try:
        lr_input = input(f"\nEnter learning rate (default: {cfg.SOLVER.BASE_LR}): ").strip()
        if lr_input:
            cfg.SOLVER.BASE_LR = float(lr_input)
        
        max_iter_input = input(f"Enter max iterations (default: {cfg.SOLVER.MAX_ITER}): ").strip()
        if max_iter_input:
            cfg.SOLVER.MAX_ITER = int(max_iter_input)
        
        eval_period_input = input(f"Enter evaluation period (default: {cfg.TEST.EVAL_PERIOD}): ").strip()
        if eval_period_input:
            cfg.TEST.EVAL_PERIOD = int(eval_period_input)
        
        ims_per_batch_input = input(f"Enter images per batch (default: {cfg.SOLVER.IMS_PER_BATCH}): ").strip()
        if ims_per_batch_input:
            cfg.SOLVER.IMS_PER_BATCH = int(ims_per_batch_input)
    except ValueError as e:
        print(f"\nError: Invalid input. Using default values. Error: {e}")
    except KeyboardInterrupt:
        print("\n\nTraining cancelled by user.")
        exit(0)

    # Update steps based on max_iter if needed
    if cfg.SOLVER.MAX_ITER > 1000:
        cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.7), int(cfg.SOLVER.MAX_ITER * 0.9))
    else:
        cfg.SOLVER.STEPS = []

    print("\n" + "="*60)
    print("Training Configuration Summary:")
    print("="*60)
    print(f"Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f"Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Evaluation Period: {cfg.TEST.EVAL_PERIOD}")
    print(f"Images per Batch: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Output Directory: {cfg.OUTPUT_DIR}")
    print("="*60)
    print("\nStarting training...\n")

    trainer = NoTensorboardTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()