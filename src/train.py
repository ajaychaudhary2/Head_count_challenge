import os
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import get_config_file, get_checkpoint_url

# === Setup logging ===
setup_logger()

# === Register your datasets ===
register_coco_instances(
    "face_train2", {},
    "/teamspace/studios/this_studio/Head_counter_CNN/train_data/updated_annotations.json",
    "/teamspace/studios/this_studio/Head_counter_CNN/train_data/train_split"
)

register_coco_instances(
    "face_val2", {},
    "/teamspace/studios/this_studio/Head_counter_CNN/train_data/val_annotations.json",
    "/teamspace/studios/this_studio/Head_counter_CNN/train_data/val_split"
)

# === Check registration ===
from detectron2.data import DatasetCatalog, MetadataCatalog

print(DatasetCatalog.get("face_train2"))
print(DatasetCatalog.get("face_val2"))
print(MetadataCatalog.get("face_train2"))
print(MetadataCatalog.get("face_val2"))

# === Build configuration ===
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("face_train2",)
cfg.DATASETS.TEST = ("face_val2",)
cfg.DATALOADER.NUM_WORKERS = 2

# === Solver settings ===
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 8000
cfg.SOLVER.STEPS = (5000, 6000)
cfg.SOLVER.CHECKPOINT_PERIOD = 500

# === ROI head settings ===
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only 'head'

# === Input resizing ===
cfg.INPUT.MIN_SIZE_TRAIN = (800,)
cfg.INPUT.MAX_SIZE_TRAIN = 1333

# === Adjust anchors for small objects ===
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] * len(cfg.MODEL.ANCHOR_GENERATOR.SIZES)

# === Output Directory ===
cfg.OUTPUT_DIR = "/teamspace/studios/this_studio/Head_counter_CNN/output_face"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# === Save Config for future reproducibility ===
with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())

# === Start Training ===
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
