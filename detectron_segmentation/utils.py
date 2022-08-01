from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

import random
import cv2
import matplotlib.pyplot as plt


def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
    # for s in dataset_custom:
        img = cv2.imread(s["file_name"])
        # swap the channels
        v = Visualizer(img[:,:,::-1], metadata = dataset_custom_metadata, scale= 0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    '''
    This method determines the configuration for which model will be used for training.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)

    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg

def predict_image(image_path,predictor):
    meta = MetadataCatalog.get("cells_train")
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1], metadata=meta, scale= 1.8, instance_mode= ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cfos_count = 0
    instances = outputs["instances"].pred_classes
    instance_size = len(instances)
    print("TOTAL INSTANCES: " + str(instance_size))
    for inst in instances:
        if inst == 0:
            cfos_count = cfos_count + 1
    print("NUMBER CFOS: " + str(cfos_count))
    print("NUMBER CFOS NEGATIVE: " + str(instance_size - cfos_count))
    print()
    return v



