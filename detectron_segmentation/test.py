from detectron2.engine import DefaultPredictor
import os
import pickle

from utils import *


def main():
    # use different pickle files for different models
    cfg_save_path = "CI_cfg.pickle"
    with open(cfg_save_path, 'rb') as f:
        cfg = pickle.load(f)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4

    predictor = DefaultPredictor(cfg)

    ### CHANGE IMAGE PATH HERE ####
    image_path = "/Users/tannerwatts/Desktop/serotonin-segmentation/prediction_imgs/m5_1_6_3(1).jpg"

    ### CHANGE DIR PATH HERE ###
    dir_path = "/Users/tannerwatts/Desktop/serotonin-segmentation/prediction_imgs/"

    # loop through each image in a directory.
    image_list = []
    index = 1
    for filename in [file for file in os.listdir(dir_path) if file.endswith('.jpg')]:
        image_pth = os.path.join(dir_path, filename)
        # suround with loop
        print("IMAGE #: " + str(index))
        print(filename)
        image_list.append(predict_image(image_pth, predictor))
        index = index + 1

    for v in image_list:
        plt.figure(figsize=(14, 10))
        plt.imshow(v.get_image())
        plt.show(block=False)

    plt.show()



if __name__ == '__main__':
    main()


