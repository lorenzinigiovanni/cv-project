import cv2
import os
import os.path as osp
from pathlib import Path
import sys
import json
from math import ceil
import argparse
import shutil

# import some common libraries
import numpy as np
import cv2
import tqdm
import glob

ROOT_PATH = osp.dirname(os.path.abspath(__file__))
INPUT_PATH = ROOT_PATH+'/../input'
OUT_PATH = ROOT_PATH+'/../ouput'
POINTREND_ROOT_PATH = osp.join(ROOT_PATH, "detectron2", "projects", "PointRend")

os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

try:
    import detectron2
except:
    print(
        "Please install Detectron2 by selecting the right version",
        "from https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md",
    )
# import PointRend project

##### use detectron for segmentation
if not os.path.exists(POINTREND_ROOT_PATH):
    import urllib.request, zipfile

    print("Downloading minimal PointRend source package")
    zipfile_name = "pointrend_min.zip"
    urllib.request.urlretrieve(
        "https://alexyu.net/data/pointrend_min.zip", zipfile_name
    )
    with zipfile.ZipFile(zipfile_name) as zipfile:
        zipfile.extractall(ROOT_PATH)
    os.remove(zipfile_name)

sys.path.insert(0, POINTREND_ROOT_PATH)
import point_rend

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


parser = argparse.ArgumentParser()
parser.add_argument(
    "--coco_class",
    type=int,
    default=[56],  # 0 person,   2 car,    56 chair
    help="COCO class wanted (0 = human, 2 = car)",
)
parser.add_argument(
    "--size",
    "-s",
    type=int,
    default=256,
    help="output image side length (will be square)",
)
parser.add_argument(
    "--scale",
    "-S",
    type=float,
    default=2.5,
    help="bbox scaling rel minor axis of fitted ellipse. "
    + "Will take max radius from this and major_scale.",
)
parser.add_argument(
    "--major_scale",
    "-M",
    type=float,
    default=.8,
    help="bbox scaling rel major axis of fitted ellipse. "
    + "Will take max radius from this and major_scale.",
)
parser.add_argument(
    "--const_border",
    action="store_true",
    help="constant white border instead of replicate pad",
)
args = parser.parse_args()

def _crop_image(img, rect, const_border=False, value=0):
    """
    Image cropping helper
    """
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1] - (x + w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0] - (y + h)) if y + h >= img.shape[0] else 0

    color = [value] * img.shape[2] if const_border else None
    new_img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT if const_border else cv2.BORDER_REPLICATE,
        value=color,
    )
    if len(new_img.shape) == 2:
        new_img = new_img[..., None]

    x = x + left
    y = y + top

    return new_img[y : (y + h), x : (x + w), :]


def _is_image_path(f):
    return (
        f.endswith(".jpg")
        or f.endswith(".jpeg")
        #or f.endswith(".png")
        or f.endswith(".bmp")
        or f.endswith(".tiff")
        or f.endswith(".gif")
    )


class PointRendWrapper:
    def __init__(self, filter_class=-1):
        """
        :param filter_class output only intances of filter_class (-1 to disable). Note: class 0 is person.
        """
        if isinstance(filter_class, int):
            filter_class = [filter_class]
        self.filter_class = filter_class
        self.coco_metadata = MetadataCatalog.get("coco_2017_val")
        self.cfg = get_cfg()

        # Add PointRend-specific config

        point_rend.add_pointrend_config(self.cfg)

        # Load a config from file
        self.cfg.merge_from_file(
            os.path.join(
                POINTREND_ROOT_PATH,
                "configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml",
            )
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl"
        self.predictor = DefaultPredictor(self.cfg)

    def segment(self, im, out_name="", visualize=False):
        """
        Run PointRend
        :param out_name if set, writes segments B&W mask to this image file
        :param visualize if set, and out_name is set, outputs visualization rater than B&W mask
        """
        outputs = self.predictor(im)

        predictions = outputs["instances"]
        if self.filter_class != -1:
            for cl in self.filter_class:
                insts = predictions[predictions.pred_classes == cl]  # 0 is person
                if(len(insts.pred_masks) != 0):
                    break
        if visualize:
            v = Visualizer(
                im[:, :, ::-1],
                self.coco_metadata,
                scale=1.2,
                instance_mode=ColorMode.IMAGE_BW,
            )

            point_rend_result = v.draw_instance_predictions(insts.to("cpu")).get_image()
            if out_name:
                cv2.imwrite(out_name + ".png", point_rend_result[:, :, ::-1])
            return point_rend_result[:, :, ::-1]
        else:
            im_names = []
            masks = []
            for i in range(len(insts)):
                mask = insts[i].pred_masks.to("cpu").permute(
                    1, 2, 0
                ).numpy() * np.uint8(255)
                if out_name:
                    im_name = out_name
                    if i:
                        im_name += "_" + str(i) + ".png"
                    else:
                        im_name += ".png"
                    im_names.append(im_name)
                    cv2.imwrite(im_name, mask)
                masks.append(mask)
            if out_name:
                with open(out_name + ".json", "w") as fp:
                    json.dump({"files": im_names}, fp)
            return masks


pointrend = PointRendWrapper(args.coco_class)

input_images = glob.glob(os.path.join(INPUT_PATH, "*"))

os.makedirs(OUT_PATH, exist_ok=True)

for num, image_path in enumerate(tqdm.tqdm(input_images)):
    print(image_path)
    im = cv2.imread(image_path)
    img_no_ext = os.path.split(os.path.splitext(image_path)[0])[1]
    masks = pointrend.segment(im)

    for i, mask in enumerate(masks):
        mask = masks[i]

        mask = mask.astype(np.float32) / 255.0
        masked = im.astype(np.float32) * mask + 255 * (1.0 - mask)
        masked = masked.astype(np.uint8)

        out_masked_path = os.path.join(OUT_PATH, img_no_ext + "." + str(i) + ".png")
        print(out_masked_path)
        cv2.imwrite(out_masked_path, masked)
