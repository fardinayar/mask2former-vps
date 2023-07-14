

# Detectron2 evaluator class for cityscapes video panoptic segmentaion
# Modified from detectron2/evaluation/panoptic_evaluation.py
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator
from .cityscapes_vps_eval import vpq_compute

logger = logging.getLogger(__name__)

class CityscapesVPSEvaluator(DatasetEvaluator):
    """
    Evaluate Cityscapes-vps dataset
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )
                
    def vpq_main(self, submit_dir, truth_dir, pan_gt_json_file):
        output_dir = submit_dir

        pan_pred_json_file = os.path.join(submit_dir, 'predictions.json')
        with open(pan_pred_json_file, 'r') as f:
            pred_jsons = json.load(f)
            
        with open(pan_gt_json_file, 'r') as f:
            gt_jsons = json.load(f)

        categories = gt_jsons['categories']
        categories = {el['id']: el for el in categories}

        gt_pans = []
        files = [item['file_name'].replace('_newImg8bit.png','_final_mask.png').replace('_leftImg8bit.png','_gtFine_color.png') for item in gt_jsons['images']]
        files.sort()
        for idx, file in enumerate(files):
            image = np.array(Image.open(os.path.join(truth_dir, file)))
            gt_pans.append(image)

        pred_pans = []
        files = [item['id']+'.png' for item in gt_jsons['images']]
        for idx, file in enumerate(files):
            image = np.array(Image.open(os.path.join(submit_dir, file)))
            pred_pans.append(image)
        assert len(gt_pans) == len(pred_pans), "number of prediction does not match with the groud truth."

        gt_image_jsons = gt_jsons['images']
        gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
        nframes_per_video = 6
        vid_num = len(gt_jsons)//nframes_per_video # 600//6 = 100

        gt_pred_all = list(zip(gt_jsons, pred_jsons, gt_pans, pred_pans, gt_image_jsons))
        gt_pred_split = np.array_split(gt_pred_all, vid_num)

        vpq_all, vpq_thing, vpq_stuff = [], [], []

        # for k in [0,5,10,15] --> num_frames_w_gt [1,2,3,4]
        for nframes in [1,2,3,4]:
            gt_pred_split_ = copy.deepcopy(gt_pred_split)
            vpq_all_, vpq_thing_, vpq_stuff_ = vpq_compute(
                    gt_pred_split_, categories, nframes, output_dir)
            del gt_pred_split_
            print(vpq_all_, vpq_thing_, vpq_stuff_)
            vpq_all.append(vpq_all_)
            vpq_thing.append(vpq_thing_)
            vpq_stuff.append(vpq_stuff_)

        ret = {}
        ret["vpq_all"] = (sum(vpq_all)/len(vpq_all))
        ret["vpq_thing"] = (sum(vpq_thing)/len(vpq_thing))
        ret["vpq_stuff"] = (sum(vpq_stuff)/len(vpq_stuff))
        return ret

        

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))
                


            with contextlib.redirect_stdout(io.StringIO()):
                vpq_res = vpq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )


        results = OrderedDict({"panoptic_seg": vpq_res})
        _print_panoptic_results(vpq_res)

        return results


def _print_panoptic_results(pq_res):
    print("******************")
    print(pq_res)
