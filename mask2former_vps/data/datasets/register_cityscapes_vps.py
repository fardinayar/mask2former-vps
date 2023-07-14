# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager
import glob
from collections import defaultdict
logger = logging.getLogger(__name__)


_CITYSCAPES_VPS_PANOPTIC_SPLITS = {
    "cityscapes_vps_panoptic_train": (
        "train/img",
        "train/panoptic_video",
        "panoptic_gt_train_city_vps.json",
    ),
    "cityscapes_vps_panoptic_val": (
        "val/img",
        "val/panoptic_video",
        "panoptic_gt_val_city_vps.json",
    ),
    # "cityscapes_fine_panoptic_test": not supported yet
}


def get_cityscapes_vps_video_dict(json_info):
    if type(json_info) == str:
        with open(json_info) as f:
            json_info = json.load(f)
            
    frame_list = [x['id'] for x in json_info['images']]
    videos_to_frames = defaultdict(list)
    for file in frame_list:
        videos_to_frames[file[:4]].append(file)
    return videos_to_frames


def get_cityscapes_vps_dicts(image_dir, gt_dir, json_info, meta):
    
    if type(json_info) == str:
        with open(json_info) as f:
            json_info = json.load(f)
            
    videos_to_frame_ids = get_cityscapes_vps_video_dict(json_info)
    image_ids_to_image = {x['id']: x for x in json_info['images']}
    image_ids_to_annos = {x['image_id']: x for x in json_info['annotations']}
    
    logger.info(f"{len(videos_to_frame_ids)} videos found in '{gt_dir}'.")
    videos_list = []
    
    for video in videos_to_frame_ids.keys():
        video_frame_ids = videos_to_frame_ids[video]
        video_dict = {}
        video_dict['file_names'] = []
        video_dict['image_ids'] = []
        video_dict['pan_seg_files'] = []
        video_dict['sem_seg_files'] = []
        video_dict['segments_info'] = []
        video_dict['length'] = 6
        video_dict['all_image_files'] = []
        for frame_id in video_frame_ids:
            image_file = os.path.join(image_dir, image_ids_to_image[frame_id]['file_name'])
            video_dict['file_names'].append(image_file)
            
            video_dict['image_ids'].append(frame_id)
            
            annotation_file = os.path.join(gt_dir, image_ids_to_annos[frame_id]['file_name'])
            video_dict['pan_seg_files'].append(annotation_file)
            
            assert image_file is not None, "No image {} found for annotation {}".format(
                frame_id, annotation_file
            )
            
            segments_info = image_ids_to_annos[frame_id]['segments_info']
            video_dict['segments_info'].append(segments_info)
            
            sem_seg_file = annotation_file.replace('panoptic_video', 'labelmap')
            video_dict['sem_seg_files'].append(sem_seg_file)
            
        assert len(video_dict["file_names"]) == video_dict['length'], video_frame_ids
            
        videos_list.append(video_dict)


    assert len(videos_list), "No images found in {}".format(image_dir)
    assert PathManager.isfile(videos_list[0]['file_names'][0]), videos_list[0]['file_names'][0]
    assert PathManager.isfile(videos_list[0]['pan_seg_files'][0]), videos_list[0]['pan_seg_files'][0]
    return videos_list


def register_all_cityscapes_panoptic(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json) in _CITYSCAPES_VPS_PANOPTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir, z=gt_json: get_cityscapes_vps_dicts(x, y, z, meta)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_vps",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )


if __name__ == "__main__":
    register_all_cityscapes_panoptic('/home/fardin/VPS/Mask2Former/datasets/cityscapes-vps')
    d = MetadataCatalog.get('cityscapes_vps_panoptic_train')
    print(d)