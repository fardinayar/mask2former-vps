# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
import random


class MaskFormerPanopticDatasetMapper():
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        #assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        if sem_seg_gt is not None:
            sem_seg_gt = aug_input.sem_seg

        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            pan_seg_gt = F.pad(
                pan_seg_gt, padding_size, value=0
            ).contiguous()  # 0 is the VOID panoptic label

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Pemantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = []
        masks = []
        ids = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])
                ids.append(segment_info["id"])

        classes = np.array(classes)
        ids = np.array(ids)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.gt_ids = torch.tensor(ids, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances

        return dataset_dict
    
    
class MaskFormerPanopticVideoDatasetMapper():
    """
    From MinVIS: https://github.com/NVlabs/MinVIS with modifications
    """
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
    ):
        self.sampling_frame_num = sampling_frame_num
        self.sampling_frame_range = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.is_train= is_train

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        
        self.image_loader =  MaskFormerPanopticDatasetMapper(
            is_train= is_train,
            augmentations= augmentations,
            image_format= image_format,
            ignore_label= ignore_label,
            size_divisibility= size_divisibility,
        )
        

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "sampling_frame_num": cfg.INPUT.SAMPLING_FRAME_NUM,
            "sampling_frame_range": cfg.INPUT.SAMPLING_FRAME_RANGE,
            "sampling_frame_shuffle": cfg.INPUT.SAMPLING_FRAME_SHUFFLE,
        }
        return ret
    
    def select_frames(self, video_length):
        # From MinVIS: https://github.com/NVlabs/MinVIS
        """
        Args:
            video_length (int): length of the video

        Returns:
            selected_idx (list[int]): a list of selected frame indices
        """
        ref_frame = random.randrange(video_length)

        start_idx = max(0, ref_frame-self.sampling_frame_range)
        end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

        selected_idx = np.random.choice(
            np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
            self.sampling_frame_num - 1,
        )
        selected_idx = selected_idx.tolist() + [ref_frame]
        selected_idx = sorted(selected_idx)

        return selected_idx
    
    def __call__(self, dataset_dict): 
        """
        Args:
            dataset_dict (dict): Metadata of one video

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict_ = {} # Output dict to return.
        
        video_length = dataset_dict["length"]
        
        if self.is_train:
            selected_idx = self.select_frames(video_length)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
            all_frames_segments_info = []
            for frame_idx in selected_idx:
                all_frames_segments_info.extend([segment_info for segment_info in dataset_dict['segments_info'][frame_idx]])
        else:
            all_frames_segments_info = []
            selected_idx = range(video_length)
            
        dataset_dict_ = defaultdict(list)
        
        
        for frame_idx in selected_idx:
            image_dict = {
                "file_name": dataset_dict['file_names'][frame_idx],
                "image_id": dataset_dict['image_ids'][frame_idx],
                "pan_seg_file_name": dataset_dict['pan_seg_files'][frame_idx],
                "sem_seg_file_name": dataset_dict['sem_seg_files'][frame_idx],
                "segments_info": dataset_dict['segments_info'][frame_idx] if len(all_frames_segments_info) == 0 else \
                    all_frames_segments_info,
            }
            frame_dict = self.image_loader(image_dict)
            
            for key in frame_dict:
                dataset_dict_[key].append(frame_dict[key])
                
                
        return dataset_dict_
    
    
    
