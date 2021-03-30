import os

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputs, fast_rcnn_inference_single_image)
from loguru import logger

# Load VG Classes
#butd_dir = '/dependencies/py-bottom-up-attention/demo/'
butd_dir = '/home/eugene/Documents/workspace/py-bottom-up-attention/demo/'
data_path = os.path.join(butd_dir, 'data/genome/1600-400-20')

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

cfg = get_cfg()
cfg.merge_from_file(os.path.join(
    butd_dir, "../configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml"))
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
predictor = DefaultPredictor(cfg)

NUM_OBJECTS = 100


def butd_raw_output(raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        logger.info(f"Original image size: ({raw_height}, {raw_width})")

        # Preprocessing
        image = predictor.transform_gen.get_transform(
            raw_image).apply_image(raw_image)
        logger.info(f"Transformed image size: {image.shape[:2]}")
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(
            images, features, None)
        proposal = proposals[0]
        logger.info(f'Proposal Boxes size: {proposal.proposal_boxes.tensor.shape}')

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        logger.info(f'Pooled features size: {feature_pooled.shape}')

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(
            feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        logger.info(instances)

    return instances, roi_features


def add_spatial_features(instances, features):
    boxes = instances.pred_boxes.tensor
    image_height = instances.image_size[0]
    image_width = instances.image_size[1]

    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    scaled_width = box_width / image_width
    scaled_height = box_height / image_height
    scaled_x = boxes[:, 0] / image_width
    scaled_y = boxes[:, 1] / image_height
    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]
    spatial_features = np.concatenate(
        (scaled_x,
         scaled_y,
         scaled_x + scaled_width,
         scaled_y + scaled_height,
         scaled_width,
         scaled_height),
        axis=1)

    full_features = torch.from_numpy(
        np.concatenate((features, spatial_features), axis=1))
    return full_features


def get_labels(instances):
    classes = np.asarray(vg_classes)
    return " ".join(classes[instances.pred_classes])


def get_image_features(img):
    instances, features = butd_raw_output(img)
    features = add_spatial_features(instances, features)
    labels = get_labels(instances)
    logger.info(labels)

    return features, labels
