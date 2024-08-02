# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction
)
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.render import class_pr_curve, class_tp_curve, dist_pr_curve, summary_plot, visualize_sample
from nuscenes.eval.detection.evaluate import DetectionEval

def load_gts(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> EvalBoxes:
    """
    Loads object predictions from GTs JSON file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data, box_cls)
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results


def filter_eval_boxes(boxes: EvalBoxes, classes_filter: dict[str, list[str]]) -> EvalBoxes:
    new_boxes = EvalBoxes()
    old_classes_to_new_classes_map: dict[str, str] = {}
    for new_class, old_classes_list in classes_filter.items():
        for old_class in old_classes_list:
            old_classes_to_new_classes_map[old_class] = new_class
    
    for sample_token in boxes.sample_tokens:
        old_sample_boxes: list[DetectionBox] = boxes[sample_token]
        new_sample_boxes: list[DetectionBox] = []

        for old_box in old_sample_boxes:
            if old_box.detection_name in old_classes_to_new_classes_map:
                old_box.detection_name = old_classes_to_new_classes_map[old_box.detection_name]
                new_sample_boxes.append(old_box)

        new_boxes.add_boxes(sample_token, new_sample_boxes)
    
    return new_boxes


class GenericDetectionEval(DetectionEval):
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 config: DetectionConfig,
                 result_path: str,
                 gts_path: str,
                 filter_path: str = None,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.result_path = result_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
        self.gt_boxes = load_gts(gts_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)

        if filter_path:
            if verbose:
                print('Filtering classes')
            
            with open(filter_path, mode='r') as json_file:
                classes_filter = json.load(json_file)
            
            print(classes_filter)
            self.cfg.class_names = list(classes_filter.keys())

            self.pred_boxes = filter_eval_boxes(self.pred_boxes, classes_filter)
            self.gt_boxes = filter_eval_boxes(self.gt_boxes, classes_filter)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        self.sample_tokens = self.gt_boxes.sample_tokens