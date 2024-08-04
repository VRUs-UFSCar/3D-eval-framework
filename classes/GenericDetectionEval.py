# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import json
import os

from functions.filter_eval_boxes import filter_eval_boxes
from functions.load_gts import load_gts
from functions.render import class_pr_curve, class_tp_curve, dist_pr_curve, summary_plot

from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig, DetectionMetricDataList, DetectionMetrics
from nuscenes.eval.detection.evaluate import DetectionEval


class GenericDetectionEval(DetectionEval):
    """
    This is an adaptation of the official nuScenes detection evaluation.
    It will calculate the same metrics, but it can be used for any dataset, given the GTs JSON.
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
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param gts_path: Path of the GTs JSON file.
        :param filter_path: Path to JSON filter file. If not given, it will not use any filters.
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
            
            self.cfg.class_names = list(classes_filter.keys())

            self.pred_boxes = filter_eval_boxes(self.pred_boxes, classes_filter)
            self.gt_boxes = filter_eval_boxes(self.gt_boxes, classes_filter)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        self.sample_tokens = self.gt_boxes.sample_tokens

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')
        
        detection_names = self.cfg.class_names
        pretty_detection_names = {}
        detection_colors = {}
        for i, detection_name in enumerate(detection_names):
            detection_colors[detection_name] = f'C{i}'
            pretty_detection_names[detection_name] = detection_name.replace('_', ' ').capitalize()

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'), detection_names=detection_names, pretty_detection_names=pretty_detection_names)

        for detection_name in detection_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'), pretty_detection_names=pretty_detection_names)

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'), pretty_detection_names=pretty_detection_names)

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)), pretty_detection_names=pretty_detection_names, detection_colors=detection_colors)