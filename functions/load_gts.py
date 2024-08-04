
from nuscenes.eval.common.data_classes import EvalBoxes
import json

def load_gts(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads bounding boxes from GTs JSON file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: EvalBoxes object with the GTs boxes.
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