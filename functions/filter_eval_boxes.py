
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox


def filter_eval_boxes(boxes: EvalBoxes, classes_filter: dict[str, list[str]]) -> EvalBoxes:
    """
    Filter and rename boxes classes
    :param boxes: EvalBoxes that will be filtered and renamed
    :param classes_filter: A dict where the keys are new class names and the values are arrays with old class names that will be replaced by the new name. Old classes that not appear in the values will be removed.
    :return: new EvalBoxes object with the boxes filtered and renamed.
    """
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
