
import argparse
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox
import json
import numpy as np
from nuscenes.eval.common.config import config_factory
import os

'''
Esse script tem como objetivo gerar um JSON das GTs da NuScenes.
Este JSON poderá ser usado para avaliar algum modelo na base da NuScenes posteriormente utilizando ../eval.py

Verifique os argumentos abaixo para possíveis configurações deste script
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform NuScenes GTs database into JSON file.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='data/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to get GTs on, train, val or test.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes GTs to get GTs, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--save_path', type=str, default='',
                        help='Where the JSON with GTs will be saved. If not specified, it will be saved in this path: gts/detection_<version>_<eval_set>.json')
    
    args = parser.parse_args()

    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    verbose_ = bool(args.verbose)
    save_path_ = args.save_path

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    gts = load_gt(nusc_, eval_split=eval_set_, box_cls=DetectionBox, verbose=verbose_)
    
    if verbose_:
        print('Filtering ground truth annotations')
    gts = add_center_dist(nusc_, gts)
    gts = filter_eval_boxes(nusc_, gts, config_factory('detection_cvpr_2019').class_range, verbose=verbose_)
    
    if verbose_:
        print('Serializing GTs to JSON...')
    gts_json = gts.serialize()

    # Converte para lista os objetos que não podem ser colocados em um .JSON
    for key, boxes in gts_json.items():
        for box in boxes:
            for key, value in box.items():
                if isinstance(value, tuple):
                    box[key] = list(value)
                elif isinstance(value, np.ndarray):
                    box[key] = value.tolist()
    
    if verbose_:
        print('Saving JSON to disk...')

    if save_path_ == '':
        save_path_ = os.path.join('gts', f'detection_{version_[5:]}_{eval_set_}.json')
    os.makedirs(os.path.dirname(save_path_), exist_ok=True)
    with open(save_path_, 'w', encoding='utf-8') as file:
        json.dump(gts_json, file, ensure_ascii=False, indent=4)
