
import argparse
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes, load_prediction
from nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox
import json
import numpy as np
from nuscenes.eval.common.config import config_factory

'''
Esse script tem como objetivo pré-processar (filtar) os resultados de previsões para a base de dados NuScenes.
O filtro utilizado aqui é o mesmo que é utilizado no desafio de detecção da NuScenes 

Verifique os argumentos abaixo para possíveis configurações deste script( Preprocessing: https://nuscenes.org/object-detection)
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the bounding boxes for NuScenes challenge (Preprocessing: https://nuscenes.org/object-detection)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', type=str, help='JSON file with the bounding boxes to be converted')
    parser.add_argument('--output_path', type=str, default='',
                        help='Path to save the resulting JSON after preprocessing. If not specified, it will be saved in the same path of the input, with the "_filtered" pos-fix')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes GTs to get GTs, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--dataroot', type=str, default='data',
                        help='Default nuScenes data directory.')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    verbose = args.verbose
    version = args.version
    dataroot = args.dataroot

    nusc = NuScenes(version=version, verbose=verbose, dataroot=dataroot)

    cfg = config_factory('detection_cvpr_2019')

    boxes, meta = load_prediction(input_path, cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
    boxes = add_center_dist(nusc, boxes)
    boxes = filter_eval_boxes(nusc, boxes, cfg.class_range, verbose=verbose)

    if verbose:
        print('Serializing GTs to JSON...')
    boxes_json = boxes.serialize()

    # Converte para lista os objetos que não podem ser colocados em um .JSON
    for key, boxes in boxes_json.items():
        for box in boxes:
            for key, value in box.items():
                if isinstance(value, tuple):
                    box[key] = list(value)
                elif isinstance(value, np.ndarray):
                    box[key] = value.tolist()
    

    gts_json_final = {
        'results': boxes_json,
        'meta': meta
    }
    
    if verbose:
        print('Saving JSON to disk...')

    if output_path == '':
        output_path = f'{input_path[:-5]}_filtered.json'
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(gts_json_final, file, ensure_ascii=False, indent=4)