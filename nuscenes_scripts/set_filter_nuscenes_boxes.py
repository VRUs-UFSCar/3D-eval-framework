
import argparse
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_prediction
from nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox
import json
import numpy as np
from nuscenes.eval.common.config import config_factory
import os

'''
Esse script tem como objetivo pré-processar (filtar) um conjunto de resultados de previsões para a base de dados NuScenes.
O filtro utilizado aqui é o mesmo que é utilizado no desafio de detecção da NuScenes 

Verifique os argumentos abaixo para possíveis configurações deste script ( Preprocessing: https://nuscenes.org/object-detection )
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the bounding boxes for NuScenes challenge (Preprocessing: https://nuscenes.org/object-detection)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', type=str, help='''
                        JSON file with a list of dicts with infers informations. Each dict has:
                        - infer_path: where the predictions are saved
                        - (optional) save_path: where to save the filtered predictions
                        ''')
    parser.add_argument('--use_save_paths', type=int, default=0,
                        help='Wheter to use the save_path given the results JSON. If 0, all filtered predictions will be saved in the same directory of the predictios, with the pos-fix "_filtered.json"')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes GTs to get GTs, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--dataroot', type=str, default='data/nuscenes',
                        help='Default nuScenes data directory.')

    args = parser.parse_args()

    input_path = args.input_path
    use_save_paths = args.use_save_paths
    verbose = args.verbose
    version = args.version
    dataroot = args.dataroot

    nusc = NuScenes(version=version, verbose=verbose, dataroot=dataroot)

    cfg = config_factory('detection_cvpr_2019')

    with open(input_path, 'r') as f:
        infers_set = json.load(f)
    
    for i, infer_info in enumerate(infers_set):
        print(f'Filtering {i+1}/{len(infers_set)}')

        infer_path = infer_info['infer_path']

        boxes, meta = load_prediction(infer_path, cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
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

        if not use_save_paths:
            output_path = f'{infer_path[:-5]}_filtered.json'
        else:
            output_path = infer_info['save_path']
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(gts_json_final, file, ensure_ascii=False, indent=4)