
import argparse
import os
from nuscenes.eval.common.config import config_factory
import json
from nuscenes.eval.detection.data_classes import DetectionConfig
from classes.GenericDetectionEval import GenericDetectionEval


# Código baseado em: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py
'''
Avalia um conjunto de predições
'''
if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Framework de avaliação 3D genérico.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gts_path', type=str, 
                        help='''
                        Caminho o qual o JSON com as GTs estão guardados. 
                        Podem ser passados alguns atalhos também:
                        - nuscenes_challenge: utiliza GTs do desafio de detecção NuScenes (em `gts/detection_trainval_val.json`), sem usar nenhum filtro.
                        - nuscenes_vrus-and-cars: utiliza GTs do desafio de detecção NuScenes (em `gts/detection_trainval_val.json`) utilizando o filtro para pedestre, ciclista, motociclista e carros. O filtro usado está disponível em `filters/nuscenes_vrus-and-cars.json`.
                        - nuscenes_vrus: utiliza GTs do desafio de detecção NuScenes (em `gts/detection_trainval_val.json`) utilizando o filtro para VRUs (pedestre, ciclista e motociclista). O filtro usado está disponível em `filters/nuscenes_vrus.json`.

                        OBS: os filtros usados aqui podem ser sobrescritos pelo argumento "--filter-path"
                        ''')
    parser.add_argument('result_path', type=str, help='''
                        Arquivo JSON com uma lista de dicionários com informações sobre predições. Cada dicionário possui:
                        - name: nome das predições (usado para identificar as métricas de cada execução específica)
                        - infer_path: onde as predições estão salvas
                        - save_path: onde as métricas e gráficos serão salvos
                        ''')
    parser.add_argument('--output_dir', type=str, default='./agg_results.json',
                        help='Local onde os resultados agregados serão armazenados. É um arquivo JSON mapeando métricas de cada execução (baseado no nome de cada resultado passado no JSON fornecido em `result_path`)')
    parser.add_argument('--filter_path', type=str, default='',
                        help='Caminho para o JSON com os filtros de classes. Caso não seja fornecido, nenhum filtro será aplicado.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Caminho do arquivo de configuração'
                             'Se não for fornecido, configurações padrões do desafio da NuScenes serão utilizadas.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Gera ou não gera gráficos de curvas de PR e TP')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Adiciona ou remove prints no terminal')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    config_path = args.config_path
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)
    gts_path_ = args.gts_path
    filter_path_arg = args.filter_path
    filter_path_ = None  # It will be defined soon

    # Load gts_path
    if gts_path_ == 'nuscenes_challenge':
        gts_path_ = 'gts/detection_trainval_val.json'
    elif gts_path_ == 'nuscenes_vrus-and-cars':
        gts_path_ = 'gts/detection_trainval_val.json'
        filter_path_ = 'filters/nuscenes_vrus-and-cars.json'
    elif gts_path_ == 'nuscenes_vrus':
        gts_path_ = 'gts/detection_trainval_val.json'
        filter_path_ = 'filters/nuscenes_vrus.json'
    elif gts_path_ == 'nuscenes_vrus-and-vehicles':
        gts_path_ = 'gts/detection_trainval_val.json'
        filter_path_ = 'filters/nuscenes_vrus-and-vehicles.json'

    # Overwrite filter_path if passed by argument
    if filter_path_arg != '':
        filter_path_ = filter_path_arg
    
    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    with open(result_path_, 'r') as f:
        infers_set = json.load(f)

    agg_metrics = {}
    for i, infer_info in enumerate(infers_set):
        print(f"Evaluating {infer_info['name']}: {i+1}/{len(infers_set)}")

        nusc_eval = GenericDetectionEval(result_path=infer_info['infer_path'], gts_path=gts_path_, filter_path=filter_path_, config=cfg_, output_dir=infer_info['save_path'], verbose=verbose_)
        metrics = nusc_eval.main(plot_examples=0, render_curves=render_curves_)

        agg_metrics[infer_info['name']] = metrics

    os.makedirs(os.path.dirname(output_dir_), exist_ok=True)

    with open(output_dir_, 'w', encoding='utf-8') as file:
            json.dump(agg_metrics, file, ensure_ascii=False, indent=4)