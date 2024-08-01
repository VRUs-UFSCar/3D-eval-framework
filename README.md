
# Framework de avaliação 3D

Este repositório possui scripts Python para avaliar predições de bounding boxes 3D. A avaliação é feita nos moldes da [competição de detecção da base de dados NuScenes](https://www.nuscenes.org/object-detection).

## Instalação via docker

Para utilizar os scripts neste repositório é recomendado fazer a instalação via Docker.

Para instanciar um container para a execução dos scripts, siga os passos abaixo:

1- Crie a imagem utilizando o Dockerfile contido neste repositório:

`docker build -t 3d-eval-framework .`

2- Crie um container executando esse comando:

```
docker run -it \
  -v [gts-path]:/app/gts \
  -v [nuscenes-dataset-path]:/app/data/nuscenes \
  -v [output-path]:/app/outputs \
  3d-eval-framework /bin/bash
```

Antes de executar o comando acima, substitua os valores marcados:
- `[gts-path]`: caminho onde os JSONs *ground truth* (GT) estão armazenados na máquina real. Não é obrigatório fornecer isso, mas é recomendado para não precisar gerar esses JSONs toda vez que for usar os scripts contidos aqui.
- `[nuscenes-dataset-path]`: caminho onde a base de dados da NuScenes está armazenada na máquina real. É necessário fornecer esse caminho apenas quando for usar os scripts na pasta `nuscenes_scripts`.
- `[output-path]`: caminho onde os JSONs de previsões estão armazenados na máquina real. Esse caminho pode ser usado para salvar resultados de scripts também.

Após instanciar o container, é possível executar os scripts que serão descritos a seguir.

## Utilizando o script de avaliação genérico

[TODO]

### Padrão dos arquivos JSON

[TODO]

## Avaliando modelos na base de dados NuScenes

[TODO]
