
# Framework de avaliação 3D

Este repositório possui scripts Python para avaliar predições de bounding boxes 3D. A avaliação é feita nos moldes da [competição de detecção da base de dados NuScenes](https://www.nuscenes.org/object-detection).

OBS: esse código apenas avaliará (gerando métricas e gráficos) as predições já feitas e armazenadas, não deve ser utilizado para realizar o treinamento de modelos ou geração das predições em si.

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

Para avaliar as predições de um modelo, utilize o script disponível em `eval.py`. Para executar o script, utilize o comando abaixo:

```
python eval.py \
  [gts_path] \
  [result_path] \
  --output_dir [output_dir] \
  --filter_path [filter_path] \
  --config_path [config_path] \
  --render_curves [render_curves] \
  --verbose [verbose]
```

Fornecendo os seguintes argumentos:
- `[gts_path]`: Caminho o qual o JSON com as GTs estão guardados. Podem ser passados alguns atalhos também:
  - `nuscenes_challenge`: utiliza GTs do desafio de detecção NuScenes (em `gts/detection_trainval_val.json`), sem usar nenhum filtro.
  - `nuscenes_vrus-and-cars`: utiliza GTs do desafio de detecção NuScenes (em `gts/detection_trainval_val.json`) utilizando o filtro para pedestre, ciclista, motociclista e carros. O filtro usado está disponível em `filters/nuscenes_vrus-and-cars.json`.
  - `nuscenes_vrus`: utiliza GTs do desafio de detecção NuScenes (em `gts/detection_trainval_val.json`) utilizando o filtro para VRUs (pedestre, ciclista e motociclista). O filtro usado está disponível em `filters/nuscenes_vrus.json`.
    
    OBS: os filtros usados aqui podem ser sobrescritos pelo argumento "--filter-path"
- `[result_path]`: O caminho para o arquivo JSON contendo as predições.
- `[output_dir]`: Parâmetro opcional, sendo o local onde os resultados serão armazenados (métricas, gráficos, etc.). Caso não seja fornecido, será salvo em `./metrics`.
- `[filter_path]`: Parâmetro opcional, sendo o caminho para o JSON com os filtros de classes. Caso não seja fornecido, nenhum filtro será aplicado.
- `[config_path]`: Parâmetro opcional, sendo o caminho para o arquivo de configurações. Se não for fornecido, [configurações padrões do desafio da NuScenes serão utilizadas](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/configs/detection_cvpr_2019.json).
- `[render_curves]`: Parâmetro opcional, definindo se os gráficos de curvas de PR e TP serão gerados ou não. Por padrão será gerado (1), mas pode ser passado o valor 0 para desabilitar.
- `[verbose]`: Parâmetro opcional, definindo se mensagens serão impressas no terminal ou não. Por padrão serão imprimidas as mensagens (1), mas pode ser passado o valor 0 para desabilitar.

### Padrão dos arquivos JSON

O script `eval.py` pode utilizar vários arquivos JSON em sua execução. Os padrões de cada JSON estão definidos a seguir:

#### Padrão dos GTs

O arquivo JSON contendo os GTs, que deve ser passado no argumento `[gts_path]`, segue um padrão bem parecido com os [resultados do desafio da NuScenes](https://www.nuscenes.org/object-detection), descrito logo abaixo:

```
{
  sample_token : [                      // Para cada sample da base de dados, deve ser fornecido uma lista com informações das bounding boxes veradeiras (GTs). Cada sample precisa ser identificada unicamente por um "sample_token"
    {
      "sample_token": "str"             // Uma string contendo o "sample_token" o qual a bounding box pertence. Deve ser o mesmo valor da chave utilizada para salvar a lista de bounding boxes.
      "translation": "list = float[3]"  // Uma lista contendo 3 valores de ponto flutuante, representando a posição global da bbox. Segue o padrão [<center_x>, <center_y>, <center_z>].
      "size": "list = float[3]"         // Uma lista contendo 3 valores de ponto flutuante, representando o tamanho da bbox. Segue o padrão [<width>, <length>, <height>].
      "rotation": "list = float[4]"     // Uma lista contendo 4 valores de ponto flutuante, representando a rotação da bbox. Segue o padrão [<w>, <x>, <y>, <z>].
      "velocity": "list = float[2]"     // Uma lista contendo 2 valores de ponto flutuante, representando a velocidade da bbox. Segue o padrão [<vx>, <vy>].
      "detection_name": "str"           // Nome da classe que deve ser prevista.
      "detection_score" "float"         // Não é obrigatório, mas se for passado, deve ser sempre -1 (já que são GTs, esse valor funciona para as predições)
      "attribute_name": "str"           // Atributo do objeto atual (exemplo: vehicle.moving, vehicle.parked). Pode ser uma string vazia "" se não houver esse tipo de atributo para o objeto.
    }
    ...                                 // Segue o mesmo padrão definido acima para todas as bounding boxes
  ]
}
```

### Padrão das predições

O arquivo JSON contendo as predições, que deve ser passado no argumento `[result_path]`, segue o padrão definido na [competição de detecção da NuScenes](https://www.nuscenes.org/object-detection). Este padrão está descrito logo abaixo:

```
{
  "meta": {
        "use_camera":   <bool>                // Se o modelo utilizou dados de câmera, marcar como verdadeiro.
        "use_lidar":    <bool>                // Se o modelo utilizou dados de LiDAR, marcar como verdadeiro.
        "use_radar":    <bool>                // Se o modelo utilizou dados de radar, marcar como verdadeiro.
        "use_map":      <bool>                // Se o modelo utilizou dados de mapa, marcar como verdadeiro.
        "use_external": <bool>                // Se o modelo utilizou dados externos, marcar como verdadeiro.
    },
    "results": {
        sample_token : [                      // Para cada sample da base de dados, deve ser fornecido uma lista com informações das bounding boxes veradeiras (GTs). Cada sample precisa ser identificada unicamente por um "sample_token"
          {
            "sample_token": "str"             // Uma string contendo o "sample_token" o qual a bounding box pertence. Deve ser o mesmo valor da chave utilizada para salvar a lista de bounding boxes.
            "translation": "list = float[3]"  // Uma lista contendo 3 valores de ponto flutuante, representando a posição global da bbox. Segue o padrão [<center_x>, <center_y>, <center_z>].
            "size": "list = float[3]"         // Uma lista contendo 3 valores de ponto flutuante, representando o tamanho da bbox. Segue o padrão [<width>, <length>, <height>].
            "rotation": "list = float[4]"     // Uma lista contendo 4 valores de ponto flutuante, representando a rotação da bbox. Segue o padrão [<w>, <x>, <y>, <z>].
            "velocity": "list = float[2]"     // Uma lista contendo 2 valores de ponto flutuante, representando a velocidade da bbox. Segue o padrão [<vx>, <vy>].
            "detection_name": "str"           // Nome da classe que deve ser prevista.
            "detection_score" "float"         // Pontuação obtida pela classe definida em "detection_name"
            "attribute_name": "str"           // Atributo do objeto atual (exemplo: vehicle.moving, vehicle.parked). Pode ser uma string vazia "" se não houver esse tipo de atributo para o objeto.
          }
          ...                                 // Segue o mesmo padrão definido acima para todas as bounding boxes
        ]
    }
}
```

### Padrão do filtro

O JSON contendo o filtro, que pode ser passado no argumento `[filter_path]`, deve seguir o seguinte formato:

```
{
  new_class_name: "list[str]"  // É um JSON o qual as chaves são novos nomes para as classes e os valores são listas contendo nomes de classes antigos (que estavam no GTs e predições originalmente)
                               // OBS: todos os nomes de classes antigas que não aparecerem em nenhuma das lista de valores serão removidas da avaliação final
}
```

Um exemplo de filtro é fornecido logo abaixo:

```
{
    "VRU": ["pedestrian", "bicycle", "motorcycle"]
}
```

Este filtro pode ser utilizado na avaliação da NuScenes, por exemplo. Neste caso, as classes antigas de *pedestrian*, *bicycle* e *motorcycle* serão trocadas por apenas uma classe chamada "VRU". Todas as outras classes que não constam nesse JSON (como "car", "barrier", etc.) serão removidas da avaliação.

## Avaliando modelos na base de dados NuScenes

Para fazer a validação na base de dados da nuScenes, existem dois scripts que precisam ser utilizados antes do script de avaliação `eval.py`:

1- O primeiro script que precisa ser utilizado é responsável por gerar o JSON com os GTs da base de dados da NuScenes. É preciso ter toda a base de dados instalada no diretório `data/nuscenes` para fazer isso. Para criar o JSON, execute o comando abaixo:

```
python nuscenes_scripts/generate_nuscenes_gts.py
```

Ao executar esse script, as GTs serão criadas no caminho `gts/detection_trainval_val.json`. É possível configurar esse script por argumentos também, verifique o código disponível em `nuscenes_scripts/generate_nuscenes_gts.py`.

2- O segundo script faz um filtro de algumas bounding boxes contidas nas previsões. Essa filtragem é a mesma que é feita no [pré-processamento da avaliação da NuScenes](https://www.nuscenes.org/object-detection). Para fazer essa filtragem execute o comando abaixo:

```
python nuscenes_scripts/filter_nuscenes_boxes.py [input_path]
```

Onde `[input_path]` deve ser substituido pelo caminho onde o JSON das previsões estão armazenados. Ao executar o script, o JSON com os bounding boxes filtrados serão guardados no mesmo caminho do arquivo original, em um arquivo JSON com o mesmo nome com a adição de um pós-fixo de `_filtered`. É possível configurar esse script por argumentos também, verifique o código disponível em `nuscenes_scripts/filter_nuscenes_boxes.py`.
