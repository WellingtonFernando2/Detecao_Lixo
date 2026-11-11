# Detecção e contagem de resíduos em tempo real com YOLO

Sistema que usa uma câmera (ou o celular, pelo aplicativo Iriun Webcam) para identificar
resíduos de **metal, papel, plástico e vidro** em tempo real e contar quantos de cada tipo
passaram pela câmera. Desenvolvido como Trabalho de Conclusão de Curso.

## Como funciona

- **Detecção:** modelo YOLOv8n (biblioteca Ultralytics), treinado com um conjunto de dados próprio.
- **Rastreamento:** ByteTrack dá um número de identificação a cada objeto.
- **Contagem:** uma linha vertical no meio da imagem simula uma esteira. Cada objeto é contado
  uma única vez, quando o centro da sua caixa cruza a linha.
- **Interface:** a imagem da câmera à esquerda e, à direita, um painel com a contagem por classe.
- **Relatório:** a tecla `s` salva a contagem em um arquivo CSV, que abre no Excel.

## Como instalar

Testado no Windows 11 com Python 3.12.7 e uma placa NVIDIA GeForce RTX 3050 (6 GB).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

O `requirements.txt` traz as versões exatas usadas no trabalho, inclusive o PyTorch 2.5.1 com
CUDA 12.1 (suporte à placa de vídeo, que deixa o treino muito mais rápido). Em um computador sem
placa NVIDIA o programa também funciona, só que mais devagar.

## Como usar

Com o modelo treinado que já vem na pasta `modelos`:

```bash
python app\realtime_detector.py --weights modelos\pequeno_v2.pt --camera celular --conf 0.65
```

- `--camera celular` abre o celular conectado pelo Iriun Webcam. Também aceita o número da
  câmera (`--camera 0`). Para ver as câmeras disponíveis: `python scripts\listar_cameras.py`.
- `--conf` é a confiança mínima de uma detecção.
- `--linha vertical|horizontal|off` e `--posicao 0.5` ajustam a linha de contagem.

Teclas na janela:

| Tecla | Ação |
|---|---|
| `s` | salva a contagem em um CSV, na pasta `relatorios` |
| `f` | salva uma foto do quadro atual em `fotos_brutas\erros` (para usar nos casos de erro) |
| `r` | zera a contagem |
| `q` | sai |

## Como treinar com as suas próprias imagens

1. Fotografe os objetos com a câmera que será usada na detecção:
   `python scripts\capturar_fotos.py --saida fotos_brutas\objetos --camera celular`
2. Fotografe também o fundo, sem nenhum resíduo (parede, mesa vazia, mãos), em
   `fotos_brutas\negativos`. Essas imagens ensinam o modelo o que não detectar.
3. Anote os objetos no [Roboflow](https://roboflow.com) (caixa ou contorno em cada objeto) e
   exporte o conjunto no formato **YOLOv8**.
4. Monte o conjunto de dados:
   `python scripts\montar_dataset.py --roboflow-dir "pasta_do_export" --negativos fotos_brutas\negativos --manter metal papel plastico vidro`
5. Treine:
   `python scripts\train.py --data data\dataset_pequeno.yaml --epochs 100 --name meu_treino --device 0 --batch 8`

O modelo treinado fica em `runs\detect\meu_treino\weights\best.pt`.

## Resultados do modelo final

Conjunto de dados: 514 fotos de objetos (520 imagens anotadas) e 294 imagens negativas.
Treino com 650 imagens e validação com 163.

| Classe | Precisão | Recall | mAP50 |
|---|---|---|---|
| Metal | 0,972 | 0,925 | 0,966 |
| Papel | 0,946 | 0,973 | 0,980 |
| Plástico | 0,948 | 1,000 | 0,995 |
| Vidro | 0,903 | 0,967 | 0,957 |
| **Geral** | **0,942** | **0,966** | **0,974** |

As imagens de validação vêm da mesma coleta das de treino, então o desempenho com a câmera ao
vivo pode ser um pouco menor.

## Estrutura do projeto

```
app/          programa principal (camera.py e realtime_detector.py)
scripts/      captura de fotos, listagem de câmeras, montagem do conjunto de dados e treino
data/         dataset.yaml, modelo de configuração das classes
modelos/      modelo treinado final (pequeno_v2.pt)
```

O conjunto de dados não está neste repositório, porque é pesado. Ele está disponível no link da
seção seguinte.

## Como reproduzir o trabalho

1. Instale o ambiente (seção "Como instalar").
2. Baixe o conjunto de dados em
   [https://drive.google.com/drive/folders/1LcE0yuvg9jutIsCDxb0fTip9AJV0oRFa](https://drive.google.com/drive/folders/1LcE0yuvg9jutIsCDxb0fTip9AJV0oRFa)
   (arquivo `dataset_pequeno.zip`) e
   descompacte na pasta principal do projeto, de modo que exista a pasta `dataset_pequeno`
   (com `images` e `labels`, em `train` e `val`). Por privacidade, esta versão não tem as imagens em
   que aparecem rostos de pessoas: são 616 imagens (492 de treino e 124 de validação), contra 813 no
   conjunto original. Por isso, treinar de novo com ela pode dar resultados um pouco diferentes dos
   do trabalho.
3. Para repetir o treinamento do modelo final:

   ```bash
   python scripts\train.py --data data\dataset_pequeno.yaml --epochs 100 --name pequeno_v2 --device 0 --batch 8
   ```

   Os parâmetros usados no trabalho estão em `modelos\pequeno_v2_args.yaml` (100 épocas, imagem de
   640 x 640, lote 8, semente 0 e modo determinístico). Ao final, o script faz a validação e mostra
   as métricas. Em outro computador, pequenas diferenças nos números podem aparecer, porque os
   cálculos feitos na placa de vídeo podem variar.
4. Para conferir o modelo que acompanha o repositório, sem treinar de novo, a validação com o
   conjunto de dados dá precisão 0,942, recall 0,966 e mAP50 0,974 (123 imagens e 138 objetos).
5. Para executar o sistema, use o comando da seção "Como usar".

Os comandos devem ser executados a partir da pasta principal do projeto.

## Créditos e licenças

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [OpenCV](https://opencv.org), [PyTorch](https://pytorch.org)
- [Roboflow](https://roboflow.com) para anotação e [Iriun Webcam](https://iriun.com) para usar o celular como câmera
