# Rodar numa GPU local (4 a 8 GB)

Este guia serve para uma GPU NVIDIA comum num desktop ou notebook, como RTX 3050/3060/4060 ou GTX 1650. Sem GPU, também dá para usar a CPU, mas só para testes e inferência.

## 1. Instalação

Pré-requisitos:
- Linux, ou Windows com **WSL2** (recomendado no Windows);
- driver NVIDIA recente (`nvidia-smi` precisa funcionar);
- Python 3.10 ou mais novo.

```bash
git clone -b claude/gifted-albattani-ikhwce https://github.com/warrior-graph/PAS-PolypAutoSegmentation.git
cd PAS-PolypAutoSegmentation
python -m venv .venv && source .venv/bin/activate          # Windows nativo: .venv\Scripts\activate

# 1) PyTorch com CUDA (escolha o índice em https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 2) resto das dependências
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Se aparecer `False`, o PyTorch foi instalado sem CUDA: reinstale com o `--index-url` certo para o seu driver.

## 2. Dados e testes

```bash
python scripts/download_data.py      # ~700 MB -> data/TrainDataset e data/TestDataset
pytest tests/                        # ~1 min na CPU; deve terminar com "27 passed"
```

Se o download do Google Drive falhar, o script mostra os links para baixar à mão e descompactar em `data/`.

## 3. Treinar

```bash
python train.py --config configs/emcad_b0.yaml
```

- **O que o treino faz:**
  - treina por 100 épocas;
  - guarda o melhor checkpoint pela validação (10% do treino) em `runs/emcad_b0/best.pth`;
  - no fim, avalia nos 5 conjuntos de teste e salva o resultado em `runs/emcad_b0/test_results.md`.
- **Teste rápido antes do treino longo:** `--max-steps 20` só confere se cabe na memória e se não dá erro.
- **Precisão:** é escolhida automaticamente. RTX 30/40 usam bf16; RTX 20, GTX 16 e GTX 10 usam fp16.

### Ajuste por placa

Qualquer chave do YAML pode ser trocada com `--set secao.chave=valor`.

| GPU | Modelo | Comando |
|---|---|---|
| 4 GB | B0 (padrão) | `python train.py --config configs/emcad_b0.yaml` (batch 8 × acumulação 2) |
| 4 GB | B0 em 448 px | `... --set data.image_size=448 train.batch_size=4 train.grad_accum=4` |
| 8 GB | B0 | `... --set train.batch_size=16 train.grad_accum=1` |
| 8 GB | B2 (padrão) | `python train.py --config configs/emcad_b2.yaml` (batch 8 × acumulação 2) |
| 8 GB | B2 em 448 px | `python train.py --config configs/emcad_b2.yaml --set data.image_size=448 train.batch_size=4 train.grad_accum=4` |

- **Os tamanhos de batch acima não foram medidos nessas placas.** Se der `CUDA out of memory`, divida `train.batch_size` por 2 e multiplique `train.grad_accum` por 2. O batch efetivo continua o mesmo.
- **Workers:** use `data.num_workers` igual ao número de núcleos da CPU (o padrão é 4). Se der problema no Windows nativo, use `data.num_workers=0`.
- **Tempo:** também não foi medido nessas placas. A referência é 26 s por época do B0 numa T4 do Kaggle, o que dá ~45 min por 100 épocas. Para medir na sua placa, use o `--bench` da seção 5.

## 4. Avaliar e usar

```bash
python eval.py --checkpoint runs/emcad_b0/best.pth                         # protocolo limpo
python eval.py --checkpoint runs/emcad_b0/best.pth --tta --scales 1.0 1.25 1.5   # flip + multiescala
python predict.py --checkpoint runs/emcad_b0/best.pth --input pasta_ou_imagem --output preds/
pip install gradio && python app.py --checkpoint runs/emcad_b0/best.pth    # demo no navegador
```

O `eval.py` grava `eval*.md`, `.csv` e `.json` ao lado do checkpoint. Para usar o modelo treinado no Kaggle, descompacte o `emcad_b0_run.zip` em `runs/`.

## 5. Matriz de experimentos (rodada 2)

O executor roda vários treinos em sequência. Cada um é avaliado em 4 leituras, e no fim o executor gera `runs/round2/summary.md`. Os defaults do YAML assumem uma GPU grande, então numa placa pequena passe `--set`:

```bash
# medir quanto tempo leva uma época na sua placa
python scripts/run_experiments.py experiments/round2.yaml --bench --set train.batch_size=8 train.grad_accum=2 data.num_workers=4

# rodar só alguns experimentos (4 GB: só os B0)
python scripts/run_experiments.py experiments/round2.yaml --only e0_b0_basic e1_b0_strong \
    --set train.batch_size=8 train.grad_accum=2 data.num_workers=4
```

- **Placa de 4 GB:** para `e2`/`e3` (448 px), use `train.batch_size=4 train.grad_accum=4`.
- **B2 (`e4`/`e5`):** em 8 GB, use `train.batch_size=4 train.grad_accum=4`.
- **Interrupção:** se parar, rode o mesmo comando de novo; o que já terminou é pulado.

## Sem GPU (CPU)

Instale com `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` e depois `pip install -r requirements.txt`.

Na CPU funcionam os testes, o `predict.py`, o `app.py` e o `eval.py` com `--device cpu`. Treinar na CPU é inviável: medimos de 9 a 15 min por época, a 256 px e com 4 núcleos.
