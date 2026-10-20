# Rodar na DGX Spark (laboratório)

A DGX Spark tem CPU ARM (Grace, aarch64), GPU Blackwell GB10 e 128 GB de memória unificada. O caminho recomendado é o **container NGC do PyTorch**, que já traz um PyTorch com CUDA para essa GPU. O script [`scripts/dgx/run.sh`](../scripts/dgx/run.sh) cuida de construir e usar o container.

> Os scripts foram testados fora da Spark: sintaxe, montagem dos comandos e o executor rodando em CPU. **Ainda não rodaram numa Spark de verdade.** Por isso o passo 1 abaixo serve de checagem.

## Pré-requisitos

- **Docker com GPU:** `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.09-py3 nvidia-smi` precisa funcionar. O DGX OS já vem com o NVIDIA Container Toolkit.
- **Permissão no Docker:** seu usuário precisa estar no grupo `docker`; teste com `docker ps`. Se não tiver permissão, use a seção [Sem Docker](#sem-docker).
- **Rede:** acesso à Internet para baixar a imagem do NGC (~10–20 GB na primeira vez), os pesos ImageNet do PVTv2 (Hugging Face) e os dados (Google Drive).

## Passo a passo

```bash
git clone -b claude/gifted-albattani-ikhwce https://github.com/warrior-graph/PAS-PolypAutoSegmentation.git
cd PAS-PolypAutoSegmentation

scripts/dgx/run.sh check     # 1. constrói a imagem pas-dgx; mostra GPU, torch, CUDA e o dtype; roda os testes
scripts/dgx/run.sh bench     # 2. baixa os dados (~700 MB) e cronometra 3 épocas do B0 (~3 min)
nohup scripts/dgx/run.sh matrix > dgx_round2.log 2>&1 &   # 3. matriz completa, segue rodando se o SSH cair
tail -f dgx_round2.log       # acompanhar; Ctrl+C só sai do tail
```

O que olhar em cada passo:
1. **`check`:** a saída deve mostrar `autocast dtype torch.bfloat16`, a capability da GPU (≥ 8) e terminar com `27 passed`.
2. **`bench`:** a última linha dá os segundos por época do B0 e a estimativa para 100 épocas. A referência é 26 s por época numa T4 do Kaggle.
3. **`matrix`:** treina e avalia os 6 experimentos de [`experiments/round2.yaml`](../experiments/round2.yaml), um de cada vez, e no fim escreve `runs/round2/summary.md`.

### O que a matriz roda

| ID | Receita | Estimativa na Spark |
|---|---|---|
| `e0_b0_basic` | baseline refeito na mesma máquina, para comparar no mesmo hardware | 20–35 min |
| `e1_b0_strong` | aumentações fortes de escala e aparência, com reflexos sintéticos | 25–40 min |
| `e2_b0_448` | entrada de 448 px | 30–50 min |
| `e3_b0_strong_448` | fortes + 448 px + 150 épocas | 50–80 min |
| `e4_b2_basic` | encoder B2 (26,8M parâmetros) | 1–1,5 h |
| `e5_b2_strong_448` | B2 + fortes + 448 px + 150 épocas | 2,5–3,5 h |

**Total estimado:** de 6 a 9 h, então dá para deixar rodando de um dia para o outro. As estimativas ainda não foram medidas na Spark; o `bench` do passo 2 dá o número real.

Cada experimento é avaliado em quatro leituras:
- `clean`: protocolo limpo, sem TTA;
- `tta`: com flip;
- `tta_ms`: flip + escalas 1,0/1,25/1,5;
- `tta_ms_cc`: o anterior + filtro de componentes.

### Variações úteis

```bash
scripts/dgx/run.sh matrix --only e1_b0_strong e2_b0_448    # só alguns experimentos
scripts/dgx/run.sh matrix --jobs 2                          # dois experimentos dividindo a GPU
scripts/dgx/run.sh matrix --set train.epochs=50             # override aplicado a todos
scripts/dgx/run.sh matrix --summary-only                    # só refaz a tabela com o que terminou
scripts/dgx/run.sh shell                                    # shell dentro do container
scripts/dgx/run.sh -- python eval.py --checkpoint runs/round2/e1_b0_strong/best.pth --tta
```

- **`--jobs 2`:** os 128 GB comportam dois runs. Só compensa se a GPU estiver ociosa: compare o tempo por época com o do `bench`.
- **Interrupção:** se o processo cair, rode o mesmo comando de novo. Treinos com `test_results.json` e avaliações já salvas são pulados.
- **Acompanhar a GPU:** em outro terminal, `watch -n 5 nvidia-smi`. Os logs de cada experimento ficam em `runs/round2/<id>/runner.log` e `log.csv`.

### Variáveis de ambiente do `run.sh`

| Variável | Padrão | Uso |
|---|---|---|
| `DATA_DIR` | `<repo>/data` (baixado no primeiro uso) | pasta do host com `TrainDataset/` e `TestDataset/`, montada em `/data` |
| `BASE` | `nvcr.io/nvidia/pytorch:25.09-py3` | outra tag do NGC (use com `REBUILD=1`) |
| `REBUILD` | `0` | `1` reconstrói a imagem |
| `IMAGE` | `pas-dgx` | nome da imagem local |
| `MATRIX` | `experiments/round2.yaml` | outra matriz de experimentos |

## Trazer os resultados

O que importa é pequeno. Os `.pth` têm 15 a 100 MB cada e só precisam vir se você quiser usar o modelo.

```bash
cat runs/round2/summary.md
# copiar só tabelas e logs para outra máquina:
rsync -av --include='*/' --include='*.md' --include='*.csv' --include='*.json' --include='config.yaml' \
      --exclude='*' runs/round2/ usuario@outra-maquina:PAS/runs/round2/
```

Para eu analisar, basta mandar o `summary.md`, ou os `*.json` de cada experimento.

## Problemas comuns

| Sintoma | Solução |
|---|---|
| `manifest unknown` / tag não encontrada ao construir | escolher uma tag existente em https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch e rodar `BASE=nvcr.io/nvidia/pytorch:<tag> REBUILD=1 scripts/dgx/run.sh check` |
| `pip` falha no build por conflito com `PIP_CONSTRAINT` do NGC | no `docker/Dockerfile.dgx`, trocar a linha por `RUN PIP_CONSTRAINT= pip install --no-cache-dir -r /tmp/requirements-dgx.txt` e reconstruir com `REBUILD=1` |
| `permission denied ... docker.sock` | pedir para entrar no grupo `docker`, ou usar o venv abaixo |
| `autocast dtype torch.float16` no `check` | a GPU não foi reconhecida como Ampere ou mais nova; mande a saída do `check` |
| arquivos em `runs/` com dono errado | o `run.sh` já roda com o seu UID; se vieram de outra execução, use `sudo chown -R $USER runs data .cache` |
| download dos dados falha | baixar à mão (links no erro) e usar `DATA_DIR=/caminho` |

## Sem Docker

O [`scripts/dgx/setup_venv.sh`](../scripts/dgx/setup_venv.sh) cria um venv com o PyTorch CUDA para aarch64 e testa uma convolução em bf16 na GPU.

```bash
scripts/dgx/setup_venv.sh                    # TORCH_INDEX=cu130 por padrão; tente cu129/cu128 se falhar
source .venv-dgx/bin/activate
python scripts/download_data.py
python scripts/run_experiments.py experiments/round2.yaml --bench
nohup python scripts/run_experiments.py experiments/round2.yaml > dgx_round2.log 2>&1 &
```
