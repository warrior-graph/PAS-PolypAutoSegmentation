# PAS: Polyp Auto Segmentation

Segmentação automática de pólipos em imagens de colonoscopia.

O ponto de partida é um baseline leve e reprodutível: o encoder PVTv2 com o decoder **EMCAD** (CVPR 2024). Ele é treinado no protocolo padrão da literatura e pensado para GPUs de 4 a 8 GB e para o Kaggle.

> Protótipo de pesquisa. **Não é um dispositivo médico.**

## Como rodar

Há um guia para cada ambiente:

| Onde | Guia | Para quê |
|---|---|---|
| GPU local (4–8 GB) ou CPU | [docs/rodar_local.md](docs/rodar_local.md) | desenvolvimento, treinos B0, inferência e demo |
| Kaggle (T4 grátis) | [docs/rodar_kaggle.md](docs/rodar_kaggle.md) | treinos completos sem hardware próprio, pela interface web ou pela API |
| DGX Spark (laboratório) | [docs/rodar_dgx.md](docs/rodar_dgx.md) | matriz de experimentos inteira, B2 e alta resolução |

Resumo, em qualquer lugar com PyTorch instalado:

```bash
pip install -r requirements.txt
python scripts/download_data.py                       # ~700 MB -> data/TrainDataset, data/TestDataset
python train.py --config configs/emcad_b0.yaml        # treina e avalia nos 5 conjuntos de teste
python eval.py --checkpoint runs/emcad_b0/best.pth --tta --scales 1.0 1.25 1.5
python predict.py --checkpoint runs/emcad_b0/best.pth --input minha_imagem.png --output preds/
python scripts/run_experiments.py experiments/round2.yaml --only e1_b0_strong   # um experimento da matriz
```

Qualquer chave do YAML pode ser sobrescrita com `--set secao.chave=valor`. Os testes rodam com `pytest tests/`.

## Protocolo de avaliação

- **Treino:** `TrainDataset`, com 900 imagens do Kvasir-SEG e 550 do CVC-ClinicDB. É o split do PraNet e do Polyp-PVT.
- **Teste:**
  - Kvasir (100) e CVC-ClinicDB (62): mesmas fontes do treino;
  - CVC-ColonDB (380), ETIS-LaribPolypDB (196) e CVC-300 (60): **centros não vistos**, onde se mede a generalização.
- **Métricas:**
  - mDice e mIoU, com limiar 0,5;
  - S-measure (Sα);
  - E-measure média e máxima;
  - MAE.

  Todas são calculadas por imagem na resolução original e depois promediadas.
- **Seleção do checkpoint:** usamos 10% do `TrainDataset`, separados para validação. Muitos artigos escolhem o checkpoint diretamente nos conjuntos de teste, o que infla os números. Aqui isso não acontece.
- `eval.py` também reporta parâmetros, GFLOPs (multiplicações-somas, como nos artigos) e FPS.

## Resultados

### EMCAD-B0 (3,92M parâmetros, 1,8 GFLOPs) — Kaggle T4, 1 seed

- **Configuração:** `configs/emcad_b0.yaml`, 100 épocas, batch 16, fp16.
- **Tempo:** cerca de 45 minutos de treino.
- **Checkpoint:** escolhido pela validação separada do treino; o melhor foi o da época 77.
- **Velocidade:** cerca de 58 FPS na T4 com batch 1.
- **Artefatos:** [`results/emcad_b0_kaggle_t4/`](results/emcad_b0_kaggle_t4/).

| Dataset | mDice | mDice (+TTA flip) | mIoU (+TTA) | Sα (+TTA) | MAE (+TTA) |
|---|---|---|---|---|---|
| Kvasir | 0.914 | **0.918** | 0.865 | 0.926 | 0.029 |
| CVC-ClinicDB | **0.932** | 0.925 | 0.876 | 0.943 | 0.010 |
| CVC-ColonDB *(não visto)* | 0.755 | **0.765** | 0.683 | 0.845 | 0.041 |
| ETIS *(não visto)* | 0.758 | **0.781** | 0.701 | 0.865 | 0.020 |
| CVC-300 *(não visto)* | 0.883 | **0.892** | 0.816 | 0.933 | 0.009 |

Com flip + multiescala (`eval.py --tta --scales 1.0 1.25 1.5`), o mDice fica assim:

| Kvasir | ClinicDB | ColonDB | ETIS | CVC-300 |
|---|---|---|---|---|
| 0.917 | 0.939 | 0.759 | **0.803** | 0.899 |

Somando o filtro de componentes (`--min-component-rel 0.3`), o ETIS sobe para 0.810.

Para referência, os números publicados no mesmo protocolo (mDice) são estes:

| Modelo | Params | Kvasir | ClinicDB | ColonDB | ETIS | CVC-300 |
|---|---|---|---|---|---|---|
| PraNet (MICCAI 2020) | 32,6M | 0.898 | 0.899 | 0.709 | 0.628 | 0.871 |
| Polyp-PVT (2023) | 25,1M | 0.917 | 0.937 | 0.808 | 0.787 | 0.900 |
| **EMCAD-B0 (nosso, com TTA)** | **3,9M** | **0.918** | **0.925** | **0.765** | **0.781** | **0.892** |

Com cerca de 6× menos parâmetros, o B0 empata com o Polyp-PVT nos datasets vistos no treino, no ETIS e no CVC-300. A maior diferença está no **ColonDB, cerca de 4 pontos**.

Ressalvas: é uma única seed, e muitos artigos escolhem o checkpoint diretamente no conjunto de teste, o que não fazemos aqui.

### Diagnóstico do B0: onde ele erra

A análise foi feita imagem por imagem, sem TTA.

- **Poucas imagens concentram o erro.** As imagens com Dice abaixo de 0,5 respondem por **59% do erro no ColonDB** (67 de 380) e por **56% no ETIS** (33 de 196).
- **ColonDB:** o problema está nos extremos de tamanho.
  - Pólipos com menos de 1% da imagem ficam com Dice de 0,53.
  - Pólipos com mais de 10% da imagem ficam com 0,55; 16 deles passam totalmente despercebidos (lesões planas, de baixo contraste).
  - Os de tamanho intermediário ficam em torno de 0,85.
- **ETIS:** o problema é falso positivo.
  - Nas imagens ruins, a área marcada por engano é 7,9× a do gabarito, e o falso positivo domina em 30 de 33 casos.
  - O modelo gera mais de uma mancha em 70 imagens, embora 190 das 196 tenham um pólipo só.
- **Truques só na predição rendem pouco.** O ETIS ganha +3,9 com multiescala + flip; o ColonDB, quase nada. O filtro de componentes dá cerca de +0,5, e o limiar não muda nada.

Conclusão: o ganho real tem de vir do treino, e é isso que a rodada 2 ataca:
- aumentações de escala e aparência (`data.aug=strong`, com reflexos especulares sintéticos);
- resolução de entrada maior;
- encoder B2.

## Rodada 2 (em andamento)

A matriz [`experiments/round2.yaml`](experiments/round2.yaml) ataca o diagnóstico acima com seis experimentos:
- `e0`: o baseline refeito;
- `e1`: aumentações fortes;
- `e2`: entrada de 448 px;
- `e3`: aumentações fortes + 448 px;
- `e4`/`e5`: o mesmo com o encoder B2.

Cada experimento é avaliado sem TTA, com flip, com flip + multiescala e com o filtro de componentes. O executor é [`scripts/run_experiments.py`](scripts/run_experiments.py). Como rodar: [DGX](docs/rodar_dgx.md), [Kaggle](docs/rodar_kaggle.md#receitas-da-rodada-2-no-kaggle) ou [local](docs/rodar_local.md#5-matriz-de-experimentos-rodada-2).

## Estado da arte (outubro de 2026), em resumo

- **Datasets vistos no treino estão saturados.** O Dice fica entre 0,92 e 0,94 no Kvasir e no ClinicDB. Exemplos: Polyp-PVT, CASCADE, EMCAD, MEGANet, SCABNet.
- **A lacuna é a generalização.** Em centros não vistos, o Dice cai para cerca de 0,80 no ColonDB e para 0,75–0,82 no ETIS. Os casos difíceis são pólipos pequenos e planos, reflexos especulares e troca de equipamento.
- **Modelos de fundação (SAM, SAM2, DINOv2), com adaptadores ou Mamba, são os mais fortes fora do domínio.** Exemplos: SAM-Mamba (WACV 2025) e SAM-MaGuP. São caros demais para tempo real.
- **Destilação de modelos de fundação para modelos leves é a tendência de 2026.** Os concorrentes diretos são:
  - Polyp-DiFoM (WACV 2026), que destila SAM e DINOv2 para U-Net/U-Net++;
  - LiteBounD (arXiv 2604.17865), que faz destilação guiada por borda.
- **Em vídeo,** os destaques são o SUN-SEG, o FreeVPS (SAM2 sem treino) e o VP-SAM (ECCV 2024).

### Onde atacar

Usar um **professor pesado offline** e um **aluno leve online**:
- o SAM2 e o DINOv2 rodam uma vez na DGX Spark, e suas features e máscaras ficam salvas em disco;
- o aluno (este baseline) treina nas GPUs de 4 a 8 GB, lendo esse cache.

Para se diferenciar do Polyp-DiFoM e do LiteBounD, os candidatos são:
- (a) randomização de aparência e iluminação para robustez de domínio (o ETIS é exatamente o caso de troca de equipamento);
- (b) foco em pólipos pequenos e planos;
- (c) calibração e incerteza;
- (d) adaptação leve em tempo de teste.

## Roteiro

| Fase | Conteúdo | Hardware | Status |
|---|---|---|---|
| 0 | Infraestrutura, dados, métricas e testes | CPU | ✅ |
| 1 | Baselines reprodutíveis (EMCAD-B0/B2, depois U-Net e PraNet), com 3 seeds | 4–8 GB / Kaggle | 🟡 EMCAD-B0 treinado (1 seed); faltam B2 e seeds |
| 2 | Professor offline: SAM2 e DINOv2, com cache de features | DGX Spark | ⏳ |
| 3 | Aluno destilado + contribuição (a/b/c/d), ablações e PolypGen | 4–8 GB | ⏳ |
| 4 | Ferramenta: ONNX/TensorRT, demo em vídeo e consistência temporal | — | ⏳ |

## Estrutura

```
pas/
  data/dataset.py   # leitura no layout do PraNet, aumentações e split treino/val
  models/emcad.py   # decoder EMCAD (MSCB, EUCB, LGAG, CAB, SAB) + encoder PVTv2 do timm
  losses.py         # BCE + IoU ponderadas (PraNet) e supervisão profunda MUTATION (EMCAD)
  metrics.py        # Dice, IoU, Sα, Eφ (vetorizada) e MAE
  evaluation.py     # avaliação por conjunto, tabela, parâmetros/FLOPs/FPS
  inference.py      # Predictor, TTA flip/multiescala, filtro de componentes e overlay
train.py  eval.py  predict.py  app.py
configs/  experiments/round2.yaml          # configs base e matriz de experimentos
scripts/run_experiments.py                 # executor da matriz (treino + avaliações + resumo)
scripts/dgx/{run.sh,setup_venv.sh}  docker/Dockerfile.dgx   # DGX Spark
scripts/download_data.py  notebooks/kaggle_train.ipynb  tests/
docs/rodar_{local,kaggle,dgx}.md           # guias de execução
```

## Referências

- Fan et al., *PraNet*, MICCAI 2020: https://github.com/DengPingFan/PraNet
- Dong et al., *Polyp-PVT*, CAAI AIR 2023: https://github.com/DengPingFan/Polyp-PVT
- Rahman & Marculescu, *EMCAD*, CVPR 2024: https://arxiv.org/abs/2405.06880
- *Polyp-DiFoM*, WACV 2026: https://wacv.thecvf.com/virtual/2026/poster/845
- *LiteBounD*, 2026: https://arxiv.org/abs/2604.17865
- Dutta et al., *SAM-Mamba*, WACV 2025: https://arxiv.org/abs/2412.08482
- Ali et al., *PolypGen*, Scientific Data 2023: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9902556/
- *FreeVPS*, 2025: https://arxiv.org/abs/2508.19705
- Revisão sistemática, J. Imaging 2025: https://www.mdpi.com/2313-433X/11/9/293
