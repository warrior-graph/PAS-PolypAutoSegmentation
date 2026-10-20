# PAS: Polyp Auto Segmentation

Segmentação automática de pólipos em imagens de colonoscopia.

O ponto de partida é um baseline leve e reprodutível: o encoder PVTv2 com o decoder **EMCAD** (CVPR 2024). Ele é treinado no protocolo padrão da literatura e pensado para GPUs de 4 a 8 GB e para o Kaggle.

> Protótipo de pesquisa. **Não é um dispositivo médico.**

## Início rápido

```bash
pip install -r requirements.txt
python scripts/download_data.py                       # ~700 MB -> data/TrainDataset, data/TestDataset
python train.py --config configs/emcad_b0.yaml        # treina e avalia nos 5 conjuntos de teste
python eval.py --checkpoint runs/emcad_b0/best.pth --tta
python predict.py --checkpoint runs/emcad_b0/best.pth --input minha_imagem.png --output preds/
pip install gradio && python app.py --checkpoint runs/emcad_b0/best.pth
```

- **Kaggle:** use [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb), com GPU T4 e Internet ativada.
- **Teste rápido:** `python train.py --config configs/emcad_b0.yaml --max-steps 20`.
- **Testes:** `pytest tests/`.

### Qual configuração usar em cada hardware

| Hardware | Config | Ajuste sugerido |
|---|---|---|
| GPU 4 GB | `emcad_b0.yaml` | padrão: `batch_size=8`, `grad_accum=2` |
| GPU 8 GB | `emcad_b0.yaml` / `emcad_b2.yaml` | B0 com `--set train.batch_size=16 train.grad_accum=1`; B2 no padrão |
| Kaggle T4 (16 GB) | qualquer | `--set train.batch_size=16 train.grad_accum=1` |
| DGX Spark | `emcad_b2.yaml` | batch maior; o modo bf16 é usado automaticamente |

Qualquer chave do YAML pode ser sobrescrita com `--set secao.chave=valor`.

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
| 1 | Baselines reprodutíveis (EMCAD-B0/B2, depois U-Net e PraNet), com 3 seeds | 4–8 GB / Kaggle | 🟡 EMCAD pronto para treinar |
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
  inference.py      # Predictor (checkpoint -> mapa de probabilidade) e overlay
train.py  eval.py  predict.py  app.py
configs/  scripts/download_data.py  notebooks/kaggle_train.ipynb  tests/
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
