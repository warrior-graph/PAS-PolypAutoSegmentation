# Rodar no Kaggle (GPU T4 grátis)

O notebook [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb) faz tudo sozinho:
1. clona este repositório;
2. instala as dependências;
3. baixa os dados;
4. treina;
5. avalia nos 5 conjuntos de teste, sem TTA, com flip e com flip + multiescala;
6. empacota o checkpoint num zip.

**Tempo medido numa T4:** o B0 com 100 épocas levou ~45 min de treino e ~5 min de avaliação.

## Antes de começar

- **Telefone verificado:** vá em *Settings → Phone verification*. Sem isso, o Kaggle não libera GPU nem Internet.
- **Cota:** cerca de 30 h de GPU por semana, com no máximo 12 h por sessão.
- **Token:** o repositório é público, então não precisa. Se ele voltar a ser privado, crie o *Secret* `GITHUB_TOKEN` com um token de leitura. Use um token *fine-grained*, com *Contents: Read-only* e resource owner `warrior-graph`, e anexe o secret ao notebook em *Add-ons → Secrets*.

## Opção A: pela interface web

1. **Importar o notebook:** vá em *Create → New Notebook → File → Import Notebook*. Cole o link abaixo, ou baixe o arquivo e faça upload:
   ```
   https://raw.githubusercontent.com/warrior-graph/PAS-PolypAutoSegmentation/claude/gifted-albattani-ikhwce/notebooks/kaggle_train.ipynb
   ```
2. **Configurar a sessão:** em *Session options*, coloque *Accelerator* = **GPU T4 x2**. **Não use a P100**, que não é suportada pelas versões recentes do PyTorch. Coloque *Internet* = **On**.
3. **Ajustar os parâmetros:** edite a primeira célula de código.

   | Parâmetro | Padrão | Para que serve |
   |---|---|---|
   | `CONFIG` | `configs/emcad_b0.yaml` | `configs/emcad_b2.yaml` para o encoder B2 |
   | `EPOCHS` | `100` | número de épocas |
   | `BATCH_SIZE` | `16` | cabe na T4 para B0 e B2 em 352 px |
   | `EXTRA_SET` | `""` | overrides extras, por exemplo `"data.aug=strong data.image_size=448"` |
   | `EXP_NAME` | `""` | nome do run e do zip (vazio = nome da config) |

4. **Rodar:** use **Save Version → Save & Run All (Commit)**. Assim o notebook roda em segundo plano, mesmo com a aba fechada. Com *Run All* na sessão interativa ele para se você fechar o navegador.
5. **Resultado:** abra a versão salva. A aba **Output** tem o `<EXP>_run.zip`, com `best.pth`, `log.csv`, `test_results.md` e `eval_tta*.md`.

### Receitas da rodada 2 no Kaggle

Rode um experimento por versão, ou crie dois notebooks para rodar dois em paralelo; o Kaggle permite 2 sessões de GPU ao mesmo tempo.

| Experimento | `CONFIG` | `EPOCHS` | `BATCH_SIZE` | `EXTRA_SET` | Tempo na T4 |
|---|---|---|---|---|---|
| `e1_b0_strong` | `emcad_b0.yaml` | 100 | 16 | `data.aug=strong` | ~50 min (est.) |
| `e2_b0_448` | `emcad_b0.yaml` | 100 | 16 | `data.image_size=448` | ~1,3 h (est.) |
| `e3_b0_strong_448` | `emcad_b0.yaml` | 150 | 16 | `data.aug=strong data.image_size=448` | ~2 h (est.) |
| `e4_b2_basic` | `emcad_b2.yaml` | 100 | 16 | | ~2–2,5 h (est.) |
| `e5_b2_strong_448` | `emcad_b2.yaml` | 150 | 8 | `data.aug=strong data.image_size=448 train.grad_accum=2` | ~5–6 h (est.) |

Use `EXP_NAME` com o ID do experimento, para os zips não se confundirem. As estimativas partem do B0 medido (45 min) e escalam pelo custo de cada receita; confira no `log.csv` do primeiro run. O `e5` cabe nas 12 h, mas gasta boa parte da cota semanal. Se a T4 der falta de memória no `e5`, use `BATCH_SIZE=4` e `train.grad_accum=4`.

## Opção B: pela linha de comando (API do Kaggle)

Útil para disparar runs de outro computador e acompanhar sem abrir o navegador.

```bash
pip install kaggle
# token novo (KGAT_...): kaggle.com -> Settings -> API -> Create New Token
mkdir -p ~/.kaggle && printf '%s' 'KGAT_seu_token' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
kaggle kernels list --mine                      # confere a autenticação
```

Para baixar um notebook seu já existente, ajustar e disparar:

```bash
kaggle kernels pull <usuario>/kaggle-train -p kpush -m          # baixa o notebook + kernel-metadata.json
cp notebooks/kaggle_train.ipynb kpush/kaggle-train.ipynb         # versão atual do repo (edite os parâmetros)
```

No `kpush/kernel-metadata.json`, faça três ajustes:
- coloque `"enable_gpu": true`, `"enable_internet": true` e `"machine_shape": "NvidiaTeslaT4"`;
- **apague a linha `docker_image`**. Se ela apontar para a imagem de CPU, o run fica sem GPU;
- `"code_file"` precisa ser o nome do `.ipynb` que está na pasta.

```bash
kaggle kernels push -p kpush --accelerator NvidiaTeslaT4
kaggle kernels status <usuario>/kaggle-train                     # QUEUED -> RUNNING -> COMPLETE/ERROR
# baixar só o zip e o log (o Output inteiro é grande e a API limita requisições: erro 429)
kaggle kernels output <usuario>/kaggle-train -p resultado --file-pattern '.*_run\.zip$|.*\.log$'
```

Para um notebook novo, crie um `kernel-metadata.json` com `id` `"<usuario>/<slug>"`, `title`, `"code_file"`, `"language": "python"`, `"kernel_type": "notebook"` e `"is_private": true`, além dos campos de GPU e Internet acima.

## Problemas que já apareceram (e o que fazer)

| Sintoma no log | Causa | Solução |
|---|---|---|
| `Sem GPU: Settings -> Accelerator...` ou `nvidia-smi` sem placa | acelerador desligado ou `docker_image` de CPU fixado | escolher GPU T4; tirar o `docker_image` do metadata |
| `could not read Username for 'https://github.com'` | repositório privado sem token | tornar público ou anexar o secret `GITHUB_TOKEN` |
| `Could not resolve host` | Internet desligada | *Settings → Internet: On* (exige telefone verificado) |
| `unable to find an engine` | bf16 na T4 (já corrigido no código) | atualizar o clone (`git pull`) |
| `429 Too Many Requests` na API | limite de requisições do Kaggle | esperar 1–2 min e baixar com `--file-pattern` |
| `CUDA out of memory` | batch grande para a configuração | `BATCH_SIZE` pela metade e `train.grad_accum=2` em `EXTRA_SET` |
