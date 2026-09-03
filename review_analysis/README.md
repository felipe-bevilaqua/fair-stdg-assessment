# review_analysis — respostas às dúvidas do revisor

Pasta autocontida: tudo o que é necessário para reproduzir as duas análises
está aqui dentro (dados, logs, script, notebooks e resultados prontos).

## Perguntas respondidas

1. **Distribuições do alvo (Y), do atributo sensível (S) e conjunta P(S, Y)**
   nos dados sintéticos, por base × gerador × fold, com a partição real de
   treinamento de cada fold como referência e média ± desvio-padrão entre folds.
   → notebook `sy_distributions.ipynb`, CSVs `outputs/sy_distributions_*.csv`,
   figuras `outputs/fig_joint_sy_<base>.png`.
2. **Tempos aproximados de treinamento e ajuste por gerador/base**
   (por trial, por fold, totais de tuning e intervalos representativos).
   → notebook `training_times.ipynb`, CSVs `outputs/training_times_*.csv`,
   figura `outputs/fig_training_times.png`.

## Conteúdo

- `compute_sy_distributions.py` — script que produz todos os CSVs e figuras
  (`python3 compute_sy_distributions.py`; requer `pandas` e `matplotlib`).
  Usa automaticamente os dados locais desta pasta quando presentes.
- `sy_distributions.ipynb`, `training_times.ipynb` — notebooks executados,
  com tabelas e figuras inline.
- `outputs/` — CSVs agregados e figuras já gerados.
- `data/synthetic_data/<base>/` — dados sintéticos usados nos experimentos:
  `<base>_<gerador>_fold_<k>_best.csv` (melhor configuração de cada gerador,
  4 bases × 6 geradores × 5 folds).
- `data/original_data/` — bases originais (adult, bank_marketing, compas, german).
- `data/fold_indexes/` — índices train/val/test de cada fold (JSON); a
  referência real é `df.loc[fold_indexes['fold_k']['train']]`.
- `configs/datasets_config.json` — definição de `target_col`/`target_value`,
  `sensitive_col`/`sensitive_value` e tipo (Protected/Privileged) por base.
- `results/from_s3/` — logs dos trials do Optuna
  (`result_trials_<base>_<gerador>_<fold>.csv`); a coluna `running_time` é o
  tempo de parede (s) de um trial = ajuste do gerador + amostragem
  (arquivos `*_error*` são ignorados).

## Convenções

Idênticas a `src/utils.run_clf` do repositório: Y = 1 sse
`target_col == target_value`; S = 1 marca o grupo **protegido**
(`col == sensitive_value` quando o tipo é Protected; `col != sensitive_value`
quando Privileged — COMPAS, protegido = não-caucasiano). Em `bank_marketing`
a coluna `age` já está binarizada nos dados originais (protegido = `age == 0`).
