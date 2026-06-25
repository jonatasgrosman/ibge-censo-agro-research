# 04 - AprendizadoDeMaquina: Detecção de Anomalias não Paramétrica

Esta pasta contém o pipeline de aprendizado de máquina voltado à **avaliação de algoritmos não paramétricos de detecção de anomalias** sobre dados do **Censo Agropecuário 2017 do IBGE**. O objetivo é comparar detectores baseados em árvores, proximidade, histogramas, probabilidade e redes neurais, identificando a melhor configuração para sinalizar estabelecimentos potencialmente inconsistentes.

---

## 📋 Conteúdo

| Notebook | Propósito | Produto Principal |
|---|---|---|
| `01_engenharia_features.ipynb` | Construir features privadas e datasets enriquecidos a partir do banco SQLite. | `../data/experimentos/abordagem1/*.parquet` + `engenharia_features_audit.json` |
| `02_grid_search.ipynb` | Realizar busca em grade de hiperparâmetros de detectores PyOD, maximizando Average Precision combinada. | Pasta `../data/resultados/gs_ap_combined_*` com `results_iterative.csv`, `best_by_model.csv`, scores e checkpoints |
| `03_grid_search_results_exploration.ipynb` | Re-treinar cada modelo com a melhor combinação, calcular métricas clássicas e PU, plotar curvas ROC/PR e analisar concordância. | Visualizações, ranking final e análise de consenso entre modelos |

---

## 🚀 Pré-requisitos

### Ambiente Python

- Python 3.10+
- Pacotes listados em `requirements.txt` na raiz do projeto, com destaque para:
  - `pandas`, `numpy`, `scipy`
  - `scikit-learn`
  - `pyod` (detectores de anomalias)
  - `matplotlib`, `seaborn`
  - `torch` (para AutoEncoder)
  - `pyarrow` (leitura/escrita de Parquet)

Instale as dependências, se ainda não o fez:

```bash
pip install -r requirements.txt
```

> O notebook `02_grid_search.ipynb` também pode ser executado no **Google Colab**, com salvamento iterativo dos resultados no Google Drive.

### Conhecimentos prévios recomendados

- **Python**: manipulação de dados com `pandas` e `numpy`.
- **Detecção de anomalias**: familiaridade com conceitos como *contamination*, *Average Precision* (AP), *ROC-AUC*, *precision/recall* e métricas PU.
- **PyOD**: uso de detectores como `IForest`, `HBOS`, `ECOD`, `COPOD`, `LOF`, `INNE`, `LODA`, `CBLOF` e `AutoEncoder`.
- **Scikit-learn**: `train_test_split`, `SimpleImputer`, métricas de classificação.
- **SQLite básico**: consultas em tabelas do banco gerado na etapa ETL.
- **JSON**: estrutura dos arquivos de configuração privada.
- **Git**: versionar apenas os arquivos `.example.json` e nunca os `.private.json` (estes já estão no `.gitignore`).

---

## 🗂️ Dados necessários

### Entrada obrigatória

| Origem | Arquivo / Pasta | Gerado por |
|---|---|---|
| Banco SQLite processado | `data/amstr_geral.db` | `01 - ETL/02_criando_base_dados.ipynb` |
| Mapa de variáveis | `data/mapa_variaveis.json` ou `data/mapa_variaveis_new.json` | `01 - ETL/01_mapa_variaveis.ipynb` |
| Ground truth de anomalias | `data/experimentos/ground_truth.csv` | Anotação humana (não é gerado pelos notebooks anteriores) |

### Estrutura esperada para execução

```
data/
├── amstr_geral.db                              # Banco SQLite com amostras
├── mapa_variaveis.json                         # Mapa de variáveis
└── experimentos/
    ├── ground_truth.csv                        # Labels de anomalia (is_anomaly)
    └── abordagem1/                             # Saída do notebook 01
        ├── df_estabel_final.parquet
        ├── df_lav_temp_final.parquet
        ├── df_pecu_final.parquet
        └── engenharia_features_audit.json
```

> O `ground_truth.csv` é a **única exceção** à regra de dependência entre notebooks: ele contém anotações humanas e deve ser fornecido externamente.

---

## ⚙️ Configuração

Os notebooks leem configurações a partir de arquivos JSON na pasta `config/`. **Nunca edite diretamente os arquivos `.example.json`.** Crie a versão `.private.json` correspondente e ajuste os caminhos conforme seu ambiente.

### Arquivos de configuração utilizados

| Configuração | Notebook que a utiliza |
|---|---|
| `config/engenharia_features.example.json` | `01_engenharia_features.ipynb` |

### Detalhes do arquivo de configuração

- Arquivo: `config/engenharia_features.example.json`
- Configuração do notebook: `01_engenharia_features.ipynb`
- Principais seções:

#### `io`
Configurações de entrada e saída.

| Campo | Descrição |
|---|---|
| `input_db` | Caminho do banco SQLite gerado na etapa ETL (`data/amstr_geral.db`). |
| `output_dir` | Diretório onde os arquivos Parquet e a auditoria serão salvos. |
| `audit_path` | Caminho do JSON de auditoria com resumo de linhas, colunas e nulos por dataset. |
| `variable_map_path` | Caminho do mapa de variáveis para enriquecimento das features. |
| `overwrite_outputs` | Se `true`, permite sobrescrever arquivos Parquet já existentes. |

#### `constants`
Constantes globais aplicadas durante a engenharia de features.

| Campo | Descrição |
|---|---|
| `key_columns` | Colunas de chave do estabelecimento (ex.: `["V010100", "NUM_QUADRA", "NUM_FACE", "V010800"]`). |
| `empty_category` | Valor usado para preencher categorias nulas (padrão: `"<EMPTY>"`). |
| `no_category` | Valor usado para preencher campos categóricos com resposta negativa/nula (padrão: `"1"`). |
| `flag_dtype` | Tipo das colunas de flag geradas (ex.: `"category"`). |

#### `datasets`
Lista de especificações de datasets. Cada item define:

| Campo | Descrição |
|---|---|
| `name` | Nome interno do dataset (ex.: `"estabel"`, `"lav_temp"`, `"pecu"`). |
| `table` | Nome da tabela no SQLite. |
| `output_name` | Nome de saída do dataset processado. |
| `select_columns` | Colunas a manter (vazio = todas). |
| `drop_columns` | Colunas a descartar. |
| `imputation` | Estratégia de imputação por tipo de coluna (`numeric_zero`, `numeric_median`, `categorical_empty`, `categorical_no`). |
| `rules` | Regras declarativas de transformação (ver tabela abaixo). |

#### Tipos de regras suportados (`rules`)

| `kind` | Descrição | Campos esperados |
|---|---|---|
| `ratio` | Calcula razão entre duas colunas e gera flag de threshold. | `inputs.numerator`, `inputs.denominator`, `outputs.column`, `threshold` |
| `difference` | Calcula diferença entre duas colunas. | `inputs.left`, `inputs.right`, `outputs.column` |
| `sum` | Soma múltiplas colunas. | `inputs.columns`, `outputs.column` |
| `flag` | Gera flag categórica 0/1 com base em `condition`. | `condition`, `outputs.column` |
| `lookup_flag` | Gera flag para registros cujas chaves aparecem em registros que satisfazem uma condição. | `keys`, `outputs.column` |
| `aggregate_merge` | Agrega por chaves e faz merge de volta no dataset. | `group_keys`, `aggregations`, `merge_how` |
| `merge_context` | Faz merge com outro dataset já processado. | `source_dataset`, `on_columns`, `how`, `suffixes` |
| `fallback_python` | Permite expressão Python customizada. | `fallback_python`, `outputs.column` |

> O notebook aborta a execução se o `config/engenharia_features.private.json` não for encontrado ou estiver incompleto.

---

## ▶️ Ordem de execução

Execute os notebooks na seguinte ordem:

1. **`01_engenharia_features.ipynb`**
2. **`02_grid_search.ipynb`**
3. **`03_grid_search_results_exploration.ipynb`**

### Por que respeitar a ordem?

- O notebook **01** lê `data/amstr_geral.db` e o mapa de variáveis para gerar os datasets enriquecidos em Parquet (`df_estabel_final.parquet`, `df_lav_temp_final.parquet`, `df_pecu_final.parquet`).
- O notebook **02** depende dos Parquet gerados no notebook **01** e do `ground_truth.csv` para treinar, avaliar e ranquear cada combinação de detector.
- O notebook **03** depende dos resultados do notebook **02** (`RUN_DIR/best_by_model.csv`, `results_iterative.csv`, scores e checkpoints) para re-treinar os melhores modelos e gerar análises visuais.

---

## 🤖 Modelos avaliados

O grid search considera os seguintes detectores do PyOD:

| Família | Modelo | Hiperparâmetros no grid |
|---|---|---|
| Ensemble | `IForest` | `n_estimators`, `max_samples`, `max_features`, `bootstrap` |
| Ensemble | `LODA` | `n_bins`, `n_random_cuts` |
| Ensemble | `INNE` | `n_estimators`, `max_samples` |
| Histograma | `HBOS` | `n_bins`, `alpha`, `tol` |
| Proximity | `CBLOF` | `n_clusters`, `alpha`, `beta`, `use_weights` |
| Proximity | `LOF` | `n_neighbors`, `leaf_size`, `p` |
| Probabilístico | `ECOD` | Sem grid (determinístico) |
| Probabilístico | `COPOD` | Sem grid (determinístico) |
| Neural | `AutoEncoder` | `hidden_neuron_list`, `epoch_num`, `batch_size`, `lr` |

> O `AutoEncoder` pode ser desabilitado ajustando `ENABLE_AUTOENCODER = False` no notebook **02**, caso não haja GPU ou deseje-se reduzir o tempo de execução.

---

## 📦 Produtos gerados e destino

| Arquivo de saída | Notebook produtor | Uso |
|---|---|---|
| `data/experimentos/abordagem1/df_*_final.parquet` | `01_engenharia_features.ipynb` | Datasets prontos para treinamento e avaliação. |
| `data/experimentos/abordagem1/engenharia_features_audit.json` | `01_engenharia_features.ipynb` | Auditoria do volume processado. |
| `data/resultados/gs_ap_combined_*/results_iterative.csv` | `02_grid_search.ipynb` | Resultado de cada combinação de hiperparâmetros. |
| `data/resultados/gs_ap_combined_*/best_by_model.csv` | `02_grid_search.ipynb` | Melhor combinação por modelo. |
| `data/resultados/gs_ap_combined_*/scores/*.parquet` | `02_grid_search.ipynb` | Scores por dataset para cada combinação. |
| `data/resultados/gs_ap_combined_*/combined_scores/*.parquet` | `02_grid_search.ipynb` | Scores combinados entre datasets. |
| `data/resultados/gs_ap_combined_*/checkpoint_state.json` | `02_grid_search.ipynb` | Estado de retomada do grid search. |
| Visualizações ROC/PR, histogramas e heatmaps | `03_grid_search_results_exploration.ipynb` | Análise comparativa e explicabilidade. |

---

## 🛠️ Dicas e cuidados

- Sempre execute as células de dependências (imports, variáveis globais e funções) primeiro, especialmente após reiniciar o kernel do notebook.
- O notebook `01_engenharia_features.ipynb` não expõe regras de negócio no corpo do notebook; toda a lógica sensível deve estar em `config/engenharia_features.private.json`.
- O notebook `02_grid_search.ipynb` salva resultados de forma iterativa e pode ser retomado a partir de um `RUN_DIR` existente (ajuste `CHECKPOINT_DIR` para apontar a pasta anterior).
- A métrica objetivo do grid search é o **Average Precision combinado** (`ap_combined`), calculado pelo score máximo entre os datasets por registro.
- O notebook `03_grid_search_results_exploration.ipynb` espera que o `MODEL_SPECS` definido internamente seja compatível com o grid original que gerou `best_by_model.csv`.
- Não versione arquivos `.private.json`; eles contêm caminhos e regras específicas do ambiente local.
- Para adicionar detectores customizados, edite `scripts/custom_detectors.py` (já inclui exemplos de RCF, EIF e SubspaceIForest) e registre o novo modelo no `MODEL_SPECS` do notebook.
