# 02 - Recursos: Avaliação de Consumo de Recursos Computacionais

Esta pasta contém um experimento de **benchmarking de recursos computacionais** para algoritmos de detecção de anomalias aplicados aos dados do **Censo Agropecuário 2017 do IBGE**. O objetivo é medir, para cada detector, o **tempo de execução** e o **pico de uso de memória** em diferentes tamanhos de amostra, auxiliando na escolha de modelos viáveis para grandes volumes de dados censitários.

---

## 📋 Conteúdo

| Arquivo | Propósito | Produto Principal |
|---|---|---|
| `check_recursos.ipynb` | Notebook orquestrador que define os experimentos, executa o script auxiliar via `mprof` e plota os resultados. | `data/check_recursos_results.json` e visualizações comparativas |
| `check_recursos.py` | Script auxiliar que carrega os dados, aplica o pré-processamento (`utils.py`) e treina um detector específico. | Métricas de tempo e memória coletadas pelo notebook |

---

## 🚀 Pré-requisitos

### Ambiente Python

- Python 3.10+
- Pacotes listados em `requirements.txt` na raiz do projeto, com destaque para:
  - `pandas`, `numpy`, `scipy`
  - `scikit-learn`
  - `pyod` (detectores de anomalias)
  - `matplotlib`, `seaborn`
  - `tqdm`
  - `memory_profiler` (para medição de pico de memória via `mprof run`)

Instale as dependências, se ainda não o fez:

```bash
pip install -r requirements.txt
```

### Instalação e uso do `memory_profiler`

O notebook `check_recursos.ipynb` utiliza o comando `mprof run` para medir o pico de uso de memória de cada detector. O `mprof` faz parte do pacote `memory_profiler`, que já está listado em `requirements.txt`. Para garantir que tudo funcione corretamente:

1. **Instale o pacote** (caso ainda não tenha instalado as dependências do projeto):

```bash
pip install memory_profiler
```

2. **Verifique se o comando `mprof` está disponível no PATH**:

```bash
mprof --version
```

Se o comando não for encontrado, pode ser necessário instalar com a flag `--user` ou ativar o ambiente virtual onde o pacote foi instalado:

```bash
python -m pip install --user memory_profiler
```

3. **Como funciona a medição**:

- O notebook chama o script `check_recursos.py` via `subprocess` com o prefixo `mprof run --output=<caminho_do_log>`.
- O `mprof` registra o consumo de memória do processo ao longo do tempo e salva em `data/memory_log.txt`.
- Após a execução, o notebook lê esse log e extrai o valor máximo (`peak_memory_in_mb`) para cada experimento.

4. **Caso o `mprof` não esteja disponível**, o notebook ainda tentará executar, mas a medição de memória falhará e o campo `peak_memory_in_mb` ficará zerado.

### Conhecimentos prévios recomendados

- **Python**: manipulação de dados com `pandas` e `numpy`.
- **PyOD**: uso de detectores como `ECOD`, `COPOD`, `PCA`, `KPCA`, `HBOS`, `CBLOF`, `IForest`, `DIF`, `R-Graph`, `LUNAR`, `VAE` e `ALAD`.
- **Scikit-learn**: pré-processamento com `MinMaxScaler`.
- **Medição de recursos**: uso do `memory_profiler`/`mprof` e análise de tempo de execução.
- **Estrutura do Censo Agropecuário 2017**: variáveis de estabelecimento, identificadores geográficos e universo de variáveis do questionário.

---

## 🗂️ Dados necessários

### Entrada obrigatória

| Arquivo | Descrição | Origem |
|---|---|---|
| `data_example/samples/amstr_estabel.csv` | Amostra CSV de estabelecimentos do Censo Agro. | Dados de exemplo do repositório. |
| `data_example/mapa_variaveis.json` | Mapeamento JSON das variáveis com metadados (nome, tipo, valores possíveis). | Dados de exemplo do repositório. |
| `utils.py` | Módulo auxiliar com funções `prepare_data` e `get_variable_encoding_map`. | Deve estar no mesmo diretório do notebook/script ou no `PYTHONPATH`. |

> **Cenário de demonstração**: por padrão, o notebook e o script apontam para os arquivos de exemplo em `data_example/`. Isso permite executar o fluxo imediatamente, sem depender de dados reais do IBGE.
>
> **Cenário com dados reais**: para usar a amostra real, crie a pasta `data/` na raiz do projeto e coloque os arquivos `a_estabel_sample.csv` e `mapa_variaveis.json` (ou ajuste os caminhos nas variáveis `DATA_FILEPATH` e `VARIABLE_MAP_FILEPATH`). Os artefatos de execução (log de memória e resultados) sempre são salvos em `data/`.

### Estrutura esperada para execução

```
ibge-censo-agro-research/
├── data_example/
│   ├── samples/
│   │   └── amstr_estabel.csv       # Amostra de exemplo
│   └── mapa_variaveis.json         # Mapa de variáveis de exemplo
├── data/                           # Criada automaticamente; armazena artefatos de execução
│   ├── memory_log.txt              # Gerado pelo mprof durante a execução
│   └── check_recursos_results.json # Resultados acumulados dos experimentos
├── utils.py                        # Funções auxiliares de pré-processamento
├── check_recursos.ipynb            # Notebook orquestrador
└── check_recursos.py               # Script auxiliar executado pelo notebook
```

---

## ⚙️ Configuração

Os caminhos de entrada são definidos nas primeiras células do notebook e nas constantes do script `check_recursos.py`. Não há arquivo `.private.json` para esta etapa, mas é necessário garantir que os arquivos de entrada existam nos caminhos esperados.

### Campos configuráveis no notebook

| Variável | Descrição | Valor padrão (demonstração) |
|---|---|---|
| `DATA_FILEPATH` | Caminho do CSV de estabelecimentos. | `../data_example/samples/amstr_estabel.csv` |
| `VARIABLE_MAP_FILEPATH` | Caminho do JSON de mapeamento de variáveis. | `../data_example/mapa_variaveis.json` |
| `DATA_DIR` | Diretório para artefatos de execução. | `../data` |
| `TMP_MEMORY_LOG_PATH` | Caminho do log de memória do `mprof`. | `../data/memory_log.txt` |
| `RESULTS_PATH` | Caminho do JSON de resultados acumulados. | `../data/check_recursos_results.json` |

### Detectores avaliados (`MODELS`)

| Modelo | Família |
|---|---|
| `ECOD` | Probabilistic |
| `COPOD` | Probabilistic |
| `PCA` | Linear |
| `KPCA` | Linear |
| `HBOS` | Proximity-Based |
| `CBLOF` | Proximity-Based |
| `IForest` | Outlier Ensembles |
| `DIF` | Outlier Ensembles |
| `R-Graph` | Graph-based |
| `LUNAR` | Graph-based |
| `VAE` | Neural Networks |
| `ALAD` | Neural Networks |

> Modelos comentados no script (`ROD`, `AutoEncoder`, `DevNet`) podem ser reativados manualmente, desde que as dependências correspondentes estejam instaladas.

---

## ▶️ Execução

Esta pasta possui apenas **um notebook orquestrador** e um **script auxiliar**. O fluxo é:

1. **`check_recursos.ipynb`** executa iterativamente o script `check_recursos.py` via `subprocess`, um detector de cada vez, para cada combinação de `dataset_ratio` e `seed`.
2. **`check_recursos.py`** realiza o carregamento, pré-processamento e `fit` do detector especificado via linha de comando.

Não é necessário executar o script manualmente — ele é chamado pelo notebook. O `check_recursos.py` pode ser executado isoladamente para testes rápidos:

```bash
python check_recursos.py --model=IForest --dataset_ratio=0.1 --seed=42 \
  --data_filepath=../data_example/samples/amstr_estabel.csv \
  --variable_map_filepath=../data_example/mapa_variaveis.json
```

### Proporções de amostra e sementes

O notebook testa o produto cartesiano:

- **Modelos**: todos os 12 listados em `MODELS`.
- **Sementes**: `[2, 3, 5]`.
- **Proporções do dataset**: `[0.001, 0.01, 0.1, 0.2, 0.5, 1.0]`.

Isso resulta em até **216 execuções** independentes. O notebook pula automaticamente combinações já presentes em `data/check_recursos_results.json`, permitindo retomada segura.

---

## 📦 Produtos gerados e destino

| Arquivo de saída | Produtor | Uso |
|---|---|---|
| `data/memory_log.txt` | `mprof` (durante execução do notebook) | Log temporário de uso de memória. |
| `data/check_recursos_results.json` | `check_recursos.ipynb` | Resultados consolidados de tempo e pico de memória por experimento. |
| Gráficos de barras e linhas | `check_recursos.ipynb` | Visualização comparativa de tempo e memória por modelo e proporção. |

---

## 🛠️ Dicas e cuidados

- Certifique-se de que `memory_profiler` está instalado **e** que o comando `mprof` está disponível no terminal: `mprof --version`.
- O notebook utiliza `subprocess.check_output` com `timeout=1800` (30 minutos) por experimento. Detectores lentos ou amostras muito grandes podem estourar esse limite; ajuste conforme necessário.
- Resultados com `output_msg` não vazio indicam falha na execução do experimento. O notebook filtra esses casos nas análises, mas é importante inspecionar as mensagens de erro.
- **Amostras muito pequenas**: ao usar os dados de exemplo (`amstr_estabel.csv` com apenas 10 registros), as proporções `0.001` e `0.01` resultam em 0 amostras, e alguns detectores de deep learning (como `VAE` e `ALAD`) podem falhar mesmo em proporções maiores devido ao `batch_size` padrão. Essas falhas são esperadas em cenários de demonstração e são excluídas das visualizações.
- Para retomar uma execução interrompida, basta executar novamente o notebook: ele ignorará automaticamente os experimentos já concluídos.
- O pré-processamento depende do módulo `utils.py`. Certifique-se de que ele está acessível no mesmo diretório do notebook ou no `PYTHONPATH`.
