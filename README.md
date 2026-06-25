# ibge-censo-agro-research

Este repositório reúne o pipeline completo de pesquisa e desenvolvimento de técnicas de **aprendizado de máquina para apoio à revisão de dados do Censo Agropecuário 2017 do IBGE**. O trabalho está organizado em quatro etapas principais, que vão desde o processamento dos metadados e amostras censitárias até a avaliação de algoritmos de detecção de anomalias e modelos de embeddings semânticos.

## 🎯 Objetivo geral

O objetivo central é investigar e comparar abordagens computacionais capazes de identificar estabelecimentos e observações potencialmente inconsistentes nos dados do Censo Agro, subsidiando a revisão e anotação humana em ferramentas especializadas. Para isso, o repositório combina:

- **Processamento de metadados**: extração, transformação e carga de dicionários, quadros, filtros, variáveis derivadas e amostras.
- **Avaliação de recursos computacionais**: medição de tempo e memória de detectores de anomalias em diferentes escalas.
- **Recuperação semântica**: seleção de modelos de embeddings para busca por similaridade em observações textuais.
- **Detecção de anomalias**: engenharia de features, grid search e análise comparativa de algoritmos não paramétricos.

## 🗺️ Como navegar neste repositório

Cada pasta numerada representa uma etapa do pipeline, com seu próprio README detalhado. A ordem reflete, de forma geral, o fluxo de dependência dos dados:

1. Comece pelo **`01 - ETL`**, que processa os metadados e amostras e gera a base de dados SQLite e o mapa de variáveis.
2. O **`02 - Recursos`** pode ser executado em paralelo ou após o ETL: ele mede o consumo computacional de detectores sobre uma amostra de estabelecimentos.
3. O **`03 - Embeddings`** é independente do fluxo principal e trabalha com observações textuais classificadas.
4. O **`04 - AprendizadoDeMaquina`** consome os dados processados pelo ETL e pelo ground truth de anomalias para treinar, avaliar e comparar detectores.

> **Importante**: os notebooks de cada pasta devem ser executados na ordem indicada em seus respectivos READMEs. A maioria das etapas depende de arquivos `.private.json` criados a partir dos exemplos em `config/`. Nunca versione arquivos `.private.json` — eles contêm caminhos e regras específicas do ambiente local.

## 🚀 Primeiros passos

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Configure os arquivos `.private.json` na pasta `config/` copiando os exemplos `.example.json` correspondentes e ajustando os caminhos para o seu ambiente.

3. Execute os notebooks de cada etapa na ordem recomendada, conforme descrito nos READMEs das pastas.

## Estrutura do Projeto

### 01 - ETL

Pipeline de extração, transformação e carga de dados. Este módulo processa metadados e amostras do Censo Agro para gerar o mapa de variáveis, a base de dados SQLite com críticas aplicadas e as condições de exibição.
**Essencial para alimentar a ferramenta de anotação de dados.**

A pasta contém quatro notebooks que devem ser executados em ordem:

1. **`01_mapa_variaveis.ipynb`** — gera `data/mapa_variaveis_new.json`, unificando dicionários, filtros, quadros e variáveis derivadas.
2. **`02_criando_base_dados.ipynb`** — converte os CSVs de amostras em `data/amstr_geral.db` (SQLite), com otimização de tipos.
3. **`03_mapeamento_de_criticas.ipynb`** — extrai regras de crítica dos quadros e aplica-as aos estabelecimentos, criando as tabelas `criticas` e `estabel_criticas` no banco SQLite.
4. **`04_arquivos_complementares.ipynb`** — gera `data/generated/condicoes_exibicao.json` e `data/generated/form_structure_generated.json`, mapeando condições de exibição e estruturando o formulário por seções.

Os notebooks utilizam arquivos de configuração `.private.json` criados a partir dos exemplos em `config/`. É necessário copiar e preencher os arquivos antes da execução. A ordem dos notebooks deve ser respeitada, pois cada etapa depende dos artefatos gerados nas anteriores.

Para mais detalhes, consulte o [README da pasta 01 - ETL](./01%20-%20ETL/README.md).

### 02 - Recursos

Avaliação de **consumo computacional** (tempo de execução e pico de memória) de algoritmos de detecção de anomalias sobre uma amostra de estabelecimentos do Censo Agro. Esta etapa tem como objetivo identificar quais detectores são viáveis para grandes volumes de dados censitários.

A pasta contém:

- **`check_recursos.ipynb`** — notebook orquestrador que executa o script auxiliar iterativamente para cada detector, proporção de amostra e semente, medindo tempo e memória via `memory_profiler`/`mprof`.
- **`check_recursos.py`** — script auxiliar que carrega os dados, aplica o pré-processamento do módulo `utils.py` e treina um detector específico passado por argumento.

Os detectores avaliados incluem: `ECOD`, `COPOD`, `PCA`, `KPCA`, `HBOS`, `CBLOF`, `IForest`, `DIF`, `R-Graph`, `LUNAR`, `VAE` e `ALAD`. Os experimentos são executados nas proporções `[0.001, 0.01, 0.1, 0.2, 0.5, 1.0]` do dataset, com as sementes `[2, 3, 5]`. Os resultados são salvos em `data/check_recursos_results.json` e visualizados em gráficos comparativos.

Por padrão, o notebook e o script usam os dados de exemplo em `data_example/` (`amstr_estabel.csv`, `mapa_variaveis.json`), permitindo executar o fluxo de demonstração sem dados reais do IBGE. Para usar a amostra real, basta colocar os arquivos na pasta `data/` e ajustar os caminhos nas variáveis `DATA_FILEPATH` e `VARIABLE_MAP_FILEPATH`.

Para mais detalhes, consulte o [README da pasta 02 - Recursos](./02%20-%20Recursos/README.md).

### 03 - Embeddings

Seleção e geração do modelo de embeddings para recuperação semântica de observações do Censo Agro. Esta etapa tem como objetivo identificar observações relevantes para posterior validação humana por meio da ferramenta de anotação de dados.

A pasta contém o notebook `01_selecao_modelos.ipynb`, que:

1. Carrega um dataset de observações classificadas (configurável em `config/selecao_modelos.private.json`).
2. Avalia múltiplos modelos de embeddings (ex.: `bge-m3`, `all-MiniLM-L6-v2`, `BERTimbau`, `multilingual-e5-large-instruct`, entre outros) usando FAISS e métricas de classificação.
3. Gera a base final de conhecimento com o modelo escolhido (`data/generated/faiss_observation_index.bin` e `data/generated/faiss_observation_reference.csv`).
4. Exporta métricas comparativas dos modelos (`data/generated/selecao_modelos_metrics.json`).

Os notebooks desta etapa dependem de configuração em `config/selecao_modelos.private.json`, criado a partir de `config/selecao_modelos.example.json`. É necessário ajustar os caminhos e colunas de entrada conforme o dataset de observações classificadas disponível.

Para mais detalhes, consulte o [README da pasta 03 - Embeddings](./03%20-%20Embeddings/README.md).

### 04 - AprendizadoDeMaquina

Avaliação de algoritmos **não paramétricos de detecção de anomalias** sobre os dados processados nas etapas anteriores. Esta etapa tem como objetivo comparar detectores do ecossistema PyOD e identificar configurações capazes de sinalizar estabelecimentos potencialmente inconsistentes.

A pasta contém três notebooks que devem ser executados em ordem:

1. **`01_engenharia_features.ipynb`** — lê `data/amstr_geral.db` e o mapa de variáveis, aplica regras declarativas definidas em `config/engenharia_features.private.json` e gera datasets enriquecidos em Parquet (`data/experimentos/abordagem1/df_*_final.parquet`).
2. **`02_grid_search.ipynb`** — executa busca em grade de hiperparâmetros sobre detectores PyOD (`IForest`, `HBOS`, `ECOD`, `COPOD`, `LOF`, `INNE`, `LODA`, `CBLOF` e `AutoEncoder`), maximizando a Average Precision combinada entre datasets. Salva resultados iterativamente em `data/resultados/gs_ap_combined_*`.
3. **`03_grid_search_results_exploration.ipynb`** — re-treina cada modelo com a melhor combinação encontrada, calcula métricas clássicas e PU, gera curvas ROC/PR, histogramas de scores e analisa concordância entre modelos.

Os notebooks desta etapa dependem dos artefatos gerados pelos notebooks do `01 - ETL`, exceto pelo `ground_truth.csv`, que contém as anotações humanas de anomalia e deve ser fornecido externamente. O notebook `01` requer configuração em `config/engenharia_features.private.json`, criado a partir de `config/engenharia_features.example.json`.

Para mais detalhes, consulte o [README da pasta 04 - AprendizadoDeMaquina](./04%20-%20AprendizadoDeMaquina/README.md).

