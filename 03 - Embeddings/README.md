# 03 - Embeddings: Seleção do Modelo de Embeddings

Esta pasta contém o pipeline de seleção e geração do modelo de embeddings utilizado para separar observações do Censo Agropecuário 2017 de acordo com o contexto do trabalho. O objetivo final é identificar automaticamente observações relevantes para posterior validação humana por meio da ferramenta de anotação de dados.

---

## 📋 Conteúdo

| Notebook | Propósito | Produto Principal |
|---|---|---|
| `01_selecao_modelos.ipynb` | Avaliar múltiplos modelos de embeddings e selecionar o melhor para recuperação semântica de observações. | `data/generated/faiss_observation_index.bin`, `data/generated/faiss_observation_reference.csv`, `data/generated/selecao_modelos_metrics.json` |

---

## 🚀 Pré-requisitos

### Ambiente Python

- Python 3.10+
- Pacotes listados em `requirements.txt` na raiz do projeto (destaque para `sentence-transformers`, `torch`, `faiss-cpu`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`).
- O notebook instala automaticamente o `faiss-cpu` na primeira célula, mas recomenda-se garantir conectividade com a internet para download dos modelos do Hugging Face.
- GPUs são opcionais, mas aceleram a execução dos modelos maiores (ex.: `llama-embed-nemotron-8b`, `Qwen3-Embedding-8B`).

Instale as dependências, se ainda não o fez:

```bash
pip install -r requirements.txt
```

### Conhecimentos prévios recomendados

- **Python**: manipulação de dados com `pandas` e `numpy`.
- **Modelos de linguagem e embeddings**: noção de como embeddings representam textos em espaços vetoriais, similaridade por produto interno/cosseno.
- **FAISS**: busca aproximada por similaridade em grandes bases de vetores.
- **Scikit-learn**: métricas de classificação (F1, precisão, recall, acurácia, ROC-AUC).
- **Hugging Face**: download e uso de modelos `sentence-transformers` e transformers (ex.: `BERTimbau`).
- **JSON**: estrutura de arquivos de configuração.
- **Git**: versionar apenas os arquivos `.example.json` e nunca os `.private.json` (estes já estão no `.gitignore`).

---

## 🗂️ Dados necessários

O notebook espera um arquivo tabular (CSV ou Excel) contendo, no mínimo:

- Uma coluna de texto com as observações a serem recuperadas (padrão: `V50000000`).
- Uma ou mais colunas de classe/contexto que indiquem se a observação é relevante para validação humana.

Exemplo esperado de colunas:

```
V50000000, Contexto (Baixa Prod), Contexto (Baixo Valor), Contexto (Alto Prod), Contexto (Alto Valor), Contexto (Perda de Prod.)
```

> O arquivo `config/selecao_modelos.example.json` aponta por padrão para `../data/generated/observacoes_classificadas_exemplo.csv`, um **dataset de exemplo gerado automaticamente** com observações fictícias e colunas de contexto. Para uso em produção, substitua `input.dataset_path` no arquivo `.private.json` pelo caminho do dataset real de observações classificadas.
>
> O dataset de exemplo pode ser regenerado executando:
> ```bash
> python scripts/gerar_observacoes_classificadas_exemplo.py
> ```

---

## ⚙️ Configuração

O notebook lê configurações a partir de `config/selecao_modelos.private.json`. **Nunca edite diretamente o arquivo `config/selecao_modelos.example.json`.** Crie a versão `.private.json` correspondente e ajuste os caminhos e parâmetros conforme seu ambiente.

### Arquivos de configuração utilizados

| Configuração | Notebook que a utiliza |
|---|---|
| `config/selecao_modelos.example.json` | `01_selecao_modelos.ipynb` |

### Detalhes dos campos de configuração

- **`input`**
  - `dataset_path`: caminho do arquivo CSV ou Excel com as observações classificadas.
  - `text_column`: nome da coluna que contém o texto das observações (ex.: `V50000000`).
  - `class_columns`: lista com os nomes das colunas que representam os contextos/classes de interesse. O notebook considera uma observação como relevante (`relevante = 1`) se pelo menos uma dessas colunas estiver preenchida.

- **`split`**
  - `knowledge_frac`: fração do dataset reservada para compor a base de conhecimento (padrão: `0.7`). O restante é usado para avaliação.
  - `random_state`: semente para reprodutibilidade do sorteio (padrão: `42`).

- **`evaluation`**
  - `thresholds`: lista de thresholds de similaridade a serem avaliados (padrão: de `0.55` a `0.90`, com incremento de `0.05`).
  - `batch_size`: tamanho do lote para geração dos embeddings.
  - `plot_roc`: se `true`, exibe a curva ROC para cada modelo.
  - `plot_metrics`: se `true`, exibe gráficos de F1, acurácia, precisão e recall por threshold.

- **`models`**
  - Dicionário de modelos candidatos. Cada entrada deve conter:
    - `enabled`: se `false`, o modelo é ignorado.
    - `label`: nome amigável para exibição.
    - `type`: `sentence_transformer` ou `bertimbau`.
    - `model_name`: identificador no Hugging Face.
    - `encode_method`: método de encoding (`encode`, `encode_query`, etc.).
    - `kwargs`: argumentos passados ao encoder.
    - `sentence_transformer_kwargs`: argumentos específicos de inicialização do `SentenceTransformer` (quando aplicável).

  Modelos pré-configurados no exemplo:

  | Chave | Modelo | Tipo |
  |---|---|---|
  | `llama_embed_nemotron_8b` | `nvidia/llama-embed-nemotron-8b` | sentence_transformer |
  | `msmarco_distilbert_cos_v5` | `sentence-transformers/msmarco-distilbert-cos-v5` | sentence_transformer |
  | `bge_m3` | `BAAI/bge-m3` | sentence_transformer |
  | `cnmoro_portuguese_en_bge_m3` | `cnmoro/portuguese-en-bge-m3` | sentence_transformer |
  | `all_minilm_l6_v2` | `sentence-transformers/all-MiniLM-L6-v2` | sentence_transformer |
  | `multilingual_e5_large_instruct` | `intfloat/multilingual-e5-large-instruct` | sentence_transformer |
  | `qwen3_embedding_8b` | `Qwen/Qwen3-Embedding-8B` | sentence_transformer |
  | `bertimbau` | `neuralmind/bert-base-portuguese-cased` | bertimbau |

- **`model_execution_order`**
  - Lista com a ordem de execução dos modelos. Todas as chaves devem existir em `models`.

- **`knowledge_base`**
  - `model_key`: chave do modelo escolhido para gerar a base final FAISS.
  - `index_output_path`: caminho de saída do índice FAISS (ex.: `../data/generated/faiss_observation_index.bin`).
  - `reference_output_path`: caminho de saída do CSV de referência com os textos e metadados (ex.: `../data/generated/faiss_observation_reference.csv`).
  > A geração da base final suporta apenas modelos do tipo `sentence_transformer`.

- **`reports`**
  - `metrics_output_path`: caminho para salvar as métricas comparativas dos modelos em JSON.

> O notebook aborta a execução se o `.private.json` não for encontrado ou se campos obrigatórios estiverem ausentes.

---

## ▶️ Ordem de execução

Esta pasta contém um único notebook, portanto a ordem é:

1. **`01_selecao_modelos.ipynb`**

### O que o notebook faz?

1. **Carrega a configuração** do arquivo `config/selecao_modelos.private.json`.
2. **Lê o dataset** de observações classificadas.
3. **Preprocessa os textos**: remove pontuação, normaliza espaços, remove duplicatas e constrói uma coluna `class` a partir das colunas de contexto.
4. **Define relevância binária**: observações com pelo menos uma classe preenchida são rotuladas como relevantes (`relevante = 1`); as demais, irrelevantes (`relevante = 0`).
5. **Separa a base** em conjunto de conhecimento (`knowledge_df`, 70%) e conjunto de avaliação (`evaluation_df`, 30%).
6. **Para cada modelo configurado**:
   - Gera embeddings das observações da base de conhecimento.
   - Indexa os embeddings com FAISS (`IndexFlatIP` com normalização L2, equivalente à similaridade cosseno).
   - Busca, para cada observação de avaliação, o vizinho mais próximo na base de conhecimento.
   - Calcula métricas (F1, precisão, recall, acurácia, ROC-AUC) para cada threshold.
   - Exibe gráficos de desempenho.
7. **Gera a base final de conhecimento** com o modelo vencedor (`knowledge_base.model_key`), salvando o índice FAISS e o CSV de referência.
8. **Salva as métricas comparativas** de todos os modelos em JSON.

---

## 📦 Produtos gerados e destino

| Arquivo de saída | Uso |
|---|---|
| `data/generated/faiss_observation_index.bin` | Índice FAISS com os embeddings das observações relevantes. |
| `data/generated/faiss_observation_reference.csv` | Dataset de referência contendo os textos originais, classes e metadados. |
| `data/generated/selecao_modelos_metrics.json` | Métricas comparativas dos modelos avaliados. |

Esses artefatos podem ser consumidos por aplicações de recuperação semântica e pela ferramenta de anotação de dados para priorizar observações que necessitem de validação humana.

---

## 🛠️ Dicas e cuidados

- Sempre execute as células de dependências (imports, variáveis globais e funções) primeiro, especialmente após reiniciar o kernel do notebook.
- Modelos grandes (como `llama-embed-nemotron-8b` e `Qwen3-Embedding-8B`) exigem hardware robusto. Desabilite-os (`enabled: false`) se o ambiente não tiver memória GPU/VRAM suficiente.
- O modelo `bertimbau` é executado por um caminho customizado via `transformers`, com embedding mean-pooling. Ele pode ser mais lento que os `sentence_transformer` equivalentes.
- Certifique-se de que o dataset indicado em `input.dataset_path` contenha a coluna de texto e as colunas de contexto esperadas; caso contrário, o notebook falhará com asserções ou resultados vazios.
- Não versione arquivos `.private.json`; eles contêm caminhos e escolhas específicas do ambiente local.
