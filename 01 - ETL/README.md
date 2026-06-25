# 01 - ETL: Pipeline de Extração, Transformação e Carga

Esta pasta contém o pipeline de ETL responsável por processar os metadados e amostras do **Censo Agropecuário 2017 do IBGE** e gerar os artefatos que alimentam a ferramenta de anotação de dados mantida em outro repositório.

---

## 📋 Conteúdo

| Notebook | Propósito | Produto Principal |
|---|---|---|
| `01_mapa_variaveis.ipynb` | Construir o mapa unificado de variáveis do Censo Agro. | `data/mapa_variaveis_new.json` |
| `02_criando_base_dados.ipynb` | Converter as amostras CSV em um banco SQLite otimizado. | `data/amstr_geral.db` |
| `03_mapeamento_de_criticas.ipynb` | Extrair regras de crítica dos quadros e aplicá-las aos estabelecimentos. | Tabelas `criticas` e `estabel_criticas` no SQLite |
| `04_arquivos_complementares.ipynb` | Mapear condições de exibição e estruturar o formulário por seções. | `data/generated/condicoes_exibicao.json` e `data/generated/form_structure_generated.json` |

---

## 🚀 Pré-requisitos

### Ambiente Python

- Python 3.10+
- Pacotes listados em `requirements.txt` na raiz do projeto (destaque para `pandas`, `numpy`, `sentence-transformers`, etc.).
- Instale as dependências, se ainda não o fez:

```bash
pip install -r requirements.txt
```

### Conhecimentos prévios recomendados

- **Python**: manipulação de dados com `pandas` e `numpy`.
- **SQL básico**: operações em SQLite, joins, chaves primárias.
- **JSON**: estrutura de arquivos de configuração e metadados.
- **Expressões lógicas do Censo Agro**: sintaxe de condições, variáveis derivadas (`VW...`), prefixos `X_`, operadores `&&`, `||`, `null`.
- **Estrutura do Censo Agropecuário 2017**: quadros, perguntas, críticas, dicionários de variáveis.
- **Git**: versionar apenas os arquivos `.example.json` e nunca os `.private.json` (estes já estão no `.gitignore`).

---

## 🗂️ Dados necessários

A pasta `data/` deve conter a seguinte estrutura mínima:

```
data/
├── samples/                          # Arquivos CSV de amostras do Censo Agro
│   ├── amstr_estabel.csv
│   ├── amstr_pecu.csv
│   ├── amstr_hort.csv
│   └── ...
├── dados_censo_2017/Metadata/
│   ├── dicionarios/                  # Arquivos XLS de dicionários de variáveis
│   │   └── dic_agro.json, dic_ajuda.json, ...
│   ├── filtros/
│   │   └── filtros.json              # Regras de filtro
│   ├── quadros/
│   │   └── *.json                    # Estruturas dos quadros do questionário
│   └── informacoes_adicionais/
│       └── *_novo.json               # Informações adicionais por produto
├── mapa_variaveis.json               # Benchmark de referência (não sobrescrito)
├── form_structure.json               # Estrutura de referência do formulário
└── generated/                        # Pasta para artefatos gerados
```

> Os dados brutos do **Censo Agropecuário 2017 do IBGE** são obrigatórios: dicionários de variáveis, quadros, fórmulas derivadas e amostras de estabelecimentos. Os dados utilizados na condução desse trabalho podem ser encontrados no link para a pasta no Drive, compartilhado no relatório do Produto 4.

---

## ⚙️ Configuração

Os notebooks leem configurações a partir de arquivos JSON na pasta `config/`. **Nunca edite diretamente os arquivos `.example.json`.** Crie a versão `.private.json` correspondente e ajuste os caminhos conforme seu ambiente.

### Arquivos de configuração utilizados

| Configuração | Notebook que a utiliza |
|---|---|
| `config/mapa_variaveis.example.json` | `01_mapa_variaveis.ipynb` |
| `config/mapeamento_criticas.example.json` | `03_mapeamento_de_criticas.ipynb` |
| `config/arquivos_complementares.example.json` | `04_arquivos_complementares.ipynb` |

Os notebooks `02_criando_base_dados.ipynb` não utilizam arquivo `.private.json`; suas configurações estão nas primeiras células do próprio notebook.

### Detalhes dos arquivos de configuração

- Arquivo: `config/mapa_variaveis.example.json`
- Configuração do notebook: `01_mapa_variaveis.ipynb`.
- Principais campos:
    - `path_derivadas`: caminho do arquivo XLS com as variáveis derivadas.
    - `path_dicionarios`: lista de caminhos dos arquivos XLS de dicionários de variáveis.
    - `nomes_validos_dicionario`: nomes das abas válidas nos dicionários (ex: `["Estabel", "Pecu"]`).
    - `path_filtros`: caminho do JSON de filtros.
    - `glob_quadros`: padrão glob para os arquivos JSON de quadros.
    - `path_output`: caminho de saída do mapa de variáveis gerado (ex: `../data/mapa_variaveis_new.json`).
    - `add_info_by_variable`: mapeamento opcional `código de variável de produto → caminho do arquivo *_novo.json` para enriquecer thresholds.

> O notebook aborta a execução se o `.private.json` não for encontrado.

---

- Arquivo: `config/mapeamento_criticas.example.json`
- Configuração do notebook: `03_mapeamento_de_criticas.ipynb`.
- Principais campos:
    - `input_glob`: glob dos arquivos JSON de quadros.
    - `input_db`: caminho do banco SQLite a ser atualizado.
    - `variable_map_path`: caminho do mapa de variáveis (preferencialmente gerado no notebook 01).
    - `overwrite_tables`: booleano indicando se as tabelas `criticas` e `estabel_criticas` devem ser recriadas.
    - `key_cols`: colunas chave do estabelecimento para join entre tabelas (ex: `["V010100", "NUM_QUADRA", "NUM_FACE", "V010800"]`).
    - `estabel_table`: nome da tabela base de estabelecimentos no SQLite.
    - `string_vars`: lista de variáveis que devem permanecer como string durante a avaliação das condições.
    - `product_code_by_source`: mapeamento `tabela → variável de código do produto`.

---

- Arquivo: `config/arquivos_complementares.example.json`
- Configuração do notebook: `04_arquivos_complementares.ipynb`.
- Principais campos:
    - `quadros_glob`: glob dos arquivos JSON de quadros.
    - `mapa_variaveis_path`: caminho do mapa de variáveis gerado no notebook 01.
    - `output_condicoes_path`: caminho de saída das condições de exibição.
    - `output_form_structure_path`: caminho de saída da estrutura do formulário.
    - `reference_form_structure_path`: caminho do formulário de referência (`data/form_structure.json`).
    - `question_block_keys`: chaves dos blocos de perguntas nos JSONs de quadros.
    - `condition_normalization`: substituições para normalizar expressões lógicas (ex: `&& → and`, `|| → or`, `null → None`, `X_ → ""`).
    - `classification`: configurações de palavras-chave para classificação em seções (`area`, `production`, `sells`, `information`).
    - `logging`: controle de verbosidade e exibição de duplicatas.

> Se `enforce_reference_output` estiver `true`, a saída segue estritamente o formato do arquivo de referência.

---

## ▶️ Ordem de execução

Execute os notebooks na seguinte ordem:

1. **`01_mapa_variaveis.ipynb`**
2. **`02_criando_base_dados.ipynb`**
3. **`03_mapeamento_de_criticas.ipynb`**
4. **`04_arquivos_complementares.ipynb`**

### Por que respeitar a ordem?

- O notebook **01** gera `data/mapa_variaveis_new.json`, que é entrada obrigatória para os notebooks **03** e **04**.
- O notebook **02** gera `data/amstr_geral.db`, a base de dados processada pelo notebook **03**.
- O notebook **03** depende tanto do mapa de variáveis (01) quanto do banco de dados (02).
- O notebook **04** depende do mapa de variáveis (01) e da estrutura de referência do formulário.

---

## 📦 Produtos gerados e destino

Os artefatos gerados nesta etapa são consumidos pela ferramenta de anotação de dados:

| Arquivo de saída | Notebook produtor | Uso na ferramenta de anotação |
|---|---|---|
| `data/mapa_variaveis_new.json` | `01_mapa_variaveis.ipynb` | Mapa completo de variáveis, tipos, valores possíveis e condições. |
| `data/amstr_geral.db` | `02_criando_base_dados.ipynb` | Base de dados SQLite com as amostras. |
| Tabelas `criticas` e `estabel_criticas` no SQLite | `03_mapeamento_de_criticas.ipynb` | Regras de crítica e ocorrências por estabelecimento. |
| `data/generated/condicoes_exibicao.json` | `04_arquivos_complementares.ipynb` | Condições de exibição de quadros e perguntas. |
| `data/generated/form_structure_generated.json` | `04_arquivos_complementares.ipynb` | Estrutura do formulário classificada por seções. |

---

## 🛠️ Dicas e cuidados

- Sempre execute as células de dependências (imports, variáveis globais e funções) primeiro, especialmente após reiniciar o kernel do notebook.
- O notebook `01_mapa_variaveis.ipynb` não sobrescreve o benchmark `data/mapa_variaveis.json`; ele apenas compara e exibe diferenças.
- O notebook `02_criando_base_dados.ipynb` remove automaticamente colunas sem nome (`Unnamed: *`), faz downcast de tipos numéricos e converte textos com poucos valores únicos para categoria.
- O notebook `03_mapeamento_de_criticas.ipynb` pode ser custoso computacionalmente dependendo do tamanho do banco e da complexidade das condições.
- Não versione arquivos `.private.json`; eles contêm caminhos específicos do ambiente local.
