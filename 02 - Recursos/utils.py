import pandas as pd
import gc
import numpy as np
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   OneHotEncoder)
from sklearn.compose import ColumnTransformer


def fill_dataset_col_with_value(df: pd.DataFrame, col: str, value):
    #verificando primeiro se é uma variável categórica
    if df[col].dtype.name == 'category' and value not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories(value)

    # Preenchendo o valor
    df[col].fillna(value, inplace=True)


def apply_null_column_filter(df: pd.DataFrame, threshold: int = 0.6):
    # Calculando o quantidade de valores nulos por coluna
    null_percentage = df.isnull().sum() / len(df)
    null_percentage = null_percentage.sort_values(ascending=False)
    null_percentage = null_percentage.to_frame()
    null_percentage.columns = ['null_percentage']

    cols_to_use = null_percentage[null_percentage['null_percentage'] < threshold].index.to_list()

    # Selecionando as colunas válidas
    del null_percentage
    gc.collect()

    return df[cols_to_use].reset_index(drop=True)


def get_feature_names(preprocessor):
    feature_names = []
    categorical_features = []
    numeric_features = []

    if hasattr(preprocessor, 'transformers_'):
      for name, transformer, features in preprocessor.transformers_:
          if name == 'num':
              feature_names.extend(features)
              numeric_features.extend(features)

          elif 'cat' in name and hasattr(transformer, 'get_feature_names_out'):
              # Só obtém nomes se o transformador tiver sido ajustado
              try:
                  cat_features = transformer.get_feature_names_out(features)
                  feature_names.extend(cat_features)
                  categorical_features.extend(cat_features)
              except Exception:
                  feature_names.extend(features)
                  categorical_features.extend(features)

    return categorical_features, numeric_features, feature_names


def get_variable_encoding_map(df: pd.DataFrame, variable_map_df: pd.DataFrame):
    """
    Aqui vamos selecionar todas as variáveis categóricas, com mais de 3 valores
    possíveis e que apresentam a palavra 'classe' ou 'classificação' em seu nome.
    Essa abordagem foi definida através da exploração das variáveis, e tem como
    objetivo recuperar aquelas que apresentam valores como:
    - De x até y
    - Maior x menor y
    - Maior que x até y

    Algumas variáveis caem no filtro, mas tratam-se de variáveis nominais. Desta
    forma, iremos ignora-las de maneira dura, evitando a codificação incorreta
    dessas variáveis.
    """
    variables_to_ignore = ['VW52133400', 'VW52133401', 'VW52133402',
                          'VW88000010', 'VW88000012', 'VW88000013',
                          'VW89000003']

    ordinal_categorical_vars = \
    variable_map_df[
      (variable_map_df.dtype == 'C') &
      (variable_map_df['possible_values'].str.len() > 3) &
      (((variable_map_df['name'].str.lower().str.contains('classe'))) |
        ((variable_map_df['name'].str.lower().str.contains('classificação')))
      ) &
      ~variable_map_df.index.isin(variables_to_ignore)
    ]

    # ordinal_categorical_vars.index.to_list()
    # ordinal_categorical_vars

    """
    Aqui vamos selecionar todas as variáveis categóricas, com mais de 3 valores
    possíveis e que apresentam a palavra 'grupo' em seu nome. Essa seleção tem o
    mesmo objetivo da célula passada e aplica a mesma filtragem de variáveis
    inválidas.
    """
    variables_to_ignore = ['VW52133400']

    ordinal_categorical_vars2 = \
    variable_map_df[
      (variable_map_df.dtype == 'C') &
      (variable_map_df['possible_values'].str.len() > 3) &
      (variable_map_df['name'].str.lower().str.contains('grupo')) &
      ~variable_map_df.index.isin(variables_to_ignore)
    ]

    # ordinal_categorical_vars2.index.to_list()
    # ordinal_categorical_vars2

    ordinal_categorical_variables = ordinal_categorical_vars.index.to_list() + \
                                ordinal_categorical_vars2.index.to_list()

    # ordinal_categorical_variables

    """
    Aqui vamos selecionar as variáveis que podem ser consideradas binárias, se
    aplicarmos um tratamento correto de dados faltantes. Esse tratamento está
    relaciona a terceira classe presente no dicionário de variáveis (comumente se
    referindo a Vazio = Não se aplica).

    Para esse cenário, vamos preencher os NA com os valores mais frequentes de cada
    variável, otimizando assim a codificação das variáveis binárias.
    """
    # Variáveis que não devem utilizar a abordagem binária
    variables_to_ignore = ['VW52133403', 'VW85000011', 'V05180100', 'V05280100',
                          'V05290100', 'V32020100', 'V32030100']

    relative_binary_variables = variable_map_df[
        (variable_map_df.dtype == 'C') &
        (variable_map_df['possible_values'].str.len() == 3) &
        ~variable_map_df.index.isin(variables_to_ignore)
    ].index.to_list()

    # relative_binary_variables

    """Por último, temos as variáveis que são estritamente binárias, apresentando
    apenas 2 valores possíveis na base de dados"""
    strict_binary_variables = variable_map_df[
        (variable_map_df.dtype == 'C') &
        (variable_map_df['possible_values'].str.len() <= 2)
    ].index.to_list()

    # strict_binary_variables

    categorical_features = list(df.select_dtypes(include=['object', 'category']).columns)

    nominal_features = list(set(categorical_features) -
                            set(ordinal_categorical_variables +
                                relative_binary_variables +
                                strict_binary_variables))

    # removendo uma variável de baixa significancia, quando presente
    if 'COD_SEQ_ESPECIE' in nominal_features:
        nominal_features.remove('COD_SEQ_ESPECIE')

    return {
        'ordinal': {
            'variables': ordinal_categorical_variables,
            'encoding': 'ordinal',
            'missing_treatment': 'unknown_class'
        },
        'nominal': {
            'variables': nominal_features,
            'encoding': 'onehot',
            'missing_treatment': 'missing_class'
        },
        'relative_binary': {
            'variables': relative_binary_variables,
            'encoding': 'binary',
            'missing_treatment': 'unknown_class'
        },
        'binary': {
            'variables': strict_binary_variables,
            'encoding': 'binary',
            'missing_treatment': 'most_frequency'
        }
    }


def prepare_data(df: pd.DataFrame, numerical_encoding: dict, categorical_encoding: dict, variable_map: dict):
    """
    Prepara os dados para clustering incluindo variáveis categóricas
    """
    cols_to_use = df.columns.to_list()

    # Identifica tipos de variáveis
    has_numeric_features = df.select_dtypes(include=[np.number])\
                            .columns.shape[0] > 0
    has_categorical_features = df.select_dtypes(include=['object', 'category'])\
                                .columns.shape[0] > 0

    # Cria preprocessador
    preprocessors = []

    if has_categorical_features:
        for k, cat_encoding in categorical_encoding.items():
            variables = list(filter(lambda v: v in cols_to_use,
                                    cat_encoding['variables']))
            encoding_method = cat_encoding['encoding']
            missing_treatment = cat_encoding['missing_treatment']

            if k == 'relative_binary':
                for col in variables:
                    # Substituindo os não se aplica por não para as variáveis
                    # que fizerem sentido
                    df[col] = df[col].replace('3', '-3')

            if missing_treatment == 'drop':
                # Removendo as ocorrências de valores nulos dessas colunas
                df.dropna(subset=variables, inplace=True)
            elif missing_treatment == 'most_frequency':
                #preenchendo os valores NA com os valores mais frequentes de cada classe
                for col in variables:
                    class_to_fill = df[col].mode()[0]
                    fill_dataset_col_with_value(df, col, class_to_fill)

            elif missing_treatment == 'unknown_class':
                #preenchendo os valores NA com a classe não se aplica ou não
                #sabe dos possiveis valores. Vamos multiplicar o valor inteiro
                #da classe por -1 para distanciar das demais, evidenciando a falta
                #de informação para a variável
                for col in variables:
                    class_to_fill = variable_map[col]['possible_values'][-1]['cod']
                    if class_to_fill.lower() == 'vazio':
                        class_to_fill = int(variable_map[col]['possible_values'][-2]['cod']) * (-1)

                    fill_dataset_col_with_value(df, col, class_to_fill)
            elif missing_treatment == 'missing_class':
                #preenchendo com uma classe Missing para evitar erros de NA
                for col in variables:
                    fill_dataset_col_with_value(df, col, 'missing')

            if encoding_method == 'onehot':
                preprocessors.append(('cat', OneHotEncoder(drop='first',
                                                           sparse_output=False),
                                      variables))

            elif encoding_method == 'ordinal':
                for col in variables:
                    class_to_fill = variable_map[col]['possible_values'][-1]['cod']
                    if class_to_fill.lower() == 'vazio':
                        class_to_fill = variable_map[col]['possible_values'][-2]['cod']

                    df[col] = df[col].replace(class_to_fill, 0)
                    df[col] = df[col].astype(int)

                # Aplicando a normalização para padronizar a escala
                preprocessors.append((f'cat_{k}', MinMaxScaler(), variables))

            elif encoding_method == 'binary':
                for col in variables:
                    # O 1 significa não no dataset e 1 significa sim
                    df[col] = df[col].replace('1', 0)
                    df[col] = df[col].replace('2', 1)

                # Aplicando a normalização para padronizar a escala
                preprocessors.append((f'cat_{k}', MinMaxScaler(), variables))

    variables = numerical_encoding['variables']
    encoding_method = numerical_encoding['encoding']
    missing_treatment = numerical_encoding['missing_treatment']
    if has_numeric_features:
        # Ajustando os dados ausentes para as features numéricas
        if missing_treatment == 'mean':
            df.loc[:, variables] = df.loc[variables].fillna(df[variables].mean())
        elif missing_treatment == 'median':
            df.loc[:, variables] =  df[variables].fillna(df[variables].median())
        elif missing_treatment == 'zero':
            df.loc[:, variables] =  df[variables].fillna(0)
        elif missing_treatment == 'min':
            df.loc[:, variables] =  df[variables].fillna(df[variables].min())
        elif missing_treatment == 'max':
            df.loc[:, variables] =  df[variables].fillna(df[variables].max())
        elif missing_treatment == 'drop':
            df.dropna(subset=variables)

        if numerical_encoding == 'standard':
            preprocessors.append(('num', StandardScaler(), variables))
        elif numerical_encoding == 'minmax':
            preprocessors.append(('num', MinMaxScaler(), variables))
        elif numerical_encoding == 'robust':
            preprocessors.append(('num', RobustScaler(), variables))

    # Compilando o Pipeline de transformação
    preprocessor = ColumnTransformer(preprocessors, remainder='drop')

    # Aplica transformação
    X_scaled = preprocessor.fit_transform(df)

    # Obtém nomes das features transformadas
    categorical_features, numeric_features, feature_names_transformed = get_feature_names(preprocessor)

    return X_scaled, feature_names_transformed, numeric_features, categorical_features