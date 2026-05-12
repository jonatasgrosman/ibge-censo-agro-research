"""
Encoder customizado para detecção de anomalias em dados do Censo Agropecuário.

Este módulo fornece uma classe transformer compatível com scikit-learn que
aplica diferentes estratégias de encoding baseadas no tipo de variável,
otimizadas para tarefas de detecção de anomalias.

Autor: Sistema de Análise IBGE
Data: 2026
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer


class AnomalyDetectionEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder customizado para detecção de anomalias em dados do Censo Agropecuário.
    
    Estratégia:
    - Variáveis categóricas nominais: Frequency Encoding (frequência normalizada)
    - Variáveis binárias (0/1): Sem transformação
    - Variáveis discretas de baixa cardinalidade (2-10 valores): Sem transformação
    - Variáveis discretas de média cardinalidade (10-50 valores): RobustScaler
    - Variáveis contínuas: RobustScaler (preserva estrutura de outliers)
    
    Parâmetros
    ----------
    key_cols : list, opcional (default=None)
        Lista de colunas de chave que não devem ser transformadas.
    
    Atributos
    ---------
    column_types_ : dict
        Dicionário mapeando tipos de variáveis para listas de nomes de colunas.
        
    transformer_ : ColumnTransformer
        Transformer sklearn configurado com as estratégias apropriadas.
        
    feature_names_out_ : list
        Lista ordenada de nomes de features após transformação.
    
    Nota
    ----
    PowerTransformer foi removido pois pode "normalizar" anomalias, dificultando
    sua detecção. RobustScaler preserva a estrutura de outliers.
    
    Exemplos
    --------
    >>> from anomaly_encoder import AnomalyDetectionEncoder
    >>> 
    >>> # Criar encoder
    >>> encoder = AnomalyDetectionEncoder(key_cols=['ID', 'MUNICIPIO'])
    >>> 
    >>> # Fit apenas nos dados de treino
    >>> encoder.fit(X_train)
    >>> 
    >>> # Transform em treino e teste
    >>> X_train_encoded = encoder.transform(X_train)
    >>> X_test_encoded = encoder.transform(X_test)
    >>> 
    >>> # Visualizar transformações aplicadas
    >>> info = encoder.get_feature_info()
    >>> print(info.groupby('transformation').size())
    """
    
    def __init__(self, key_cols=None):
        self.key_cols = key_cols or []
        self.category_maps_ = {}  # Para armazenar frequency encodings
    
    def _preprocess_dataframe(self, X):
        """
        Pré-processa o dataframe para lidar com tipos especiais.
        
        Parâmetros
        ----------
        X : pd.DataFrame ou array-like
            Dados de entrada.
            
        Retorna
        -------
        pd.DataFrame
            DataFrame processado (mantém categorias para encoding posterior).
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Substituir valores infinitos por NaN em colunas numéricas
        for col in X_df.columns:
            if X_df[col].dtype.name not in ['category', 'object'] and not str(X_df[col].dtype).startswith('string'):
                X_df[col] = X_df[col].replace([np.inf, -np.inf], np.nan)
        
        return X_df
        
    def _categorize_columns(self, X):
        """
        Categoriza as colunas por tipo para aplicar transformações adequadas.
        
        Parâmetros
        ----------
        X : pd.DataFrame
            DataFrame de entrada (já pré-processado).
            
        Retorna
        -------
        dict
            Dicionário com listas de colunas por categoria:
            - categorical: Variáveis categóricas nominais (tipo 'category')
            - binary: Variáveis binárias (0/1)
            - discrete_low: Discretas baixa cardinalidade (<10 valores)
            - discrete_med: Discretas média cardinalidade (10-50 valores)
            - continuous: Contínuas (todas usam RobustScaler)
        """
        cols = [c for c in X.columns if c not in self.key_cols]
        
        categorical_cols = []
        binary_cols = []
        discrete_low_card = []
        discrete_med_card = []
        continuous_cols = []
        
        for col in cols:
            dtype = X[col].dtype
            
            # Separar categóricas nominais (tipo 'category')
            if dtype.name == 'category':
                categorical_cols.append(col)
                continue
            
            # Ignorar colunas string/object que não são numéricas
            if dtype == 'object' or str(dtype).startswith('string'):
                try:
                    pd.to_numeric(X[col])
                    continue
                except:
                    continue
            
            # Trabalhar com valores numéricos
            values = X[col]
            n_unique = values.nunique()
            n_total = values.notna().sum()
            
            # Variáveis binárias (0/1)
            unique_vals = set(values.dropna().unique())
            if n_unique == 2 and unique_vals.issubset({0, 1, 0.0, 1.0, -1}):
                binary_cols.append(col)
            
            # Variáveis discretas vs contínuas
            elif n_unique < 10:
                discrete_low_card.append(col)
            
            elif n_unique < 50 or (n_unique / n_total) < 0.01:
                discrete_med_card.append(col)
            
            else:
                # Contínua - usar RobustScaler (preserva estrutura de outliers)
                continuous_cols.append(col)
        
        return {
            'categorical': categorical_cols,
            'binary': binary_cols,
            'discrete_low': discrete_low_card,
            'discrete_med': discrete_med_card,
            'continuous': continuous_cols
        }
    
    def _get_frequency_encoding(self, series, is_fit=True):
        """
        Calcula ou aplica frequency encoding para uma série categórica.
        
        Parâmetros
        ----------
        series : pd.Series
            Série categórica a ser encoded.
        is_fit : bool, opcional (default=True)
            Se True, calcula e armazena o mapeamento de frequências.
            Se False, aplica o mapeamento previamente calculado.
            
        Retorna
        -------
        pd.Series
            Série com valores substituídos por frequências normalizadas.
        """
        col_name = series.name
        
        if is_fit:
            # Calcular frequências normalizadas
            freq_map = series.value_counts(normalize=True, dropna=False).to_dict()
            self.category_maps_[col_name] = freq_map
        
        # Aplicar mapeamento e converter para float (necessário para séries categóricas)
        encoded = series.map(self.category_maps_[col_name])
        return pd.to_numeric(encoded, errors='coerce').fillna(0.0)
    
    def fit(self, X, y=None):
        """
        Aprende os parâmetros de transformação apenas nos dados de treino.
        
        Parâmetros
        ----------
        X : pd.DataFrame ou array-like, shape (n_samples, n_features)
            Dados de treino.
            
        y : ignorado
            Não utilizado, presente por compatibilidade sklearn.
            
        Retorna
        -------
        self : AnomalyDetectionEncoder
            Encoder ajustado.
        """
        X_df = self._preprocess_dataframe(X)
        
        # Categorizar colunas
        self.column_types_ = self._categorize_columns(X_df)
        
        # Criar transformadores
        transformers = []
        
        # 1. Variáveis binárias - sem transformação
        if self.column_types_['binary']:
            transformers.append((
                'binary',
                'passthrough',
                self.column_types_['binary']
            ))
        
        # 2. Variáveis discretas de baixa cardinalidade - sem transformação
        if self.column_types_['discrete_low']:
            transformers.append((
                'discrete_low',
                'passthrough',
                self.column_types_['discrete_low']
            ))
        
        # 3. Variáveis discretas de média cardinalidade - RobustScaler
        if self.column_types_['discrete_med']:
            transformers.append((
                'discrete_med',
                RobustScaler(),
                self.column_types_['discrete_med']
            ))
        
        # 4. Variáveis contínuas - RobustScaler (preserva outliers)
        if self.column_types_['continuous']:
            transformers.append((
                'continuous',
                RobustScaler(),
                self.column_types_['continuous']
            ))
        
        # Criar ColumnTransformer
        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # Calcular frequency encoding para variáveis categóricas
        for col in self.column_types_['categorical']:
            _ = self._get_frequency_encoding(X_df[col], is_fit=True)
        
        # Fit do transformer
        self.transformer_.fit(X_df)
        
        # Guardar nomes das colunas na ordem correta
        self.feature_names_out_ = (
            self.column_types_['categorical'] +
            self.column_types_['binary'] +
            self.column_types_['discrete_low'] +
            self.column_types_['discrete_med'] +
            self.column_types_['continuous']
        )
        
        return self
    
    def transform(self, X):
        """
        Aplica as transformações aprendidas aos dados.
        
        Parâmetros
        ----------
        X : pd.DataFrame ou array-like, shape (n_samples, n_features)
            Dados a serem transformados.
            
        Retorna
        -------
        pd.DataFrame, shape (n_samples, n_features_transformed)
            Dados transformados com nomes de features preservados.
        """
        X_df = self._preprocess_dataframe(X)
        
        # Aplicar frequency encoding para variáveis categóricas
        # Construir dicionário primeiro para evitar fragmentação de memória
        categorical_dict = {
            col: self._get_frequency_encoding(X_df[col], is_fit=False)
            for col in self.column_types_['categorical']
        }
        categorical_encoded = pd.DataFrame(categorical_dict, index=X_df.index)
        
        # Aplicar transformação do transformer para as outras variáveis
        X_transformed = self.transformer_.transform(X_df)
        
        # Combinar categóricas encoded com outras transformadas
        other_encoded = pd.DataFrame(
            X_transformed,
            columns=[c for c in self.feature_names_out_ if c not in self.column_types_['categorical']],
            index=X_df.index
        )
        
        # Concatenar na ordem correta
        result = pd.concat([categorical_encoded, other_encoded], axis=1)
        result = result[self.feature_names_out_]  # Garantir ordem correta
        
        return result
    
    def get_feature_info(self):
        """
        Retorna informações sobre como cada feature foi transformada.
        
        Retorna
        -------
        pd.DataFrame
            DataFrame com colunas 'feature', 'type' e 'transformation'.
        """
        info = []
        for col_type, cols in self.column_types_.items():
            for col in cols:
                info.append({
                    'feature': col,
                    'type': col_type,
                    'transformation': self._get_transformation_name(col_type)
                })
        return pd.DataFrame(info)
    
    def _get_transformation_name(self, col_type):
        """Retorna o nome legível da transformação aplicada."""
        mapping = {
            'categorical': 'Frequency Encoding',
            'binary': 'Passthrough (sem transformação)',
            'discrete_low': 'Passthrough (baixa cardinalidade)',
            'discrete_med': 'RobustScaler',
            'continuous': 'RobustScaler'
        }
        return mapping.get(col_type, 'Unknown')


# %%

print("Hello, world!")
# %%
