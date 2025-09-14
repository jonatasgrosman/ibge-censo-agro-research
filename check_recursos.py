import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.rgraph import RGraph
from pyod.models.lunar import LUNAR
from pyod.models.dif import DIF

import warnings
warnings.filterwarnings('ignore')


MODELS = {
    "ECOD": {"type": "Probabilistic", "model_factory": lambda: ECOD(n_jobs=1)},
    "COPOD": {"type": "Probabilistic", "model_factory": lambda: COPOD(n_jobs=1)},
    "PCA": {"type": "Linear", "model_factory": lambda: PCA()},
    "KPCA": {"type": "Linear", "model_factory": lambda: KPCA(n_jobs=1)},
    #"ROD": {"type": "Proximity-Based", "model_factory": lambda: ROD(parallel_execution=False)},
    "HBOS": {"type": "Proximity-Based", "model_factory": lambda: HBOS()},
    "CBLOF": {"type": "Proximity-Based", "model_factory": lambda: CBLOF(n_jobs=1)},
    "IForest": {"type": "Outlier Ensembles", "model_factory": lambda: IForest(n_jobs=1)},
    "DIF": {"type": "Outlier Ensembles", "model_factory": lambda: DIF(device="cpu")},
    #"AutoEncoder": {"type": "Neural Networks", "model_factory": lambda: AutoEncoder(device="cpu")},
    #"DevNet": {"type": "Neural Networks", "model_factory": lambda: DevNet(device="cpu")},
    "R-Graph": {"type": "Graph-based", "model_factory": lambda: RGraph()},
    "LUNAR": {"type": "Graph-based", "model_factory": lambda: LUNAR()},
}


DATA_FILEPATH = 'data/a_estabel_sample.csv'
DTYPE_FILEPATH = 'data/estabel_dtypes.json'


def prepare_data(df):

    # df = 228 features
    # df_num = 77 features
    # df_cat = 151 features
    # df_tranformed = 346 features

    # Calculando o quantidade de valores nulos por coluna
    null_percentage = df.isnull().sum() / len(df) * 100
    null_percentage = null_percentage.sort_values(ascending=False)
    null_percentage = null_percentage.to_frame()
    null_percentage.columns = ['null_percentage']

    cols_to_use = null_percentage[null_percentage['null_percentage'] < 60].index.to_list()

    # Selecionando as colunas válidas (menos de 60% de nulos)
    df = df[cols_to_use].reset_index(drop=True)

    # Identifica tipos de variáveis
    numeric_features = list(df.select_dtypes(include=[np.number]).columns)
    categorical_features = list(df.select_dtypes(include=['object', 'category']).columns)

    # Separa os dados categóricos e converte para int 
    # #TODO: avaliar cada feature categorica e escolher o melhor encoding
    categorical_data = df[categorical_features]
    
    # label encoding
    # categorical_data = categorical_data.apply(LabelEncoder().fit_transform) # cuidado, pode introduzir ordem onde não existe

    #one-hot encoding
    categorical_data = pd.get_dummies(categorical_data, drop_first=True)
    
    categorical_data.fillna(-1, inplace=True)

    numerical_data = df[numeric_features]
    numerical_data.fillna(-1, inplace=True)

    # Faz o scale das colunas numéricas
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(numerical_data)

    # Concatena as colunas númericas aos dados categóricos após encoding
    X_scaled = np.concatenate((X_scaled, categorical_data.to_numpy()), axis=1)

    feature_names_transformed = scaler.get_feature_names_out().tolist() + categorical_data.columns.tolist()

    return X_scaled, feature_names_transformed, numeric_features, categorical_features


def main(model_name, dataset_ratio, seed, data_filepath, dtype_filepath):

    model = MODELS[model_name]

    dtypes = json.load(open(dtype_filepath))
    df = pd.read_csv(data_filepath, dtype=dtypes)

    # Criando o ID de região do pais
    estabel_ids = df['V010100'].astype(str)

    df['region'] = estabel_ids.str[:1].astype('Int32')
    df['uf'] = estabel_ids.str[:2].astype('Int32')
    df['city'] = estabel_ids.str[:7].astype('Int32')
    df['district'] = estabel_ids.str[:9].astype('Int32')
    df['sub_district'] = estabel_ids.str[:13].astype('Int64')

    # Removendo valores de identificadores únicos da base
    df.drop(['V010100', 'NUM_QUADRA', 'NUM_FACE', 'V010800'], axis=1, inplace=True)

    X, feature_names, num_features, cat_features = prepare_data(df)

    n_samples = int(X.shape[0] * dataset_ratio),

    X_sample = X[np.random.RandomState(seed).choice(X.shape[0], n_samples, replace=False)]

    model_instance = model["model_factory"]()

    model_instance.fit(X_sample)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument('--dataset_ratio', type=float, default=0.1, help='Proporção do dataset a ser usado no teste (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Semente para seleção aleatória de amostra do dataset')
    parser.add_argument('--data_filepath', type=str, default=DATA_FILEPATH, help='Caminho para o arquivo CSV dos dados')
    parser.add_argument('--dtype_filepath', type=str, default=DTYPE_FILEPATH, help='Caminho para o arquivo JSON dos tipos de dados')
    args = parser.parse_args()

    model_name = args.model
    dataset_ratio = args.dataset_ratio
    seed = args.seed
    data_filepath = args.data_filepath
    dtype_filepath = args.dtype_filepath

    main(model_name, dataset_ratio, seed, data_filepath, dtype_filepath)