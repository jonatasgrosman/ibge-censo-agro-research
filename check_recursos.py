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
from pyod.models.vae import VAE
from pyod.models.alad import ALAD


from utils import prepare_data, get_variable_encoding_map

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
    "VAE": {"type": "Neural Networks", "model_factory": lambda: VAE(device="cpu", verbose=0)},
    "ALAD": {"type": "Neural Networks", "model_factory": lambda: ALAD(device="cpu", verbose=0)},
    "R-Graph": {"type": "Graph-based", "model_factory": lambda: RGraph()},
    "LUNAR": {"type": "Graph-based", "model_factory": lambda: LUNAR()},
}


DATA_FILEPATH = 'data/a_estabel_sample.csv'
DTYPE_FILEPATH = 'data/estabel_dtypes.json'
VARIABLE_MAP_FILEPATH = 'data/variable_map.json'


def main(model_name, dataset_ratio, seed, data_filepath, dtype_filepath, variable_map_filepath):

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

    variable_map = json.load(open(variable_map_filepath))

    # Transformando o dicionário em DataFrame, excluindo as chaves especificadas
    variable_map_df = pd.DataFrame.from_dict(variable_map, orient='index')
    variable_map_df = variable_map_df.drop(columns=['condition', 'reference'])

    categorical_variables_map = get_variable_encoding_map(df, variable_map_df)
    numerical_variables_map = {
        'variables': list(df.select_dtypes(include=[np.number]).columns),
        'encoding': "minmax",
        'missing_treatment': "zero"
    }

    X, feature_names, num_features, cat_features = prepare_data(df, numerical_variables_map, categorical_variables_map, variable_map)

    n_samples = int(X.shape[0] * dataset_ratio)

    X_sample = X[np.random.RandomState(seed).choice(X.shape[0], n_samples, replace=False)]

    model = MODELS[model_name]
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
    parser.add_argument('--variable_map_filepath', type=str, default=VARIABLE_MAP_FILEPATH, help='Caminho para o arquivo JSON do mapeamento de variáveis')
    args = parser.parse_args()

    model_name = args.model
    dataset_ratio = args.dataset_ratio
    seed = args.seed
    data_filepath = args.data_filepath
    dtype_filepath = args.dtype_filepath
    variable_map = args.variable_map_filepath

    main(model_name, dataset_ratio, seed, data_filepath, dtype_filepath, variable_map)