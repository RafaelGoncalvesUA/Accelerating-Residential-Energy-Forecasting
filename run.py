import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, Dropout, Conv1D, Lambda
from keras.callbacks import EarlyStopping
from keras.initializers import Zeros

import argparse
from time import perf_counter
from utils.vae import CustomVAE
from utils.pca import CustomPCA

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data_filepath", type=str, required=True, help="Input data file (.csv)"
)
parser.add_argument(
    "-t", "--test_filepath", type=str, required=True, help="Test data file (.csv)"
)
parser.add_argument(
    "-g",
    "--global_model",
    action="store_true",
    help="Global model (if True, the data_filepath is the folder containing the data splits)",
)
parser.add_argument(
    "-f",
    "--fine_tune",
    type=int,
    default=0,
    help="Fine-tune the model (0 = No, 1 = Yes, -1 = Only test)",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="File where the output will be appended (.csv)",
)
parser.add_argument("-m", "--model", type=str, required=True, help="Model")
parser.add_argument("-l", "--lookback", type=int, default=1, help="Lookback window")
parser.add_argument(
    "-w", "--forecast_window", type=int, default=1, help="Forecast window"
)
parser.add_argument(
    "-r",
    "--dimensionality_reduction",
    type=str,
    default=None,
    help="Dimensionality reduction method",
)
parser.add_argument("-a", "--model-args", type=str, default="", help="Model arguments")
args = parser.parse_args()

selected_model = args.model
LOOKBACK = args.lookback
OUT_STEPS = args.forecast_window
DIMENSIONALITY_REDUCTION = args.dimensionality_reduction
IS_GLOBAL_MODEL = args.global_model
FINE_TUNE = args.fine_tune

passed_args = args.model_args.split(",")
MODEL_ARGS = [float(a) for a in passed_args] if args.model_args else []

VARIABLES_TO_FORECAST = [
    "Equipment Electric Power (kWh)",  # "non_shiftable_load",
    "DHW Heating (kWh)",  # "dhw_demand",
    "Cooling Load (kWh)",  # "cooling_demand",
    "Solar Generation (W/kW)",  # "solar_generation",
    "Carbon Intensity (kg_CO2/kWh)",  # "carbon_intensity",
]


def create_dataset(X, Y):
    """
    Creates a TensorFlow time series dataset from the given input and output data.

    Parameters:
    X (pd.DataFrame or np.ndarray): The input features with shape (samples, lookback * features).
    Y (pd.DataFrame or np.ndarray): The target values with shape (samples, forecast).

    Returns:
    tf.data.Dataset: A TensorFlow dataset containing batches of the input and output data.
    """
    num_features = X.shape[1] // LOOKBACK

    if num_features == 0:
        X = X.values.reshape(X.shape[0], X.shape[1], 1)
    else:
        new_X = np.zeros((X.shape[0], LOOKBACK, num_features))
        for i in range(LOOKBACK):
            new_X[:, i, :] = X.iloc[:, i * num_features : (i + 1) * num_features].values
        X = new_X

    Y = Y.values.reshape(-1, OUT_STEPS, 1)

    # create batches of size 32
    ds = tf.data.Dataset.from_tensor_slices((X, Y)).batch(32)

    return ds


def feature_engineering(df, to_forecast):
    """
    Performs feature engineering on the given DataFrame by creating lag features and future target values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features and target variable.
    to_forecast (str): The name of the target variable to forecast.

    Returns:
    tuple: A tuple containing the transformed DataFrame and the original columns.
    """
    columns = df.columns

    # create lag features
    for i in range(1, LOOKBACK):
        for f in columns:
            df[f"{f}_lag{i}"] = df[f].shift(i).values

    # create yhat for the next 48 hours
    for i in range(1, OUT_STEPS + 1):
        df[f"{to_forecast}_future{i}"] = df[to_forecast].shift(-i).values

    # drop rows with NaN values
    df.dropna(inplace=True)

    df.index = pd.RangeIndex(len(df.index))

    return df, columns


def pd_train_preprocessing(df_arr, to_forecast):
    """
    Preprocesses the training data by performing feature engineering and scaling;
    Supports global datasets splitted across multiple files.

    Parameters:
    df_arr (list of pd.DataFrame): A list of DataFrames containing the training data.
    to_forecast (str): The name of the target variable to forecast.

    Returns:
    tuple: A tuple containing the preprocessed DataFrame and the scaler used for scaling.
    """
    # create a dataframe to then concatenate all the dataframes
    res = pd.DataFrame()

    for df in df_arr:
        df, columns = feature_engineering(df, to_forecast)
        res = pd.concat([res, df])

    scaler = StandardScaler()
    res = scaler.fit_transform(res)

    print(f"Feature engineering done for {to_forecast}, shape: {res.shape}")

    lag_features = [f"{f}_lag{i}" for i in range(1, LOOKBACK) for f in columns]
    future_features = [f"{to_forecast}_future{i}" for i in range(1, OUT_STEPS + 1)]

    all_columns = columns.tolist() + lag_features + future_features

    return pd.DataFrame(res, columns=all_columns), scaler


def pd_test_preprocessing(df, to_forecast, scaler=None):
    """
    Preprocesses the test data by performing feature engineering and scaling.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the test data.
    to_forecast (str): The name of the target variable to forecast.
    scaler (StandardScaler, optional): The scaler used for scaling the training data. If None, a new scaler will be created.

    Returns:
    tuple: A tuple containing the preprocessed DataFrame and the scaler used for scaling.
    """
    df, columns = feature_engineering(df, to_forecast)

    # reuse scaler if provided
    if scaler is None:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    else:
        df = scaler.transform(df)

    print(f"Feature engineering done for {to_forecast}, shape: {df.shape}")

    lag_features = [f"{f}_lag{i}" for i in range(1, LOOKBACK) for f in columns]
    future_features = [f"{to_forecast}_future{i}" for i in range(1, OUT_STEPS + 1)]

    all_columns = columns.tolist() + lag_features + future_features

    return pd.DataFrame(df, columns=all_columns), scaler


def tf_ann_model(model_filename, f, train_data, test_data):
    return tf_model(model_filename, f, train_data, test_data, "ann")


def tf_lstm_model(model_filename, f, train_data, test_data):
    return tf_model(model_filename, f, train_data, test_data, "lstm")


def tf_cnn_model(model_filename, f, train_data, test_data):
    return tf_model(model_filename, f, train_data, test_data, "cnn")


def tf_model(model_filename, f, train_data, test_data, model_type):
    X_train = train_data.drop(
        [f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)], axis=1
    )
    Y_train = train_data[[f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)]]
    X_test = test_data.drop([f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)], axis=1)
    Y_test = test_data[[f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)]]

    num_columns = X_train.shape[1]

    if model_type == "ann":
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
                Dropout(float(MODEL_ARGS[0])),
                Dense(32, activation="relu"),
                Dropout(float(MODEL_ARGS[0])),
                Dense(OUT_STEPS, activation="linear"),
            ]
        )

    elif model_type == "lstm":
        model = Sequential(
            [
                LSTM(
                    int(MODEL_ARGS[0]),
                    return_sequences=False,
                    recurrent_dropout=MODEL_ARGS[2],
                    dropout=MODEL_ARGS[1],
                ),
                # we use return_sequences=False because we only want to predict the last value
                Dense(OUT_STEPS * num_columns, kernel_initializer=Zeros()),
                Reshape([OUT_STEPS, num_columns]),
            ]
        )

    elif model_type == "cnn":
        model = tf.keras.Sequential(
            [
                Lambda(lambda x: x[:, -3:, :]),
                Conv1D(int(MODEL_ARGS[0]), activation="relu", kernel_size=(3)),
                Dense(OUT_STEPS * num_columns, kernel_initializer=Zeros()),
                Reshape([OUT_STEPS, num_columns]),
            ]
        )

    else:
        raise ValueError("Model type not supported")

    train_time = 0.0

    # load model if exists to fine-tune
    if FINE_TUNE != 0:  # load model
        print(f"Loading {model_filename}...")
        model = tf.keras.models.load_model(model_filename, safe_mode=False)

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    if FINE_TUNE != -1:  # fit model
        print("Training model...")
        callbacks = [
            EarlyStopping(
                monitor="loss",
                min_delta=0.0001,
                patience=30,
                mode="auto",
                restore_best_weights=True,
            )
        ]

        ds = create_dataset(X_train, Y_train)

        start = perf_counter()

        fit_args = {"epochs": 200, "verbose": 0, "callbacks": callbacks}

        if model_type == "ann":
            fit_args["x"] = X_train
            fit_args["y"] = Y_train
        else:
            fit_args["x"] = ds

        model.fit(**fit_args)

        train_time = perf_counter() - start

    print("Evaluating model...")

    start = perf_counter()

    mse = (
        model.evaluate(X_test, Y_test, verbose=0)[0]
        if model_type == "ann"
        else model.evaluate(create_dataset(X_test, Y_test), verbose=0)[0]
    )

    inf_time = perf_counter() - start

    print(f"MSE: {mse}")
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    return mse, model_type, (train_time, inf_time)


available_models = {
    "ann": tf_ann_model,
    "lstm": tf_lstm_model,
    "cnn": tf_cnn_model,
}

input_filename = args.data_filepath.split("/")[-1].split(".")[0]
test_filename = args.test_filepath.split("/")[-1].split(".")[0]

if IS_GLOBAL_MODEL:
    all_global_files = [f for f in os.listdir(args.data_filepath) if "global.csv" in f]
    raw_train_data = [
        pd.read_csv(f"{args.data_filepath}/{f}") for f in all_global_files
    ]
    raw_test_data = pd.read_csv(args.test_filepath)
else:
    raw_train_data = [pd.read_csv(args.data_filepath)]
    raw_test_data = pd.read_csv(args.test_filepath)

results = []

for f in VARIABLES_TO_FORECAST:
    formatted_f = f.replace("/", "_")
    red_str = DIMENSIONALITY_REDUCTION if DIMENSIONALITY_REDUCTION else "none"
    model_filename = f"models/{selected_model}_{red_str}_{formatted_f}_l{LOOKBACK}_f{OUT_STEPS}_{''.join([str(a) for a in MODEL_ARGS])}.keras"
    train_data = [df.copy() for df in raw_train_data]
    test_data = raw_test_data.copy()

    train_data, scaler = pd_train_preprocessing(train_data, f)
    test_data, _ = pd_test_preprocessing(test_data, f, scaler)

    if DIMENSIONALITY_REDUCTION == "vae":
        vae = CustomVAE(base_dir="models")

        train_data = vae.train(
            train_data,
            to_forecast=f,
            lookback=LOOKBACK,
            out_steps=OUT_STEPS,
            latent_dim=5,
            train_again=not os.path.exists(
                f'models/encoder_{f.replace("/", "")}_l{LOOKBACK}_f{OUT_STEPS}.keras'
            ),
        )

        test_data = vae.predict(test_data)

    elif DIMENSIONALITY_REDUCTION == "pca":
        pca_filename = model_filename.replace(".keras", ".pkl")

        pca = CustomPCA(pca_filename)
        train_data = pca.fit(train_data, to_forecast=f, out_steps=OUT_STEPS)

        test_data = pca.predict(test_data)

    mse, model_name, elapsed_time = available_models[selected_model](
        model_filename, f, train_data, test_data
    )

    results.append(
        {
            "mode": FINE_TUNE,
            "building": input_filename if input_filename else test_filename,
            "red": DIMENSIONALITY_REDUCTION,
            "var": f,
            "input": input_filename,
            "model": model_name,
            "lookback": LOOKBACK,
            "forecast": OUT_STEPS,
            "mse": mse,
            "train_time": elapsed_time[0],
            "inf_time": elapsed_time[1],
            "args": "".join([str(a) for a in MODEL_ARGS]),
        }
    )

results = pd.DataFrame(results)
results.to_csv(args.output, mode="a", header=False, index=False)
