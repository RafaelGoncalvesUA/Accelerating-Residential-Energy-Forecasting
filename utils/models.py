import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, Dropout, Conv1D, Lambda
from keras.callbacks import EarlyStopping
from keras.initializers import Zeros
from time import perf_counter

from utils.preprocessing import create_dataset


def tf_model(model_filename, f, train_data, test_data, model_type, lookback, out_steps, model_args, fine_tune):
    X_train = train_data.drop(
        [f"{f}_future{i}" for i in range(1, out_steps + 1)], axis=1
    )
    Y_train = train_data[[f"{f}_future{i}" for i in range(1, out_steps + 1)]]
    X_test = test_data.drop([f"{f}_future{i}" for i in range(1, out_steps + 1)], axis=1)
    Y_test = test_data[[f"{f}_future{i}" for i in range(1, out_steps + 1)]]

    num_columns = X_train.shape[1]

    if model_type == "ann":
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
                Dropout(float(model_args[0])),
                Dense(32, activation="relu"),
                Dropout(float(model_args[0])),
                Dense(out_steps, activation="linear"),
            ]
        )

    elif model_type == "lstm":
        model = Sequential(
            [
                LSTM(
                    int(model_args[0]),
                    return_sequences=False,
                    recurrent_dropout=model_args[2],
                    dropout=model_args[1],
                ),
                # we use return_sequences=False because we only want to predict the last value
                Dense(out_steps * num_columns, kernel_initializer=Zeros()),
                Reshape([out_steps, num_columns]),
            ]
        )

    elif model_type == "cnn":
        model = tf.keras.Sequential(
            [
                Lambda(lambda x: x[:, -3:, :]),
                Conv1D(int(model_args[0]), activation="relu", kernel_size=(3)),
                Dense(out_steps * num_columns, kernel_initializer=Zeros()),
                Reshape([out_steps, num_columns]),
            ]
        )

    else:
        raise ValueError("Model type not supported")

    train_time = 0.0

    # load model if exists to fine-tune
    if fine_tune != 0:  # load modelmodel_args
        print(f"Loading {model_filename}...")
        model = tf.keras.models.load_model(model_filename, safe_mode=False)

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    if fine_tune != -1:  # fit model
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
