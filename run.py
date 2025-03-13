import pandas as pd

import argparse
import os
from utils.vae import CustomVAE
from utils.pca import CustomPCA

from utils.preprocessing import pd_train_preprocessing, pd_test_preprocessing
from utils.models import tf_model

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_filepath", type=str, required=True, help="Input data file (.csv)")
parser.add_argument("-t", "--test_filepath", type=str, required=True, help="Test data file (.csv)")
parser.add_argument("-g", "--global_model", action="store_true", help="Global model (if True, data_filepath contains data splits)")
parser.add_argument("-f", "--fine_tune", type=int, default=0, help="Fine-tune the model (0 = No, 1 = Yes, -1 = Only test)",)
parser.add_argument("-o", "--output", type=str, required=True, help="File where the output will be appended (.csv)")
parser.add_argument("-m", "--model", type=str, required=True, help="Model")
parser.add_argument("-l", "--lookback", type=int, default=1, help="Lookback window")
parser.add_argument("-w", "--forecast_window", type=int, default=1, help="Forecast window")
parser.add_argument("-r", "--reduction", type=str, default=None, help="Dimensionality reduction method")
parser.add_argument("-a", "--model-args", type=str, default="", help="Model arguments")
args = parser.parse_args()


selected_model = args.model
LOOKBACK = args.lookback
OUT_STEPS = args.forecast_window
DIMENSIONALITY_REDUCTION = args.reduction
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

    train_data, scaler = pd_train_preprocessing(train_data, f, LOOKBACK, OUT_STEPS)
    test_data, _ = pd_test_preprocessing(test_data, f, LOOKBACK, OUT_STEPS, scaler)

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

    mse, model_name, elapsed_time = tf_model(
        model_filename, f, train_data, test_data, selected_model, LOOKBACK, OUT_STEPS, MODEL_ARGS, FINE_TUNE
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
