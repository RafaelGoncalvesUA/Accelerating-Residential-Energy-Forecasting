import os
from itertools import product
from time import perf_counter
from dotenv import load_dotenv
import requests

NUM_BUILDINGS = 6
USE_PUSHOVER = True
OUT_FILE = "out.csv"

# remove files generated by previous runs
if not os.path.exists("models"):
    print("Creating models directory")
    os.system("mkdir models")

header = (
    "mode,building,red,var,input,model,lookback,forecast,mse,train_time,inf_time,args\n"
)
with open(OUT_FILE, "a") as f:
    f.write(header)


load_dotenv()


def send_pushover_message(title, message):
    """
    Send a pushover message to the user with the given title and message.

    Parameters:
        title (str): The title of the pushover message
        message (str): The concrete message to send in the pushover notification
    """
    print("Sending pushover message:", message)

    if not USE_PUSHOVER:
        print("Pushover notifications disabled")
        return

    url = "https://api.pushover.net/1/messages.json"

    data = {
        "token": os.getenv("PUSHOVER_TOKEN"),
        "user": os.getenv("PUSHOVER_USER"),
        "device": os.getenv("PUSHOVER_DEVICE"),
        "title": title,
        "message": message,
    }

    response = requests.post(url, data=data)
    print(response.text)


def custom_error_handler(e):
    print(f"Error: {e}")
    exit(1)


def run_job(data_path, results_path, model, lb, fc, red, is_global):
    """
    Calls `run.py` to run a job with the given parameters.

    Parameters:
        data_path (tuple): The path to the training/testing data and the forecasting mode (global=-1, local=0, fine-tune=1)
        results_path (str): The path to the results file
        model (tuple): The model to use
        lb (int): The lookback parameter
        fc (int): The forecast parameter
        red (str): The reduction method to use
        is_global (bool): Whether to use the global model or not

    Returns:
        int: The exit code of the command
    """
    print(f"Running {model} with lookback {lb}, forecast {fc}, and reduction {red}")

    red_method = f"-r {red}" if red else ""
    model_args = f"-a {model[1]}" if len(model) > 1 else ""
    is_global_flag = "-g " if is_global else ""

    command = f"python3 run.py -d data/preprocessed/{data_path[0]} -t data/preprocessed/{data_path[1]} {is_global_flag} -f {data_path[2]} "
    command += (
        f"-o {results_path} -m {model[0]} -l {lb} -w {fc} {model_args} {red_method}"
    )

    print(command)
    return os.system(command)


total_start = perf_counter()

MIN_COUNTER = -1  # start from experiment 0

for mode in [("global", 0), ("global", -1), ("global", 1), ("local", 0)]:
    msg = f"Training {mode[0].capitalize()} model with mode {mode[1]}"
    print(msg)
    send_pushover_message("New benchmark", msg)

    start = perf_counter()

    if mode == ("global", 0):
        datasets = [
            ("", "b1_test.csv", 0)
        ]  # train global model from scratch (only once)
        out_file = "ignore.csv"  # do not test
        is_global = True

    elif mode == ("local", 0):
        datasets = [
            (f"b{b}_local.csv", f"b{b}_test.csv", 0)
            for b in range(1, NUM_BUILDINGS + 1)
        ]  # train local model from scratch
        out_file = OUT_FILE
        is_global = False

    elif mode[1] == -1:
        datasets = [
            ("", f"b{b}_test.csv", -1) for b in range(1, NUM_BUILDINGS + 1)
        ]  # test global model for all buildings
        out_file = OUT_FILE
        is_global = True

    elif mode[1] == 1:
        datasets = [
            (f"b{b}_local.csv", f"b{b}_test.csv", 1)
            for b in range(1, NUM_BUILDINGS + 1)
        ]  # fine tune global model for all buildings
        out_file = OUT_FILE
        is_global = (
            False  # because we are fine tuning the global model for each building
        )

    models = []

    dropout = [0.1, 0.2]
    models += [("ann", conf) for conf in dropout]

    lstm_units = [64, 128]
    dropout = [0.1, 0.2]
    recurrent_dropout = [0.1, 0.2]
    lstm_args = list(product(lstm_units, dropout, recurrent_dropout))
    models += [("lstm", f"{conf[0]},{conf[1]},{conf[2]}") for conf in lstm_args]

    cnn_units = [64]
    models += [("cnn", conf) for conf in cnn_units]

    lookback = [24, 48, 96]
    forecast = [1, 3, 5]
    reduction = [None, "pca", "vae"]

    jobs = list(product(datasets, models, lookback, forecast, reduction))

    ctr = 0
    for data_path, model, lb, fc, red in jobs:
        if ctr % 20 == 0:
            # read all lines from the csv file
            with open(OUT_FILE, "r") as f:
                body = f.readlines()[-1]
                send_pushover_message(
                    f"Job {ctr}/{len(jobs)} [{mode[0]}_{mode[1]}]", body
                )

        if ctr < MIN_COUNTER:
            ctr += 1
            continue

        if run_job(data_path, out_file, model, lb, fc, red, is_global):
            send_pushover_message("ERROR!", f"Error running job {ctr}")
            print(f"Error running job {ctr}")
            exit(1)

        ctr += 1

    end = perf_counter()
    print(f"Elapsed time: {end - start:.2f} seconds")

total_end = perf_counter()
print(f"Total elapsed time: {total_end - total_start:.2f} seconds")
send_pushover_message("DONE!", "Benchmark completed.")
