import torch
import numpy as np
import pandas as pd
import json
import os

from calculate_loss import *
from utils import *

if __name__ == "__main__":

    ROOT_PATH = "./dataset"
    files = [file for file in os.listdir(ROOT_PATH) if file.endswith(".json")]
    TOTAL_ANGULAR_LOSS = 0.0
    TOTAL_COUNT = 0

    files_count = len(files)
    eval_angular_loss = pd.DataFrame({f"batch_{i}": [] for i in range(files_count)})
    eval_mean_angular_loss = pd.DataFrame({f"batch_{i}": [] for i in range(files_count)})

    elg_model = torch.load(
        "./models/v0.2-gaze-model-trained.pth", map_location=torch.device('cpu'))
    elg_model.eval()

    for count, file in enumerate(files):

        angular_loss_agg = []
        mean_angular_loss_agg = []

        BATCH_ANGULAR_LOSS = 0.0
        JSON_PATH = os.path.join(ROOT_PATH, file)

        with open(JSON_PATH, "r") as f:
            json_data = f.read()

        dictionary = json.loads(json_data)

        IMG_COUNT = len(dictionary.keys())
        TOTAL_COUNT += IMG_COUNT

        for i in range(IMG_COUNT):

            base = dictionary[f"{i}"]

            eye = np.array(base["image"]).astype(np.float32)

            gt_headpose = np.array(base["pose"])
            gt_gaze = np.array(base["gaze"])

            loss = calculate_loss(eye, gt_headpose, gt_gaze, elg_model)

            angular_loss_agg.append(loss)

            BATCH_ANGULAR_LOSS += loss
            TOTAL_ANGULAR_LOSS += loss

        mean_angular_loss = BATCH_ANGULAR_LOSS / IMG_COUNT
        mean_angular_loss_agg.append(mean_angular_loss)
        print(
            f"Batch {count} mean angular loss: {mean_angular_loss}")

        eval_angular_loss[f"batch_{count}"] = angular_loss_agg
        eval_mean_angular_loss[f"batch_{count}"] = mean_angular_loss_agg

    print(f"Total mean angular loss: {TOTAL_ANGULAR_LOSS / TOTAL_COUNT}")

    eval_angular_loss.to_csv("./evals/eval_angular_loss.csv", index=False)
    eval_mean_angular_loss.to_csv("./evals/eval_mean_angular_loss.csv", index=False)