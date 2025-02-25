import pandas as pd 
import numpy as np
from alive_progress import alive_it
import wandb
api = wandb.Api()

runs = api.runs("tjchase34/YOCOv2", filters={"config.model_name": "yolov5n_asteroid3->asteroid3"})
for r in runs:
    imgs = r.history()['TargetResultsPlots/val_batch0_pred'].tolist()
    for img in imgs:
        r.file(img['path']).download()
