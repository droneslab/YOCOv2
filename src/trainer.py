import os
import torch
from copy import deepcopy
from datetime import datetime, timedelta
from torch.utils.data import distributed
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.utils import check_det_dataset, PIN_MEMORY
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils import colorstr, LOGGER, RANK, __version__
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from model import YOCOModel
import dill
    
class YOCOTrainer(DetectionTrainer):
    def __init__(self, 
                 overrides=None, 
                 args=None, 
                 wandb_run=None):
        super().__init__(overrides=overrides)
        self.cmd_args = args
        self.wandb_run = wandb_run
        self.yoco = not args.yolo
        self.target_yaml = args.test_ds
        self.fm = not self.cmd_args.no_pc
    
    # Model custom implements loss function
    def get_model(self, cfg, weights, verbose=True):
        model = YOCOModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1, yoco=self.yoco, args=self.args, cmd_args=self.cmd_args)
        if weights:
            model.load(weights)
        return model
    
    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import pandas as pd  # scope for faster startup
        
        metrics = {**self.metrics, **{"fitness": self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv, on_bad_lines='skip').to_dict(orient="list").items()}
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": metrics,
            "train_results": results,
            "date": datetime.now().isoformat(),
            "version": __version__,
        }
        
        # Save last and best
        torch.save(ckpt, self.last, pickle_module=dill)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best, pickle_module=dill)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt", pickle_module=dill)
    
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        self.loss_names = ['box', 'cls', 'dfl']
        if self.yoco:
            # self.loss_names = ('box_loss', 'cls_loss', 'dfl_loss', 'Dimg_loss', 'DinstL_loss', 'DinstM_loss', 'DinstS_loss')
            self.loss_names += ['Dbf', 'Dlf', 'Dmf', 'Dsf']
            if self.fm:
                self.loss_names += ['fm']            
        prog_str = ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )
        return prog_str
