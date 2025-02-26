from ultralytics.models.yolo.detect.val import DetectionValidator
from copy import deepcopy
import glob
import wandb

def domain_eval_callback(trainer):
        
    log_dir = trainer.args.project + '/' + trainer.args.name + '/'
    args = dict(data=trainer.cmd_args.target_data, split='test', batch=9, imgsz=256, 
                project=log_dir, name='TargetResultsLatest', exist_ok=True, save=True, plots=True)
    validator = DetectionValidator(args=args)
    model = deepcopy(trainer.model)
    
    all_results = validator(model=model)    
        
    # Post-process/gather results for logging
    all_metrics = list(all_results.keys())[:-1]
    all_metrics = [l.split('/')[-1] for l in all_metrics]
    all_metrics = [l.split('(B)')[0] for l in all_metrics]
    all_metrics = [f'TargetResults/all_{l}' for l in all_metrics]
    all_vals = list(all_results.values())
    all_results = dict(zip(all_metrics, all_vals))
    
    for i, c in enumerate(validator.metrics.ap_class_index):
        class_name = validator.names[c]
        class_result = validator.metrics.class_result(i)
        class_50 = class_result[2]
        class_95 = class_result[3]
        
        all_results[f'TargetResults/{class_name}_mAP50'] = class_50
        all_results[f'TargetResults/{class_name}_mAP50-95'] = class_95
    
    # fs = glob.glob(f'{log_dir}/TargetResultsLatest/*')
    # for f in fs:
    #     name = f.split('/')[-1].split('.')[0]
    #     if trainer.cmd_args.wb: trainer.wandb_run.log({f"TargetResultsPlots/{name}": wandb.Image(f)},  step=trainer.epoch + 1)
        