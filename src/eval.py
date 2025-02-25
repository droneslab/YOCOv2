import sys
from pathlib import Path
import wandb
wandb.login()
from wandb import Api
wapi = Api()
from ultralytics import YOLO
from utils.general import get_input_args

# args.test_ds should point to other domain data in which we eval on
# Batch size here needs to be low-ish for some reason
def domain_eval(model, args, model_name, model_type):    
    results = model.val(data=args.test_ds, split='test', batch=9, imgsz=256, 
                        project=f'{args.logdir}/{model_name}/', name=f'results_{model_type}', 
                        exist_ok=True, save=True, plots=True)

    # Post-process/gather results for logging
    final_results = {}
    class_nums = list(results.names.keys())
    class_names = list(results.names.values())
    metric_names = list(results.results_dict.keys())[:-1]
    metric_names = [l.split('/')[-1] for l in metric_names]
    metric_names = [l.split('(B)')[0] for l in metric_names]
    metric_names = [f'{model_type}_test_metrics/{l}' for l in metric_names]
    all_vals = list(results.results_dict.values())[:-1]

    i=0
    for mname in metric_names:
        final_results[mname] = {'all': all_vals[i]}
        i+=1
        
    for c in class_nums:
        c_res = results.class_result(c)
        i=0
        for mname in metric_names:
            final_results[mname][class_names[c]] = c_res[i]
            i+=1

    return final_results


if __name__ == '__main__':
    # --- Gather arguments
    args = get_input_args(sys.argv)
    
    # run_id = wapi.runs('tjchase34', 'YOCOv2', {'config.name': args.name})[0].id
    run = wandb.init(project='YOCOv2', dir=args.name, id='nrk9gsgo', resume='must')
    run.config.update({'test_ds': args.test_ds}, allow_val_change=True)

    # --- Setup model
    mstring = f'../logs/{args.name}/weights/last.pt'
    model = YOLO(mstring)
    
    test_metrics = domain_eval(model, args)
    run.log(test_metrics)
