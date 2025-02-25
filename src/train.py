import sys
import shutil
from utils.general import get_input_args
from trainer import YOCOTrainer
from callbacks import domain_eval_callback

# --- Gather arguments
args = get_input_args(sys.argv)

# --- Set model name for experiment logging
mtype = 'yoco' if args.yoco else 'yolo'
name = f'{mtype}{args.arch}'
train_scene_name = args.train_ds.split('/')[-1].split('.')[0]
test_scene_name   = args.test_ds.split('/')[-1].split('.')[0]
model_name = f'{name}_{train_scene_name}->{test_scene_name}'
if args.exp:
    model_name += f'_{args.exp}'
args.model_name = model_name

# --- Logging dir
save_dir=f'{args.logdir}/{model_name}/'

# --- Setup model config
if args.pretrained:
    shutil.copyfile(args.pretrained, './last.pt')
    mstring = './last.pt'
else:
    mstring = f'yolo{args.arch}.yaml'
    
# --- Setup W&B
if args.nowb:
    import wandb
    wandb.login()
    config = vars(args)
    config['train_scene'] = train_scene_name
    config['test_scene']  = test_scene_name
    config['scene'] = train_scene_name
    run = wandb.init(
            project='YOCOv2',
            dir=model_name,
            name=model_name,
            config=config
        )
else:
    run = None

# --- Set Ultralytics trainer arguments, instantiate trainer, train
# https://docs.ultralytics.com/modes/train/#train-settings
train_args = dict(
    model=mstring, data=args.train_ds, epochs=args.epochs, batch=args.batch_size, imgsz=256, device=0,
    project=args.logdir, name=model_name, exist_ok=True, save=True, save_dir=save_dir, val=False, 
    pretrained=False, plots=False, deterministic=False, amp=False, optimizer='Adam', cache=False,
    verbose=True, fraction=1.0, close_mosaic=0, freeze=None, warmup_epochs=0, box=7.5, cls=0.5, dfl=1.5,
)

# Target domain data is 'train' split from test images
trainer = YOCOTrainer(overrides=train_args, args=args, wandb_run=run)
if not args.qual:
    trainer.add_callback('on_train_epoch_end', domain_eval_callback)
trainer.train()
