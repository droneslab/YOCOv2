import sys
import shutil
from utils.general import get_input_args
from trainer import YOCOTrainer
from callbacks import domain_eval_callback

# --- Gather arguments
args = get_input_args(sys.argv)

# --- Set model name for experiment logging
mtype = 'yoco' if not args.yolo else 'yolo'
name = f'{mtype}{args.arch}'
train_scene_name = args.source_data.split('/')[-1].split('.')[0]
test_scene_name   = args.target_data.split('/')[-1].split('.')[0]
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

# --- Set Ultralytics trainer arguments, instantiate trainer, train
# https://docs.ultralytics.com/modes/train/#train-settings
train_args = dict(
    model=mstring, data=args.source_data, epochs=args.epochs, batch=args.batch_size, imgsz=256, device=0,
    project=args.logdir, name=model_name, exist_ok=True, save=True, save_dir=save_dir, val=False, 
    pretrained=False, plots=False, deterministic=False, amp=False, optimizer='Adam', cache=False,
    verbose=True, fraction=1.0, close_mosaic=0, freeze=None, warmup_epochs=0, box=7.5, cls=0.5, dfl=1.5,
)

# Target domain data is 'train' split from test images
trainer = YOCOTrainer(overrides=train_args, args=args)
trainer.train()
