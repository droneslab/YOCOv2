'''
TensorFlow Lite Model Maker
    - https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_efficientdet_model_maker_tf2.ipynb
'''

import sys
from tflite_model_maker.config import QuantizationConfig, ExportFormat
from tflite_model_maker import object_detector
import wandb
import subprocess

''' TODO
2.) wandb logs
4.) TPU eval/log
'''

dpath = sys.argv[1]
train_set = sys.argv[2]
test_set = sys.argv[3]

train_dpath = f'{dpath}/{train_set}/'
test_dpath = f'{dpath}/{test_set}/'

mname = f'efficientdet_lite0_{train_set}->{test_set}'
logdir = f'../logs/{mname}/'

wandb.login()
config = {}
config['train_scene'] = train_set
config['test_scene']  = test_set
config['epochs'] = 50
config['batch_size'] = 32
config['pretrained'] = True
if 'moon' in train_set:
    scene = train_set[:-2]
else:
    scene = train_set[:-1]
config['scene'] = scene
run = wandb.init(
        project='YOCOv2',
        dir=logdir,
        name=mname,
        config=config
    )

if 'mars' in dpath:
    label_map = ['1','2','3']
else:
    label_map = ['1']

train_data = object_detector.DataLoader.from_pascal_voc(
    f'{train_dpath}/images/train',
    f'{train_dpath}/labels/train/voc', 
    label_map=label_map,
    cache_dir=f'{train_dpath}/tfrecords/'
)

validation_data = object_detector.DataLoader.from_pascal_voc(
    f'{train_dpath}/images/val',
    f'{train_dpath}/labels/val/voc', 
    label_map=label_map,
    cache_dir=f'{train_dpath}/tfrecords/'
)

test_data = object_detector.DataLoader.from_pascal_voc(
    f'{test_dpath}/images/test',
    f'{test_dpath}/labels/test/voc',  
    label_map=label_map,
    cache_dir=f'{test_dpath}/tfrecords/'
)

# Lite0 has 320x320 image size
spec = object_detector.EfficientDetLite0Spec(hparams={'max_instances_per_image': 100000})

model = object_detector.create(
    train_data=train_data, 
    model_spec=spec, 
    validation_data=validation_data, 
    epochs=50, 
    batch_size=32,
    train_whole_model=True
)

# Export full model
# print("\n\n    --- Saving full model\n")
# model.export(logdir, export_format=[ExportFormat.SAVED_MODEL])

# Eval full model
print("\n\n    --- Eval full model\n")
full_res = model.evaluate(test_data)
for key in list(full_res.keys()):
    wandb.log({f"full_test_metrics/{key}": full_res[key]})

# Quantize and export TF-Lite
# print("\n\n    --- Quantizing/saving TF-Lite\n")
# qconfig = QuantizationConfig.for_int8(test_data)
# model.export(export_dir=logdir, tflite_filename='last_int8.tflite', quantization_config=qconfig)

# # Eval TF-Lite
# print("\n\n    --- Eval TF-Lite model\n")
# print(model.evaluate_tflite(f'{logdir}/last_int8.tflite', test_data))

# # Compile for Edge TPU
# print("\n\n    --- Compiling/saving Edge TPU\n")
# subprocess.run(["edgetpu_compiler", "-sa", "-o", f"{logdir}/", f"{logdir}/last_int8.tflite"])

# keras_model = model.create_model()
# keras_model.build((None, 320, 320, 3))

# from keras.layers import Input
# inp = Input((320, 320, 3))
# print(keras_model(inp))

# # Eval TF-Lite model
# model.export(export_dir='.')
# tflite_results = 