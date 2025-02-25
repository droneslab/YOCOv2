import argparse
import yaml

def get_input_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', help='YOLO version (e.g., v3/v5/v8) and model size (n/s/m/l/x).', default='v5n')
    parser.add_argument('--logdir', type=str, default='../logs/', help='Logging directory for model weights, tensorboard, WandB, etc.')
    parser.add_argument('--wb', default=False, action=argparse.BooleanOptionalAction, help='Enable WandB logging.')
    parser.add_argument('--exp', type=str, default='', help='Experiment name to append to wandb save string.')
    parser.add_argument('--pretrained', default='', type=str, help='Weights to load before training.')


    # --- Train Arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--source_data', type=str, help='Labeled source data config file path.', required=True)
    parser.add_argument('--target_data', type=str, help='Unlabeled target data config file path.', required=True)
    parser.add_argument('--da_type', choices=['roi', 'sff', 'kmeans', 'ptap', 'yocov0'], default='sff', help='Type of UDA alignment.')
    parser.add_argument('--da_loss', choices=['disc', 'cont'], default='disc', help='Type of UDA loss.')
    
    
    # --- Debug/eval Arguments
    parser.add_argument('--yolo', default=False, action=argparse.BooleanOptionalAction, help='Train a non-adaptive YOLO model.')
    parser.add_argument('--no-pc', default=False, action=argparse.BooleanOptionalAction, help='Disable PC regularization.')
    parser.add_argument('--qual', default=False, action=argparse.BooleanOptionalAction, help='Run qualitative evaluation mode (no test labels).')    
    
    args = parser.parse_args()
    return args