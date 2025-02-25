import pandas as pd 
import numpy as np
from alive_progress import alive_it
import wandb
api = wandb.Api()

runs = api.runs("tjchase34/YOCOv2")

dsets = [
    ('mars1', 'mars2'),
    ('mars2', 'mars2'),
    ('asteroid1', 'asteroid3'),
    ('asteroid3', 'asteroid3'),
    ('moon64', 'moon100'),
    ('moon100', 'moon100')
]

rows = []
for run in alive_it(runs):
    name = run.name
    arch = run.config['arch']
    train_scene = run.config['train_scene']
    test_scene = run.config['test_scene']
    yoco = run.config['yoco']
    try:
        da_loss = run.config['da_loss']
    except:
        if 'Disc' in name:
            da_loss = 'disc'
        elif 'Contrastive' in name:
            da_loss = 'contrastive'
        else:
            da_loss = ''
            
    try:    
        da_type = run.config['da_type']
    except:
        if 'TK' in name:
            da_type = 'tk'
        elif 'PTAP' in name:
            da_type = 'ptap'
        elif 'YOCOv0' in name:
            da_type = 'yocov0'
        elif 'KMeans' in name:
            da_type = 'kmeans'
        elif 'ROI' in name:
            da_type = 'roi'
        else:
            da_type = ''
            
    try:
        fm = run.config['fm']
    except:
        fm = True if 'FM' in name else False  
    
    if (train_scene, test_scene) not in dsets:
        continue

    map50 = run.history()['TargetResults/all_mAP50'].tolist()
    last = round(map50[-1], 3)
    last3 = round(np.mean(map50[-3:]), 3)
    last5 = round(np.mean(map50[-5:]), 3)
    scene = ''.join([i for i in train_scene if not i.isdigit()])
    
    # Paper sys name
    if da_type == 'yocov0':
        sys_name = 'YOCOv1'
    elif train_scene == test_scene:
        sys_name = 'Oracle'
    elif da_type == 'roi' and da_loss == 'disc' and fm == False and 'TK' not in name:
        sys_name = 'ViSGA Disc'
    elif da_type == 'roi' and da_loss == 'contrastive' and fm == False and 'TK' not in name:
        sys_name = 'ViSGA Cont'
    elif yoco == False:
        sys_name = 'Stock'
    elif da_type == 'ptap':
        sys_name = 'YOCO PTAP'
    elif da_type == 'kmeans':
        sys_name = 'YOCO KMeans'
    elif 'TK' in name and da_loss == 'disc':
        sys_name = 'YOCO TK Disc'
    elif 'TK' in name and da_loss == 'contrastive':
        sys_name = 'YOCO TK Cont'
    else:
        sys_name = f'YOCO {da_type.upper()}_{da_loss.title()}'
        if fm:
            sys_name += '_FM'
    
    rows.append([sys_name, arch, scene, train_scene, test_scene, yoco, da_type, da_loss, fm, last, last3, last5, name])

mars_rows = [r for r in rows if 'mars' in r[-1]]
asteroid_rows = [r for r in rows if 'asteroid' in r[-1]]
moon_rows = [r for r in rows if 'moon' in r[-1]]
cols = ['Name', 'Arch', 'Scene', 'Train Scene', 'Test Scene', 'YOCO', 'DA Type', 'DA Loss', 'FM', 'Last mAP', 'Last3 mAP', 'Last5 mAP', 'Exp Name']
mars_df = pd.DataFrame(mars_rows, columns=cols).sort_values(by=['Arch'])
asteroid_df = pd.DataFrame(asteroid_rows, columns=cols).sort_values(by=['Arch'])
moon_df = pd.DataFrame(moon_rows, columns=cols).sort_values(by=['Arch'])

mars_df.to_csv('../../results/mars_final.csv', index=False)
asteroid_df.to_csv('../../results/asteroid_final.csv', index=False)
moon_df.to_csv('../../results/moon_final.csv', index=False)