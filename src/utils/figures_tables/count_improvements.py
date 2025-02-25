import pandas as pd
import numpy as np
import sys

scene = sys.argv[1]

df = pd.read_csv(f'../../../results/{scene}_final.csv')

yoco_discs = ['YOCO KMeans', 'YOCO PTAP', 'YOCO ROI_Disc_FM', 'YOCO TK Disc', 'YOCOv1']
yoco_conts = ['YOCO ROI_Contrastive_FM', 'YOCO TK Cont']
disc_counts = {'Last mAP': [], 'Last3 mAP': [], 'Last5 mAP': []}
cont_counts = {'Last mAP': [], 'Last3 mAP': [], 'Last5 mAP': []}
disc_imps = {'Last mAP': [], 'Last3 mAP': [], 'Last5 mAP': []}
cont_imps = {'Last mAP': [], 'Last3 mAP': [], 'Last5 mAP': []}
for arch in ['v5n', 'v5s', 'v5m', 'v6n', 'v6s', 'v8n', 'v8s', 'v8m']:
    df_arch = df.loc[df['Arch'] == arch].sort_values(by=['Name'])
    # for col in ['Last mAP', 'Last3 mAP', 'Last5 mAP']:
    for col in ['Last mAP']:
        visga_disc_map = df_arch.loc[df['Name'] == 'ViSGA Disc'][col].item()
        visga_cont_map = df_arch.loc[df['Name'] == 'ViSGA Cont'][col].item()
        
        disc_count = 0
        cont_count = 0
        di = []
        ci = []
        for n in yoco_discs:
            if df_arch.loc[df['Name'] == n][col].item() > visga_disc_map:
                disc_count += 1
                di.append(n)
        for n in yoco_conts:
            if df_arch.loc[df['Name'] == n][col].item() > visga_cont_map:
                cont_count += 1
                ci.append(n)
                
        disc_counts[col].append(disc_count)
        cont_counts[col].append(cont_count)
        disc_imps[col] = di
        cont_imps[col] = ci
        
        print(f'{arch} {col}: disc {disc_count}/5 ({di}), cont {cont_count}/2 ({ci})')
    
    # print('-----------------------------------')

# print(disc_counts)
# print(cont_counts)

# print('DISCs')
# for k,v in disc_counts.items():
#     print(k,np.sum(v))
    
# print('CONTs')
# for k,v in cont_counts.items():
#     print(k,np.sum(v))