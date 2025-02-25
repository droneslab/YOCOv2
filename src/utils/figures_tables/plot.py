import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

scene = sys.argv[1]

df = pd.read_csv(f'../../../results/{scene}_final.csv')

print(df.loc[df['Arch'] == 'v5n'].sort_values(by=['Name']))

v5n = df.loc[df['Arch'] == 'v5n'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v5s = df.loc[df['Arch'] == 'v5s'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v5m = df.loc[df['Arch'] == 'v5m'].sort_values(by=['Name'])[['Name', 'Last mAP']]

v6n = df.loc[df['Arch'] == 'v6n'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v6s = df.loc[df['Arch'] == 'v6s'].sort_values(by=['Name'])[['Name', 'Last mAP']]

v8n = df.loc[df['Arch'] == 'v8n'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v8s = df.loc[df['Arch'] == 'v8s'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v8m = df.loc[df['Arch'] == 'v8m'].sort_values(by=['Name'])[['Name', 'Last mAP']]

v5_dfs = [v5n, v5s, v5m]
arch_names = ['v5n', 'v5s', 'v5m']
maps = {
    'Source Only': [df.loc[df['Name'] == 'Stock']['Last mAP'].item() for df in v5_dfs],
    # ROI methods
    'ViSGA': [df.loc[df['Name'] == 'ViSGA Disc']['Last mAP'].item() for df in v5_dfs],
    'YOCO ROI FM': [df.loc[df['Name'] == 'YOCO ROI_Disc_FM']['Last mAP'].item() for df in v5_dfs],
    'YOCO ROI TK FM': [df.loc[df['Name'] == 'YOCO TK Disc']['Last mAP'].item() for df in v5_dfs],
    # Full map methods
    'YOCOv1': [df.loc[df['Name'] == 'YOCOv1']['Last mAP'].item() for df in v5_dfs],
    'YOCO KMeans FM': [df.loc[df['Name'] == 'YOCO KMeans']['Last mAP'].item() for df in v5_dfs],
    'YOCO PTAP FM': [df.loc[df['Name'] == 'YOCO PTAP']['Last mAP'].item() for df in v5_dfs],
    # Contrastive methods
    'ViSGA Cont.': [df.loc[df['Name'] == 'ViSGA Cont']['Last mAP'].item() for df in v5_dfs],
    'YOCO ROI Cont. FM': [df.loc[df['Name'] == 'YOCO ROI_Contrastive_FM']['Last mAP'].item() for df in v5_dfs],
    'YOCO ROI Cont. TK FM': [df.loc[df['Name'] == 'YOCO TK Cont']['Last mAP'].item() for df in v5_dfs],
    
    'Target Only': [df.loc[df['Name'] == 'Oracle']['Last mAP'].item() for df in v5_dfs],
}

x = np.arange(len(list(maps.keys())))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
i=0
for k,v in maps.items():
    ax.bar([i-0.2,i,i+0.2], v, width, edgecolor='black')
    i+=1
# ax.bar([0.8,1,1.2], maps['ViSGA'], width, edgecolor='black')


# for k,v in maps.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, v, width, label=k)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('mAP@0.50')
# ax.set_title('Penguin attributes by species')
ax.set_xticks(x, list(maps.keys()))
# ax.legend(loc='upper left', ncols=3)
ax.legend()
# ax.set_ylim(0, 250)

plt.show()
