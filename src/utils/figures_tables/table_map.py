import numpy as np
import pandas as pd
import sys

scene = sys.argv[1]

df = pd.read_csv(f'../../../results/{scene}_final.csv')

v5n = df.loc[df['Arch'] == 'v5n'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v5s = df.loc[df['Arch'] == 'v5s'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v5m = df.loc[df['Arch'] == 'v5m'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v6n = df.loc[df['Arch'] == 'v6n'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v6s = df.loc[df['Arch'] == 'v6s'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v8n = df.loc[df['Arch'] == 'v8n'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v8s = df.loc[df['Arch'] == 'v8s'].sort_values(by=['Name'])[['Name', 'Last mAP']]
v8m = df.loc[df['Arch'] == 'v8m'].sort_values(by=['Name'])[['Name', 'Last mAP']]

dfs = [v5n, v5s, v5m, v6n, v6s, v8n, v8s, v8m]
method_order = ['Stock', 'ViSGA Disc', 'YOCO ROI_Disc_FM', 'YOCO TK Disc', 'ViSGA Cont', 'YOCO ROI_Contrastive_FM', 'YOCO TK Cont', 'YOCOv1', 'YOCO KMeans', 'YOCO PTAP', 'Oracle']
row_maps = []
for df in dfs:
    
    maps = []
    for n in method_order:
        maps.append(df.loc[df['Name'] == n]['Last mAP'].item())
    maps = [round(x*100,2) for x in maps]
    row_maps.append(maps)
    
combined_df = pd.DataFrame(row_maps, columns=method_order)
combined_df.index = ['5-N', '5-S', '5-M', '6-N', '6-S', '8-N', '8-S', '8-M']

table_str = '''
\\begin{table*}
\centering
\\resizebox{\\textwidth}{!}{%
\\begin{tblr}{
  width = \linewidth,
  colspec = {Q[1]Q[40]Q[60]Q[60]Q[60]Q[60]Q[60]Q[60]Q[60]Q[60]Q[60]Q[60]Q[60]},
  cells = {c},
  cell{1}{3} = {c=11}{},
  cell{2}{3} = {r=2}{},
  cell{2}{4} = {c=3}{},
  cell{2}{7} = {c=3}{},
  cell{2}{10} = {c=3}{},
  cell{2}{13} = {r=2}{},
  cell{4}{1} = {r=8}{},
  vline{3} = {4-11}{},
  vline{4,7,10,13} = {2-11}{dashed},
  hline{1} = {-}{},
  hline{3} = {4-12}{dashed},
  hline{4} = {3-13}{},
  hline{12} = {-}{},
}
 &  & \\textbf{Method} &  &  &  &  &  &  &  &  &  & \\\\
 &  & {Source\\\\Only} & Adversarial Training &  &  & Contrastive Learning &  &  & Feature Map Clustering &  &  & {Target\\\\Only}\\\\
 &  &  & ROI & FM & Top-K & ROI & FM & Top-K & YOCOv1 & KMeans & PTAP & \\\\
\\begin{sideways}\\textbf{Architecture}\end{sideways}
'''
for index, row in combined_df.iterrows():
    row_str = f'\n& {index} & '
    row = [x[1] for x in list(row.items())]
    row_sorted = sorted(row)
    uniques = np.unique(row_sorted)
    best1 = uniques[-2]
    best2 = uniques[-3]
    best3 = uniques[-4]
    
    for x in row:
        
        if x == best1:
            if len(str(x)) == 3:
                str_x = '\\phantom{0}'+str(x)
            else:
                str_x = str(x)
            row_str += '\SetCell[c=1]{c, olive9}\\textbf{'+str_x+'} & '
        elif x == best2:
            if len(str(x)) == 3:
                str_x = '\\phantom{0}'+str(x)
            else:
                str_x = str(x)
            row_str += '\SetCell[c=1]{c, yellow9}\\textbf{'+str_x+'} & '
        elif x == best3:
            if len(str(x)) == 3:
                str_x = '\\phantom{0}'+str(x)
            else:
                str_x = str(x)
            row_str += '\SetCell[c=1]{c, brown9}\\textbf{'+str_x+'} & '
        else:
            if len(str(x)) == 3:
                str_x = '\\phantom{0}'+str(x)
            else:
                str_x = str(x)
            row_str += f'{str_x} & '
    row_str = row_str[:-3] + '\\\\'
    table_str += row_str
    
table_str += '\n'
table_str += '''
\end{tblr}
}
\end{table*}
'''

print(table_str)
   