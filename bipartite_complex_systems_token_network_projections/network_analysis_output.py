import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from dotenv import load_dotenv
load_dotenv()  

# set up
path = os.environ['PROJECT_PATH']
folders = ['output_f-EMOA', 'output_f-IEMOA', 'output_f-IEMOA-EMOA', 'output_f-lp-amm', 'output_f-none', 'output_f-pcv', 'output_f-other-bridge-vesting-pcv']
filters = ['EMOA', 'IEMOA', 'IEMOA-EMOA', 'lp-amm', 'none', 'pcv', 'other-bridge-vesting-pcv']



for fil, folder in zip(filters, folders): 
    
    #load stats 
    df_stats =pd.read_csv(join(path, f'output/{folder}/stats.csv'), index_col=[0])


    ## transpose 
    df_stats_t = df_stats.transpose()



    ## progression over time 
    plt.clf()  # Clear the previous figure
    fig = plt.figure(figsize=(100, 100))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, len(df_stats_t.index)),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    fig.suptitle(f'Progression over time - {folder}')


    for ax, snapshot in zip(grid,df_stats_t.index): 

        path_img = join(path, f'output/{folder}/pics/pic_vNetwork_{snapshot}_{fil}.png')
        im = plt.imread(path_img)
        ax.imshow(im)
        ax.set_title(snapshot)


    plt.show()
    plt.savefig(f'outputs/progression_over_time/progression-{folder}.png')
    plt.savefig(join(path,f'output/{folder}/pics/progression.png'))

