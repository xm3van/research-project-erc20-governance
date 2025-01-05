import pandas as pd
import numpy as np
import ast
import json

import os
from os.path import join
import sys 
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import pearsonr
from src.utilities.metrics_and_tests import pval_to_significance

### GLOBAL VARIABLES 
FONT_SIZE_TEXT = 16
FONT_SIZE_LABEL = 20
FONT_SIZE_TITLE = 24
FONT_SIZE_VALUE = 12
FONT_SIZE_TICK = 16
LINE_SPACING=1.5
FIG_SIZE = (25, 20)
COLORS = ['white', 'black']
COLORMAP = 'CMRmap'


# Global Style Settings
plt.rcParams.update({
    # 'font.size': 16,
    # 'figure.figsize': (25, 20),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    # 'axes.labelsize': 20,
    # 'axes.titlesize': 24,
    # 'xtick.labelsize': 16,
    # 'ytick.labelsize': 16,
    # 'legend.fontsize': 12,
    # 'lines.linewidth': 2,
    # 'axes.prop_cycle': plt.cycler(color=plt.cm.magma(np.linspace(0, 1, 10)))
})

#####################################
########### Link Size #############
#####################################

def plot_link_size_over_time(metric_dataframes, group='sample', output_path="output/links/", save=True, show=True):
    
    # Define df 
    df = metric_dataframes[group]['size']

    # common index
    df_index = metric_dataframes['sample']['size']

    # Find the index of the first occurrence (value > 1) in each column (snapshot)
    first_occurrence_indices = (df_index.T > 1).idxmax()

    # Determine the minimum index (earliest occurrence) for each link across all snapshots
    min_indices = first_occurrence_indices.groupby(first_occurrence_indices.index).min()

    # Sort the links based on their minimum indices to get the desired order
    links_order = min_indices.sort_values().index.tolist()

    # Reindex link size to df_presence 
    df = df.reindex(links_order)

    # Create a mask for cells where 'link_size' is greater than 1
    mask = df.values >= 1

    # Create a custom colormap with gray for values > 1 and white for values <= 1
    cmap = mcolors.ListedColormap(COLORS)

    # Create the binary chart with custom coloring
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    im = ax.imshow(mask, cmap=cmap, aspect='auto', interpolation='none')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation='vertical')
    ax.set_yticklabels(df.index, fontsize=FONT_SIZE_TICK, va='center', linespacing=1.5)


    # Annotate the chart with 'link_size' values, skipping 'NaN' values
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, (str(round(value))), ha='center', va='center', color='white', fontsize=FONT_SIZE_VALUE)

    # Set labels and title
    plt.xlabel('Timestamps', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Links', fontsize=FONT_SIZE_LABEL)
    # plt.title('Link Size Over Time', fontsize=FONT_SIZE_TITLE)

    # Adjust spacing for vertical axis labels
    plt.tight_layout()
    
    # Save the plot to the specified output path
    if save == True:
        plt.savefig(join(output_path, f'link_size_over_time_{group}.pdf'))
        
    if show != True:
        plt.close(fig)
    else: 
        plt.show() 

#####################################
######### Link Growth #############
#####################################

def plot_link_growth_over_time(metric_dataframes, group='sample', output_path="output/links/", save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (12, 8)

    # Extract data
    df = metric_dataframes[group]['size']

    # Sort links by their average size
    links_order = df.mean(axis=1).sort_values(ascending=False).index
    df = df.reindex(links_order)

    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Plot each link's size over time
    for link in links_order:
        ax.plot(df.columns, df.loc[link], marker='o', linestyle='-', label=f'Link {link}')

    # Labels and Title
    ax.set_xlabel("Time")
    ax.set_ylabel("Link Size")
    # ax.set_title("Link Size Over Time")


    # Set tick positions and labels
    ax.set_xticks(np.arange(len(df.columns)))  # Set tick positions
    ax.set_xticklabels(df.columns, rotation=90)  # Set tick labels and rotate for better readability

    # Add legend
    ax.legend(title="Links", loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}/link_growth_over_time_{group}.pdf", bbox_inches='tight')

    if show:
        plt.show()

#####################################
###### Key Scatter Chart ############
#####################################


def plot_scatter_metrics(metric_dataframes, x_metric, y_metric, group='sample', output_path="output/metrics/", save=True, show=True, dpi=300):
    # To-do: 
    ### Add in colouring by clique

    # Constants for aesthetics
    FIG_SIZE = (10, 6)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Extract data
    x_data = metric_dataframes[group][x_metric].dropna()
    y_data = metric_dataframes[group][y_metric].dropna()

    # Prepare figure and axis for plotting
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Create a scatter plot of the two metrics
    ax.scatter(x_data, y_data)

    # Labels and Title
    ax.set_xlabel(x_metric.replace('_', ' ').title())
    ax.set_ylabel(y_metric.replace('_', ' ').title())
    # ax.set_title(f"{x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()}")

    # # Calculate correlation and p-value
    # correlation, p_value = pearsonr(x_data.mean(), y_data.mean())

    # # Show the correlation value and p-value on the plot
    # ax.text(0.05, 0.95, f'Correlation of Means: {correlation:.2f}\nP-value: {p_value:.3f}', transform=ax.transAxes,
    #         fontsize=10, verticalalignment='top')

    plt.tight_layout()

    # Save or show the figure
    if save:
        plt.savefig(f"{output_path}/{x_metric}_vs_{y_metric}_{group}.pdf", bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()

        
#####################################
###### Key Heatmap Chart ############
#####################################

def pval_to_significance(pval):
    """
    Convert p-value to significance symbols.

    Convention:
    **  p < 0.01
    *   p < 0.05
    †   p < 0.1
    """
    if pval < 0.01:
        return '‡'
    elif pval < 0.05:
        return '*'
    elif pval < 0.1:
        return '+'
    else:
        return ''
    
def plot_heatmap_chart(metric_dataframes, metric_name, pct=True, log=False, output_path="../output/links/", save=False, show=True):

    FIG_SIZE = (14, 14)  # (width, height) in inches
    FONT_SIZE_LABEL = 18   # For axis labels
    FONT_SIZE_TEXT = 14    # For tick labels and other text
    FONT_SIZE_VALUE = 14  # For cell annotations
    
    if pct==True: 
        multiplier = 100 
        # unit='%'
        unit=''
    else: 
        multiplier = 1
        unit=''

    # Define df
    if log == True: 
        df = np.log10(metric_dataframes['sample'][metric_name]) * multiplier
    else: 
        df = metric_dataframes['sample'][metric_name] * multiplier


    df_pv = metric_dataframes['pvalues'][metric_name]
    
    # reindex
    df_index = metric_dataframes['sample'][metric_name] * multiplier

    # Find the index of the first occurrence (value > 1) in each column (snapshot)
    first_occurrence_indices = (df_index.T > 0).idxmax()
    
    # Determine the minimum index (earliest occurrence) for each clique across all snapshots
    min_indices = first_occurrence_indices.groupby(first_occurrence_indices.index).min()
    
    # Sort the cliques based on their minimum indices to get the desired order
    cliques_order = min_indices.sort_values().index.tolist()

    # Reindex clique size to df
    df= df.reindex(cliques_order)
    df_pv = df_pv.reindex(cliques_order)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Create colormap
    cmap = plt.get_cmap("magma", lut=128)
    norm = mcolors.Normalize(vmin=0, vmax=round(df.max().max()*1.3))

    # Plotting the values
    im = ax.imshow(df, cmap=cmap, norm=norm, aspect='auto', interpolation='none')


    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)  # '2%' determines the width of the colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
    tick_vals = np.array(cbar.get_ticks()) 
    cbar.set_ticklabels([f'{round(val)}%' for val in tick_vals])
    cbar.ax.tick_params(labelsize=FONT_SIZE_TEXT)  # Increased tick label size

    
     # Significance box
    ax.text(0.02, 0.02, 'Relative to Control:\n‡ = 0.01\n* = 0.05\n+ = 0.1',
            transform=ax.transAxes, fontsize=FONT_SIZE_TEXT,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='lightyellow', edgecolor='black'))
    # Labels and title
    ax.set_xlabel('Date', size=FONT_SIZE_LABEL)
    ax.set_ylabel('Links', size=FONT_SIZE_LABEL)
    # ax.set_title(f'{metric_name.replace("_", " ").title()}', size=FONT_SIZE_TITLE)
    ax.set_xticks(np.arange(len(df.columns)))
    # Ensure df.columns are datetime objects or strings; adjust formatting as needed
    if isinstance(df.columns, pd.DatetimeIndex):
        ax.set_xticklabels(df.columns.strftime('%Y-%m-%d'), rotation=90, ha='center', fontsize=FONT_SIZE_TEXT)
    else:
        ax.set_xticklabels(df.columns, rotation=90, ha='center', fontsize=FONT_SIZE_TEXT)



    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, size=FONT_SIZE_TEXT)
    plt.grid(False)
    plt.tight_layout()


    # Annotate the values on the plot
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.values[i, j]
            pval = df_pv.values[i, j]
            pval = pval_to_significance(pval) 
            if not np.isnan(value):
                ax.text(j, i, f'{value:.1f}{unit}{pval}', ha='center', va='center', color='white', fontsize=FONT_SIZE_VALUE)

    # Save and/or show the plot
    if save:
        plt.savefig(join(output_path, f'{metric_name}_links.pdf'), bbox_inches='tight')
    if show:
        plt.show()

#####################################
### Key Heatmap Directional Chart ###
#####################################

def plot_heatmap_chart_directional(metric_dataframes, metric_name, pct=True, log=False, output_path="../output/links/", save=False, show=True, min_occurance=9):
    # Figure and font sizes
    FIG_SIZE = (14, 10)   # (width, height) in inches
    FONT_SIZE_LABEL = 18   # For axis labels
    FONT_SIZE_TEXT = 14    # For tick labels and other text
    FONT_SIZE_VALUE = 14  # For cell annotations


    if pct==True: 
        multiplier = 100 
        unit=''
    else: 
        multiplier = 1
        unit=''

    # Define df
    if log == True: 
        df = np.log10(metric_dataframes['sample_directional'][metric_name]) * multiplier
    else: 
        df = metric_dataframes['sample_directional'][metric_name] * multiplier


    df_pv = metric_dataframes['pvalues_directional'][metric_name]
    
    # filter for relevant rows 
    pval_filter = (df_pv <= 0.1).any(axis=1)==True
    
    # updated frames 
    df = df[pval_filter]
    df_pv = df_pv[pval_filter]

    # Additional filtering criterion to exclude links appearing less than once
    # Calculate the number of non-zero occurrences for each row (clique)
    non_zero_counts = (df > 0).sum(axis=1)
    
    # Filter rows where the number of non-zero occurrences is greater than 5
    df = df[non_zero_counts >= min_occurance]
    # df_pv = df_pv.loc[df.index]

    # reindex
    df_index = df.copy()

    # Find the index of the first occurrence (value > 1) in each column (snapshot)
    first_occurrence_indices = (df_index.T > 0).idxmax()
    
    # Determine the minimum index (earliest occurrence) for each clique across all snapshots
    min_indices = first_occurrence_indices.groupby(first_occurrence_indices.index).min()
    
    # Sort the cliques based on their minimum indices to get the desired order
    cliques_order = min_indices.sort_values().index.tolist()

    # Reindex clique size to df
    df= df.reindex(cliques_order)
    df_pv = df_pv.reindex(cliques_order)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Create colormap
    cmap = plt.get_cmap("magma", lut=128)
    norm = mcolors.Normalize(vmin=0, vmax=round(df.max().max()*1.2))

    # Plotting the values
    im = ax.imshow(df, cmap=cmap, norm=norm, aspect='auto', interpolation='none')


    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)  # '2%' determines the width of the colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
    tick_vals = np.array(cbar.get_ticks()) 
    cbar.set_ticklabels([f'{round(val)}%' for val in tick_vals])
    cbar.ax.tick_params(labelsize=FONT_SIZE_TEXT)  # Increased tick label size

    
    # Significance box
    ax.text(0.02, 0.02, 'Relative to Control:\n‡ = 0.01\n* = 0.05\n+ = 0.1',
            transform=ax.transAxes, fontsize=FONT_SIZE_VALUE,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle="square,pad=0.3", facecolor='lightyellow', edgecolor='black'))
    # Labels and title
    ax.set_xlabel('Date', size=FONT_SIZE_LABEL)
    ax.set_ylabel('Links', size=FONT_SIZE_LABEL)
    # ax.set_title(f'Directional Analysis: {metric_name.replace("_", " ").title()}', size=FONT_SIZE_TITLE)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90, ha='center', size=FONT_SIZE_TEXT)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, size=FONT_SIZE_TEXT)
    plt.grid(False)
    plt.tight_layout()


    # Annotate the values on the plot
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.values[i, j]
            pval = df_pv.values[i, j]
            pval = pval_to_significance(pval) 
            if not np.isnan(value):
                ax.text(j, i, f'{value:.1f}{unit}{pval}', ha='center', va='center', color='white', fontsize=FONT_SIZE_VALUE)

    # Save and/or show the plot
    if save:
        plt.savefig(join(output_path, f'{metric_name}_links_directional.pdf'), bbox_inches='tight')
    if show:
        plt.show()
        



#####################################
###### Key Boxplot Chart ############
#####################################
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson

#####################################
###### Key Boxplot Chart ############
#####################################

def plot_boxplot(metric_dataframes, metric, unit, group='sample', pval_group='pvalues', output_path="../output/links/", custom_index='', min_occurrences=9, significance=False, save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (10, 10)  # or (8, 5), etc.
    COLOR_MAP = {'non-significant': 'lightgray', '0.05': 'yellow', '0.01': 'orange', '0.001': 'red'}

    # Extract data
    df = metric_dataframes[group][metric]
    
    # Load p_values to control if significance is True
    df_pvalues = metric_dataframes[pval_group][metric] if significance else None

    # Sort links by average influence or use custom index if provided
    if custom_index:
        links_order = custom_index
    else:
        links_order = df.mean(axis=1).sort_values(ascending=False).index

    df = df.reindex(links_order)
    if significance:
        df_pvalues = df_pvalues.reindex(links_order)

    # Filter based on minimum occurrences
    df = df[df.count(axis=1) >= min_occurrences]
    if significance:
        df_pvalues = df_pvalues.loc[df.index]

    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Create boxplot data
    boxplot_data = [df.loc[link].dropna() for link in df.index]
    boxplot = ax.boxplot(boxplot_data, vert=False, patch_artist=True)

    # Color and annotate based on significance if enabled
    if significance:
        for i, link in enumerate(df.index):
            p_values = df_pvalues.loc[link].dropna()
            # Determine the most common significance level based on mode
            try: 
                significance_level = p_values.apply(lambda x: '0.001' if x < 0.001 else ('0.01' if x < 0.01 else ('0.05' if x < 0.05 else 'non-significant'))).mode()[0]
            except: 
                significance_level = 'non-significant'
            boxplot['boxes'][i].set_facecolor(COLOR_MAP[significance_level])

        # Adding legend for significance
        legend_patches = [patches.Patch(color=color, label=significance) for significance, color in COLOR_MAP.items()]
        ax.legend(handles=legend_patches, title="Significance Levels", loc='lower left', fontsize=12, title_fontsize=14)
    else:
        for box in boxplot['boxes']:
            box.set_facecolor('lightgray')

    # Labels and Title
    metric_name_formatted = ' '.join(metric.split('_')).title()
    ax.set_yticks(np.arange(1, len(df.index) + 1))
    ax.set_yticklabels(df.index, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)  # X-axis tick labels

    ax.set_ylabel('Links', fontsize=18)
    ax.set_xlabel(f"{metric_name_formatted} {unit}", fontsize=18)
    # ax.set_title(f"{metric_name_formatted} of Links", fontsize=20)

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}/{metric}_boxplot_{group}.pdf", bbox_inches='tight', format='pdf', dpi=300)

    if show:
        plt.show()




def plot_boxplot_with_significance(metric_dataframes, metric, unit, group='sample', output_path="output/links/", save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (12, 8)
    COLOR_MAP = {'non-significant': 'lightgray', '0.05': 'yellow', '0.01': 'orange', '0.001': 'red'}

    # Extract data
    df = metric_dataframes[group][metric]
    
    # load p_values to control
    df_pvalues = metric_dataframes['pvalues'][metric]      

    # Sort links by average influence
    links_order = df.mean(axis=1).sort_values(ascending=False).index
    df = df.reindex(links_order)
    df_pvalues = df_pvalues.reindex(links_order)

    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Create boxplot data
    boxplot_data = [df.loc[link].dropna() for link in links_order]
    boxplot = ax.boxplot(boxplot_data, vert=False, patch_artist=True)

    # Color and annotate based on significance
    for i, link in enumerate(links_order):
        p_values = df_pvalues.loc[link].dropna()
        # Determine the most common significance level based on mode
        try: 
            significance = p_values.apply(lambda x: '0.001' if x < 0.001 else ('0.01' if x < 0.01 else ('0.05' if x < 0.05 else 'non-significant'))).mode()[0]
        except: 
            significance = 'non-significant'
        boxplot['boxes'][i].set_facecolor(COLOR_MAP[significance])

    # Adding legend for significance
        if metric == 'size':

            pass

        else:
            legend_patches = [patches.Patch(color=color, label=significance) for significance, color in COLOR_MAP.items()]
            ax.legend(handles=legend_patches, title="Significance Levels", loc='upper left', bbox_to_anchor=(1, 1))


    # Labels and Title
    metric_name_formatted = ' '.join(metric.split('_')).title()
    ax.set_yticks(np.arange(1, len(links_order) + 1))
    ax.set_yticklabels(links_order)
    ax.set_ylabel('Links')
    ax.set_xlabel(f"{metric_name_formatted} {unit}")
    # ax.set_title(f"{metric_name_formatted} of Links")

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}/{metric}_significance_boxplot_{group}.pdf", bbox_inches='tight')

    if show:
        plt.show()
        
        
#####################################
###### Influence Labels #############
#####################################
def create_and_normalize_matrix(dataframe, label_column='Link Name', short_labels=None):
    """
    Create a normalized matrix from a dataframe with string-encoded dictionaries of influence labels.

    Args:
        dataframe (pd.DataFrame): The original dataframe.
        label_column (str): The name of the column to use as labels (default: 'Link Name').
        short_labels (dict): A dictionary for mapping original labels to human-readable labels.

    Returns:
        pd.DataFrame: The normalized and renamed dataframe.
    """
    def convert_str_to_dict(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, dict) else {})
        except:
            return {}

    dataframe = dataframe.copy()
    for column in dataframe.columns.difference([label_column]):
        dataframe[column] = dataframe[column].apply(convert_str_to_dict)

    def extract_and_average(row):
        contract_sums = {}
        contract_counts = {}
        for cell in row:
            for contract, value in cell.items():
                if contract in contract_sums:
                    contract_sums[contract] += value
                    contract_counts[contract] += 1
                else:
                    contract_sums[contract] = value
                    contract_counts[contract] = 1
        return {contract: contract_sums[contract] / contract_counts[contract] for contract in contract_sums}

    dataframe['averages'] = dataframe[dataframe.columns.difference([label_column])].apply(extract_and_average, axis=1)

    matrix_df = pd.DataFrame(index=dataframe[label_column])
    for index, row in dataframe.iterrows():
        for contract, avg_value in row['averages'].items():
            matrix_df.at[row[label_column], contract] = avg_value
    matrix_df.fillna(0, inplace=True)

    row_sums = matrix_df.sum(axis=1).replace(0, np.nan)
    result_df = matrix_df.div(row_sums, axis=0).fillna(0)

    if short_labels:
        result_df.rename(columns=short_labels, inplace=True)
        result_df = result_df[[short_labels[key] for key in short_labels if short_labels[key] in result_df.columns]]


    return result_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
import os

def plot_heatmap_labels(metric_dataframes, metric, group='sample', colormap='magma',
                        output_path='output/links', min_occurrences=9):
    """
    Plot a heatmap from a dataframe, filtering links with a minimum number of occurrences.
    """
    # Example short labels
    # short_readable_labels = {
    #     'EMOA': 'EOA Addresses',
    #     'IEMOA': 'Institutional Addresses',
    #     'PCV': 'Protocol Addresses',
    #     'vesting_contract': 'Vesting Contracts',
    #     'external_staking_contracts': 'Staking Contracts',
    #     'lp_amm': 'Liquidity Pools',
    #     'lending_borrowing_contract': 'Lending/Borrowing Contracts',
    #     'bridge_contract': 'Bridge Contracts',
    #     'other_contracts': 'Other Contracts',
    # }

    short_readable_labels = {
    'EMOA': 'EOAs',
    'IEMOA': 'Institutions',
    'PCV': 'Protocols',
    'vesting_contract': 'Vesting',
    'external_staking_contracts': 'Staking',
    'lp_amm': 'Liquidity',
    'lending_borrowing_contract': 'Lending',
    'bridge_contract': 'Bridges',
    'other_contracts': 'Other',
    }

    # Figure and font sizes
    FIG_SIZE = (12, 10)   # (width, height) in inches
    FONT_SIZE_LABEL = 18
    FONT_SIZE_TEXT  = 14

    # 1. Get the raw data for the specified group
    df_raw = metric_dataframes[group][metric]

    # 2. Filter for links with at least `min_occurrences` non-NaN values
    link_occurrences = df_raw.notna().sum(axis=1)
    df_filtered = df_raw[link_occurrences >= min_occurrences]

    # 3. Create & normalize the matrix (adapt to your own function/logic)
    df = create_and_normalize_matrix(
        df_filtered.reset_index(),
        label_column='Link Name',
        short_labels=short_readable_labels
    )

    if df.empty:
        print(f"No data to plot after filtering for {group} group.")
        return

    # 4. Prepare figure/axes
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 5. Convert your normalized values into percentages
    data = df.to_numpy() * 100

    # 6. We’ll manually control the normalization so the colorbar matches
    vmin, vmax = 0, np.max(data)  # Adjust as needed
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 7. Plot the heatmap with `imshow` and the chosen norm
    #    (Using `imshow` rather than `matshow` for more flexible arguments)
    heatmap = ax.imshow(data, cmap=colormap, norm=norm, aspect='auto')

    # 8. Create a divider to append colorbar on the right
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=0.2)  
    # size="1.5%" controls colorbar thickness; pad=0.2 is the gap

    cbar = plt.colorbar(
        ScalarMappable(norm=norm, cmap=colormap),
        cax=cax,
        orientation='vertical'
    )

    # 9. Ticks & labels on the colorbar
    tick_vals = cbar.get_ticks()
    # Convert them to integers or round as appropriate
    cbar.set_ticklabels([f'{int(val)}%' for val in tick_vals])
    cbar.ax.tick_params(labelsize=FONT_SIZE_TEXT)

    # 10. Axis labels
    ax.set_xlabel('Labels', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Links',  fontsize=FONT_SIZE_LABEL)

    # Move x-axis tick labels to the bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)

    # 11. Tick labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90, ha='right', fontsize=FONT_SIZE_TEXT)

    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=FONT_SIZE_TEXT)

    # 12. Annotate cells with values
    text_color_threshold = (vmax) / 2  # Halfway point to determine text color
    for (i, j), val in np.ndenumerate(data):
        text_color = 'white' if val < text_color_threshold else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                color=text_color, fontsize=FONT_SIZE_TEXT)

    # 13. Layout and save
    plt.tight_layout()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(
        os.path.join(output_path, "label_plot_links.pdf"),
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


#####################################
###### Key Lollipop chart Labels ####
#####################################

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr

def pval_to_marker(pval):
    """ Returns marker style, face color, edge color, edge width, and size based on p-value significance level. """
    if pval < 0.01:
        return 'D', 'black', 'black', 3, 8  # Filled circle
    elif pval < 0.05:
        return 'D', 'none', 'black', 3, 8  # Bold circle (outlined thicker)
    elif pval < 0.1:
        return 'D', 'none', 'black', 1, 8  # Standard circle (outlined)
    else:
        return 'D', 'none', 'none', 0, 0  # No marker

def plot_lollipop_correlation_vs_tvl(metric_dataframes, tvl_data_path, metric='internal_influence', output_path="output/", min_occurrences=1, save=True, show=True):
    # Extract internal influence data
    influence_df = metric_dataframes['sample'][metric]

    # Normalize the datetime format for influence DataFrame
    influence_df.columns = pd.to_datetime(influence_df.columns).normalize()

    # Load TVL data
    with open(tvl_data_path, 'r') as file:
        tvl_data = json.load(file)
    tvl_df = pd.DataFrame(tvl_data)
    tvl_df['date'] = pd.to_datetime(tvl_df['date'], unit='s')
    tvl_df.set_index('date', inplace=True)

    # Initialize a list to store correlations and p-values
    correlations = []
    pvals = []

    # Iterate over each link (row)
    for link in influence_df.index:
        influence = influence_df.loc[link]
        aligned_data = pd.concat([influence, tvl_df['tvl'].pct_change(1)], axis=1, join='inner').dropna()

        if not aligned_data.empty and aligned_data.shape[0] > min_occurrences:
            influence_aligned = aligned_data.iloc[:, 0]
            tvl_aligned = aligned_data.iloc[:, 1]
            correlation, pval = pearsonr(influence_aligned, tvl_aligned)
            correlations.append(correlation)
            pvals.append(pval)
        else:
            correlations.append(None)
            pvals.append(None)

    result_df = pd.DataFrame({
        'Correlation': correlations,
        'P-value': pvals
    }, index=influence_df.index).dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hlines(y=result_df.index, xmin=0, xmax=result_df['Correlation'], color='gray', alpha=0.5)
    
    for i, (corr, pval) in enumerate(zip(result_df['Correlation'], result_df['P-value'])):
        marker, facecolor, edgecolor, edgewidth, size = pval_to_marker(pval)
        if marker != 'None':
            ax.scatter(corr, result_df.index[i], marker=marker, facecolor=facecolor, edgecolor=edgecolor, s=size**2, linewidths=edgewidth)

    ax.set_xlabel("Correlation with TVL % Change")
    ax.set_ylabel("Link")
    # ax.set_title(f"Lollipop Plot of {metric.replace('_', ' ').title()} Correlations vs. TVL % Change per Link")

    # Create custom legend for significance levels
    circle_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.01', markerfacecolor='black')
    square_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.05', markerfacecolor='none', markeredgewidth=3)
    diamond_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.1', markerfacecolor='none', markeredgewidth=2)

    ax.legend(handles=[circle_patch, square_patch, diamond_patch], loc='lower right', title='Significance Levels')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(output_path, "lollipop_correlation_vs_tvl.pdf"), format='pdf', dpi=300)
    if show:
        plt.show()

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson

def plot_lollipop_correlation_vs_tvl_r(metric_dataframes, tvl_data_path, metric='internal_influence', output_path="output/", min_occurrences=1, save=True, show=True):
    # Extract internal influence data
    influence_df = metric_dataframes['sample'][metric]

    # Normalize the datetime format for influence DataFrame
    influence_df.columns = pd.to_datetime(influence_df.columns).normalize()

    # Load TVL data
    with open(tvl_data_path, 'r') as file:
        tvl_data = json.load(file)
    tvl_df = pd.DataFrame(tvl_data)
    tvl_df['date'] = pd.to_datetime(tvl_df['date'], unit='s')
    tvl_df.set_index('date', inplace=True)

    # Initialize a list to store correlations, p-values, Durbin-Watson values, and Ljung-Box results
    correlations = []
    pvals = []
    durbin_watson_stats = []
    ljung_box_results = []

    # Iterate over each link (row)
    for link in influence_df.index:
        influence = influence_df.loc[link]
        aligned_data = pd.concat([influence, tvl_df['tvl'].pct_change(1)], axis=1, join='inner').dropna()

        if not aligned_data.empty and aligned_data.shape[0] > min_occurrences:
            influence_aligned = aligned_data.iloc[:, 0]
            tvl_aligned = aligned_data.iloc[:, 1]
            correlation, pval = pearsonr(influence_aligned, tvl_aligned)
            correlations.append(correlation)
            pvals.append(pval)
            
            # Perform Durbin-Watson Test
            dw_stat = durbin_watson(influence_aligned)
            durbin_watson_stats.append(dw_stat)
            
            # Perform Ljung-Box Q Test and handle potential errors
            ljung_box_result = acorr_ljungbox(influence_aligned, lags=[min_occurrences], return_df=True)
            ljung_box_pvalue = ljung_box_result.iloc[0]['lb_pvalue'] if not ljung_box_result.empty else None
            ljung_box_results.append(ljung_box_pvalue)
            
        else:
            correlations.append(None)
            pvals.append(None)
            durbin_watson_stats.append(None)
            ljung_box_results.append(None)

    result_df = pd.DataFrame({
        'Correlation': correlations,
        'P-value': pvals,
        'Durbin-Watson': durbin_watson_stats,
        'Ljung-Box P-value': ljung_box_results
    }, index=influence_df.index).dropna()

    # Plotting Lollipop Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hlines(y=result_df.index, xmin=0, xmax=result_df['Correlation'], color='gray', alpha=0.5)
    
    for i, (corr, pval) in enumerate(zip(result_df['Correlation'], result_df['P-value'])):
        marker, facecolor, edgecolor, edgewidth, size = pval_to_marker(pval)
        if marker != 'None':
            ax.scatter(corr, result_df.index[i], marker=marker, facecolor=facecolor, edgecolor=edgecolor, s=size**2, linewidths=edgewidth)

    ax.set_xlabel("Correlation with TVL % Change")
    ax.set_ylabel("Link")
    # ax.set_title(f"Lollipop Plot of {metric.replace('_', ' ').title()} Correlations vs. TVL % Change per Link")

    # Create custom legend for significance levels
    circle_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.01', markerfacecolor='black')
    square_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.05', markerfacecolor='none', markeredgewidth=3)
    diamond_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.1', markerfacecolor='none', markeredgewidth=2)

    ax.legend(handles=[circle_patch, square_patch, diamond_patch], loc='lower right', title='Significance Levels')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(output_path, "lollipop_correlation_vs_tvl.pdf"), format='pdf', dpi=300)
    if show:
        plt.show()
    
    # Display ACF Plot for all links combined
    fig, ax = plt.subplots(figsize=(10, 6))
    combined_influence = influence_df.mean()
    plot_acf(combined_influence.dropna(), ax=ax, lags=min_occurrences)
    # plt.title(f"Autocorrelation Plot for Average {metric.replace('_', ' ').title()}")
    plt.show()

    # Display results
    print("\n===== Summary Statistics =====")
    print(result_df)

    # Generate and print the LaTeX table
    latex_table = result_df.to_latex(float_format="%.4f", index=True, header=True, column_format="lrrrr", caption="Summary Statistics for Correlation and Autocorrelation Analysis", label="tab:summary_stats")
    print("\n===== LaTeX Table =====")
    print(latex_table)

    return result_df


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson

def plot_lollipop_correlation_vs_tvl_adjusted_auto_corr(metric_dataframes, tvl_data_path, metric='internal_influence', output_path="output/", min_occurrences=1, save=True, show=True):
    # Extract internal influence data
    influence_df = metric_dataframes['sample'][metric]

    # Normalize the datetime format for influence DataFrame
    influence_df.columns = pd.to_datetime(influence_df.columns).normalize()

    # Load TVL data
    with open(tvl_data_path, 'r') as file:
        tvl_data = json.load(file)
    tvl_df = pd.DataFrame(tvl_data)
    tvl_df['date'] = pd.to_datetime(tvl_df['date'], unit='s')
    tvl_df.set_index('date', inplace=True)

    # Initialize a list to store correlations, p-values, Durbin-Watson values, and Ljung-Box results
    correlations = []
    pvals = []
    durbin_watson_stats = []
    ljung_box_results = []

    # Iterate over each link (row)
    for link in influence_df.index:
        influence = influence_df.loc[link]
        # Apply first differencing to remove autocorrelation
        influence_diff = influence.diff().dropna()
        
        aligned_data = pd.concat([influence_diff, tvl_df['tvl'].pct_change(1)], axis=1, join='inner').dropna()

        if not aligned_data.empty and aligned_data.shape[0] > min_occurrences:
            influence_aligned = aligned_data.iloc[:, 0]
            tvl_aligned = aligned_data.iloc[:, 1]
            correlation, pval = pearsonr(influence_aligned, tvl_aligned)
            correlations.append(correlation)
            pvals.append(pval)
            
            # Perform Durbin-Watson Test
            dw_stat = durbin_watson(influence_aligned)
            durbin_watson_stats.append(dw_stat)
            
            # Perform Ljung-Box Q Test and handle potential errors
            ljung_box_result = acorr_ljungbox(influence_aligned, lags=[min_occurrences], return_df=True)
            ljung_box_pvalue = ljung_box_result.iloc[0]['lb_pvalue'] if not ljung_box_result.empty else None
            ljung_box_results.append(ljung_box_pvalue)
            
        else:
            correlations.append(None)
            pvals.append(None)
            durbin_watson_stats.append(None)
            ljung_box_results.append(None)

    result_df = pd.DataFrame({
        'Correlation': correlations,
        'P-value': pvals,
        'Durbin-Watson': durbin_watson_stats,
        'Ljung-Box P-value': ljung_box_results
    }, index=influence_df.index).dropna()

    # Plotting Lollipop Chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hlines(y=result_df.index, xmin=0, xmax=result_df['Correlation'], color='gray', alpha=0.5)
    
    for i, (corr, pval) in enumerate(zip(result_df['Correlation'], result_df['P-value'])):
        marker, facecolor, edgecolor, edgewidth, size = pval_to_marker(pval)
        if marker != 'None':
            ax.scatter(corr, result_df.index[i], marker=marker, facecolor=facecolor, edgecolor=edgecolor, s=size**2, linewidths=edgewidth)

    ax.set_xlabel("Correlation with TVL % Change")
    ax.set_ylabel("Link")
    # ax.set_title(f"Lollipop Plot of {metric.replace('_', ' ').title()} Correlations vs. TVL % Change per Link")

    # Create custom legend for significance levels
    circle_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.01', markerfacecolor='black')
    square_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.05', markerfacecolor='none', markeredgewidth=3)
    diamond_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.1', markerfacecolor='none', markeredgewidth=2)

    ax.legend(handles=[circle_patch, square_patch, diamond_patch], loc='lower right', title='Significance Levels')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(output_path, "lollipop_correlation_vs_tvl_adjusted.pdf"), format='pdf', dpi=300)
    if show:
        plt.show()
    
    # Display ACF Plot for all links combined
    fig, ax = plt.subplots(figsize=(10, 6))
    combined_influence = influence_df.mean().diff().dropna()
    plot_acf(combined_influence, ax=ax, lags=min_occurrences)
    # plt.title(f"Autocorrelation Plot for Average {metric.replace('_', ' ').title()} (Differenced)")
    plt.show()

    # Display results
    print("\n===== Summary Statistics =====")
    print(result_df)

    # Generate and print the LaTeX table
    latex_table = result_df.to_latex(float_format="%.4f", index=True, header=True, column_format="lrrrr", caption="Summary Statistics for Correlation and Autocorrelation Analysis (Adjusted)", label="tab:summary_stats_adjusted")
    print("\n===== LaTeX Table =====")
    print(latex_table)

    return result_df




#####################################
###### TVL chart                 ####
#####################################

def plot_monthly_tvl(metric_dataframes, tvl_data_path, output_path="../output/tvl_monthly_chart.pdf", save=False, show=True):

    # date_range = metric_dataframes['sample']['internal_influence'].columns

    # Load TVL data
    with open(tvl_data_path, 'r') as file:
        tvl_data = json.load(file)
    tvl_df = pd.DataFrame(tvl_data)
    tvl_df['date'] = pd.to_datetime(tvl_df['date'], unit='s')
    tvl_df.set_index('date', inplace=True)
    

    df = metric_dataframes['sample']['internal_influence']
    # Convert index to datetime if not already
    if not isinstance(df.columns, pd.DatetimeIndex):
        df.columns = pd.to_datetime(df.columns)
    # Resample to monthly values
    monthly_tvl = tvl_df[tvl_df.index.isin(pd.to_datetime(df.columns))]

    
    # Plot
    plt.figure(figsize=(8, 6))
    
    plt.plot(monthly_tvl.index, monthly_tvl['tvl'], marker='x', linestyle='-', color='black')
    plt.fill_between(monthly_tvl.index, monthly_tvl['tvl'], color='black', alpha=0.1)
    # plt.title('Monthly TVL (Total Value Locked)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('TVL (USD)', fontsize=12)
     # Ensure all x-axis dates are shown
    plt.xticks(monthly_tvl.index, [date.strftime('%Y-%m') for date in monthly_tvl.index], rotation=90)
   
    plt.grid(True)
    plt.tight_layout()
    
    # Save and/or show the plot
    if save:
        plt.savefig(output_path, bbox_inches='tight')
    if show:
        plt.show()



#####################################
###### Sensitivity Analysis      ####
#####################################

def sensitivity_analysis(file_paths, metrics, output_path="../output/links/", save=True, show=True):
    def load_pickle_dynamic(file_path):
        sys.modules['numpy._core'] = np.core
        with open(file_path, 'rb') as file:
            try:
                data = pickle.load(file)
            except ModuleNotFoundError as e:
                print(f"ModuleNotFoundError: {e}")
                data = pickle.load(file, encoding='latin1')
        return data

    data_frames = {path: load_pickle_dynamic(path) for path in file_paths}
    combined_data = []

    for path, data in data_frames.items():
        threshold = path.split('_')[-1].split('.pkl')[0]  # Extract threshold level from the file path
        threshold = threshold.replace('e-', 'e-')  # Ensure proper formatting
        for date, pairs in data['sample'].items():
            for pair, metrics_dict in pairs.items():
                metrics_dict['threshold'] = threshold
                metrics_dict['date'] = date
                metrics_dict['pair'] = pair
                combined_data.append(metrics_dict)

    combined_df = pd.DataFrame(combined_data)
    combined_df['threshold'] = pd.to_numeric(combined_df['threshold'], errors='coerce')
    combined_df = combined_df.sort_values(by='threshold')

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, n_rows*3))  # Increased plot height
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        boxplot = combined_df.boxplot(column=metric, by='threshold', grid=False, patch_artist=True,
                                      boxprops=dict(facecolor='white', color='black'),
                                      medianprops=dict(color='red'),
                                      whiskerprops=dict(color='black'),
                                      capprops=dict(color='black'),
                                      
                                      flierprops=dict(markerfacecolor='black', marker='o', markersize=1),
                                      ax=ax, 
                                      whis=3.0)  # Adjust whiskers to 3.0 times the IQR to reduce outlier detection as data is super skewed

        ax.set_title('')  # Remove default title set by pandas' boxplot `by` parameter.


        ax.set_xlabel('Threshold Level')
        ax.set_ylabel(metric.replace('_', ' ').title())
        # ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_title('')
        


        if metric in ['size', 'total_influence']:  # Assuming these metrics need log scale due to their nature
            ax.set_yscale('log')
        else:
            ax.autoscale_view()  # Autoscale view to adjust for non-logarithmic data

        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # plt.suptitle('Box Plot of Metrics by Supply Threshold Level', fontsize=16)
    plt.suptitle('', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(join(output_path, "sensitivity_analysis.pdf"), format='pdf', bbox_inches='tight', dpi=300)

    if show:
        plt.show()
