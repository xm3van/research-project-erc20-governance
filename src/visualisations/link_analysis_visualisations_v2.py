import pandas as pd
import numpy as np
import ast
import json
import os
from os.path import join 
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
import sys

from src.utilities.metrics_and_tests import pval_to_significance

# Global style settings
FIG_SIZE = (16, 10)
FONT_SIZE_TEXT = 16
FONT_SIZE_LABEL = 18
FONT_SIZE_TITLE = 20
FONT_SIZE_VALUE = 14
FONT_SIZE_TICK = 14
LINE_STYLES = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]
MARKERS = ['o', '^', 's', 'p', '*', 'x', 'D', 'v', '<', '>', 'h']

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.prop_cycle': plt.cycler(color=plt.cm.magma(np.linspace(0, 1, 10)))
})

#####################################
########### Link Size #############
#####################################

def plot_link_size_over_time(metric_dataframes, group='sample', output_path="output/links/", save=True, show=True):
    df = metric_dataframes[group]['size']
    df_index = metric_dataframes['sample']['size']

    first_occurrence_indices = (df_index.T > 1).idxmax()
    min_indices = first_occurrence_indices.groupby(first_occurrence_indices.index).min()
    links_order = min_indices.sort_values().index.tolist()
    df = df.reindex(links_order)

    mask = df.values >= 1
    cmap = mcolors.ListedColormap(['white', 'black'])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    im = ax.imshow(mask, cmap=cmap, aspect='auto', interpolation='none')

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation='vertical', fontsize=FONT_SIZE_TICK)
    ax.set_yticklabels(df.index, fontsize=FONT_SIZE_TICK, va='center')

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, str(round(value)), ha='center', va='center', color='white', fontsize=FONT_SIZE_VALUE)

    plt.xlabel('Timestamps', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Links', fontsize=FONT_SIZE_LABEL)
    plt.tight_layout()
    
    if save:
        plt.savefig(join(output_path, f'link_size_over_time_{group}.pdf'), bbox_inches='tight')
    if show:
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
    ax.set_title("Link Size Over Time")

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
    os.makedirs(output_path, exist_ok=True)
    x_data = metric_dataframes[group][x_metric].dropna()
    y_data = metric_dataframes[group][y_metric].dropna()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.scatter(x_data, y_data)

    ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"{x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()}", fontsize=FONT_SIZE_TITLE)

    plt.tight_layout()
    if save:
        plt.savefig(f"{output_path}/{x_metric}_vs_{y_metric}_{group}.pdf", bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()

#####################################
###### Key Heatmap Chart ############
#####################################

def plot_heatmap_chart(metric_dataframes, metric_name, pct=True, log=False, output_path="../output/links/", save=False, show=True):
    if pct:
        multiplier = 100
        unit = '%'
    else:
        multiplier = 1
        unit = ''

    df = metric_dataframes['sample'][metric_name] * multiplier
    df_pv = metric_dataframes['pvalues'][metric_name]

    first_occurrence_indices = (df.T > 0).idxmax()
    min_indices = first_occurrence_indices.groupby(first_occurrence_indices.index).min()
    cliques_order = min_indices.sort_values().index.tolist()

    df = df.reindex(cliques_order)
    df_pv = df_pv.reindex(cliques_order)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    cmap = plt.get_cmap("magma", lut=128)
    norm = mcolors.Normalize(vmin=0, vmax=round(df.max().max() * 1.3))
    im = ax.imshow(df, cmap=cmap, norm=norm, aspect='auto', interpolation='none')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')

    ax.set_xlabel('Date', size=FONT_SIZE_LABEL)
    ax.set_ylabel('Links', size=FONT_SIZE_LABEL)
    ax.set_title(f'{metric_name.replace("_", " ").title()}', size=FONT_SIZE_TITLE)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90, ha='center', size=FONT_SIZE_TICK)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, size=FONT_SIZE_TICK)
    plt.grid(False)
    plt.tight_layout()

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.values[i, j]
            pval = df_pv.values[i, j]
            pval = pval_to_significance(pval)
            if not np.isnan(value):
                ax.text(j, i, f'{value:.1f}{unit}{pval}', ha='center', va='center', color='white', fontsize=FONT_SIZE_VALUE)

    if save:
        plt.savefig(join(output_path, f'{metric_name}_links.pdf'), bbox_inches='tight')
    if show:
        plt.show()


#####################################
###### Key Boxplot Chart ############
#####################################

def plot_boxplot_with_significance(metric_dataframes, metric, unit, group='sample', output_path="output/links/", save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (16, 10)
    COLOR_MAP = {'non-significant': 'lightgray', '0.05': 'yellow', '0.01': 'orange', '0.001': 'red'}

    # Extract data
    df = metric_dataframes[group][metric]
    
    # Load p_values to control
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
    legend_patches = [patches.Patch(color=color, label=significance) for significance, color in COLOR_MAP.items()]
    ax.legend(handles=legend_patches, title="Significance Levels", loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=14)

    # Labels and Title
    metric_name_formatted = ' '.join(metric.split('_')).title()
    ax.set_yticks(np.arange(1, len(links_order) + 1))
    ax.set_yticklabels(links_order, fontsize=14)
    ax.set_ylabel('Links', fontsize=18)
    ax.set_xlabel(f"{metric_name_formatted} {unit}", fontsize=18)
    ax.set_title(f"{metric_name_formatted} of Links", fontsize=20)

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}/{metric}_significance_boxplot_{group}.pdf", bbox_inches='tight', format='pdf', dpi=300)

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


def plot_heatmap_labels(metric_dataframes, metric, group='sample', colormap='magma', output_path='output/links', min_occurrences=9):
    """
    Plot a heatmap from a dataframe, filtering links with a minimum number of occurrences.

    Args:
        metric_dataframes (dict): A dictionary containing dataframes to plot.
        group (str): The group to select from the metric_dataframes.
        colormap (str): The colormap to use for the heatmap.
        output_path (str): The directory to save the heatmap.
        min_occurrences (int): The minimum number of occurrences required to keep a link (default: 9).

    Returns:
        None
    """
    short_readable_labels = {
        'EMOA': 'EOA Addresses',
        'IEMOA': 'Institutional Addresses',
        'PCV': 'Protocol Addresses',
        'vesting_contract': 'Vesting Contracts',
        'external_staking_contracts': 'Staking Contracts',
        'lp_amm': 'Liquidity Pools',
        'lending_borrowing_contract': 'Lending/Borrowing Contracts',
        'bridge_contract': 'Bridge Contracts',
        'other_contracts': 'Other Contracts',
    }
    
    # Get the raw data for the specified group
    df_raw = metric_dataframes[group][metric]

    # Filter for links with at least `min_occurrences` non-NaN values
    link_occurrences = df_raw.notna().sum(axis=1)
    df_filtered = df_raw[link_occurrences >= min_occurrences]

    # Create and normalize the matrix using the filtered DataFrame
    df = create_and_normalize_matrix(df_filtered.reset_index(), label_column='Link Name', short_labels=short_readable_labels)
    
    # Check if the dataframe is empty after filtering
    if df.empty:
        print(f"No data to plot after filtering for {group} group.")
        return  # Exit the function if there's no data to plot

    # Proceed with plotting if the DataFrame is not empty
    fig, ax = plt.subplots(figsize=(20, 16))
    data = df.to_numpy() * 100

    cax = ax.matshow(data, cmap=colormap, aspect='auto')
    fig.colorbar(cax, ax=ax).ax.tick_params(labelsize=14)

    ax.set_xlabel('Labels', fontsize=18)
    ax.set_ylabel('Links', fontsize=18)
    ax.set_title('Relative Control for Links of Total Influence by Label per Link', fontsize=22)

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90, ha='right', size=16)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, size=16)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)

    # Set a threshold for text color based on the maximum value in the data
    text_color_threshold = np.max(data) / 2
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                color='white' if val < text_color_threshold else 'black', fontsize=14)

    # Save plot
    plt.savefig(f"{output_path}/label_plot_links.pdf", bbox_inches='tight', format='pdf', dpi=300)
    plt.show()



#####################################
###### Key Lollipop chart Labels ####
#####################################

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
    """
    Plot a lollipop chart showing correlations between internal influence and TVL percentage change.

    Args:
        metric_dataframes (dict): A dictionary containing dataframes to plot.
        tvl_data_path (str): Path to the JSON file containing TVL data.
        metric (str): The metric to correlate with TVL (default: 'internal_influence').
        output_path (str): The directory to save the lollipop chart.
        min_occurrences (int): The minimum number of occurrences required to keep a link (default: 1).
        save (bool): Whether to save the plot as a PDF file (default: True).
        show (bool): Whether to display the plot (default: True).

    Returns:
        None
    """

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
    ax.set_title(f"Lollipop Plot of {metric.replace('_', ' ').title()} Correlations vs. TVL % Change per Link")

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
    ax.set_title(f"Lollipop Plot of {metric.replace('_', ' ').title()} Correlations vs. TVL % Change per Link")

    # Create custom legend for significance levels
    circle_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.01', markerfacecolor='black')
    square_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.05', markerfacecolor='none', markeredgewidth=3)
    diamond_patch = mlines.Line2D([], [], color='black', marker='D', markersize=10, label='p < 0.1', markerfacecolor='none', markeredgewidth=2)

    ax.legend(handles=[circle_patch, square_patch, diamond_patch], loc='lower right', title='Significance Levels')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(output_path, "lollipop_correlation_vs_tvl_r.pdf"), format='pdf', dpi=300)
    if show:
        plt.show()
    
    # Display ACF Plot for all links combined
    fig, ax = plt.subplots(figsize=(10, 6))
    combined_influence = influence_df.mean()
    plot_acf(combined_influence.dropna(), ax=ax, lags=min_occurrences)
    plt.title(f"Autocorrelation Plot for Average {metric.replace('_', ' ').title()}")
    plt.show()

    # Display results
    print("\n===== Summary Statistics =====")
    print(result_df)

    # Generate and print the LaTeX table
    latex_table = result_df.to_latex(float_format="%.4f", index=True, header=True, column_format="lrrrr", caption="Summary Statistics for Correlation and Autocorrelation Analysis", label="tab:summary_stats")
    print("\n===== LaTeX Table =====")
    print(latex_table)

    return result_df


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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hlines(y=result_df.index, xmin=0, xmax=result_df['Correlation'], color='gray', alpha=0.5)
    
    for i, (corr, pval) in enumerate(zip(result_df['Correlation'], result_df['P-value'])):
        marker, facecolor, edgecolor, edgewidth, size = pval_to_marker(pval)
        if marker != 'None':
            ax.scatter(corr, result_df.index[i], marker=marker, facecolor=facecolor, edgecolor=edgecolor, s=size**2, linewidths=edgewidth)

    ax.set_xlabel("Correlation with TVL % Change")
    ax.set_ylabel("Link")
    ax.set_title(f"Lollipop Plot of {metric.replace('_', ' ').title()} Correlations vs. TVL % Change per Link")

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
    plt.title(f"Autocorrelation Plot for Average {metric.replace('_', ' ').title()} (Differenced)")
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
    plt.figure(figsize=(12, 8))
    
    plt.plot(monthly_tvl.index, monthly_tvl['tvl'], marker='x', linestyle='-', color='black')
    plt.fill_between(monthly_tvl.index, monthly_tvl['tvl'], color='black', alpha=0.1)
    plt.title('Monthly TVL (Total Value Locked)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('TVL (USD)', fontsize=12)
    # Ensure all x-axis dates are shown
    plt.xticks(monthly_tvl.index, [date.strftime('%Y-%m') for date in monthly_tvl.index], rotation=90)
   
    plt.grid(True)
    plt.tight_layout()
    
    # Save and/or show the plot
    if save:
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
    if show:
        plt.show()

#####################################
###### Sensitivity Analysis      ####
#####################################

def sensitivity_analysis(file_paths, metrics, highlight_threshold='5e-06', output_path="output/sensitivity_analysis.pdf", save=True, show=True):
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
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 6))  # Increased plot height
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

        ax.set_xlabel('Threshold Level')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}')

        if metric in ['size', 'total_influence']:  # Assuming these metrics need log scale due to their nature
            ax.set_yscale('log')
        else:
            ax.autoscale_view()  # Autoscale view to adjust for non-logarithmic data

        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Box Plot of Metrics by Supply Threshold Level', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save:
        plt.savefig(output_path, format='pdf')
    if show:
        plt.show()
