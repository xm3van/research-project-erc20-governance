import pandas as pd
import numpy as np
import ast
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

#####################################
########### Clique Size #############
#####################################


def plot_clique_size_over_time(metric_dataframes, method='upper_bound', group='sample', output_path="output/cliques/", save=True, show=True):
    
    # Define df 
    df = metric_dataframes[method][group]['size_clique']
    df_influenece = metric_dataframes[method][group]['total_influence']

    
    # common index
    df_index = metric_dataframes['upper_bound']['sample']['size_clique']

    # Find the index of the first occurrence (value > 1) in each column (snapshot)
    first_occurrence_indices = (df_index.T > 1).idxmax()

    # Determine the minimum index (earliest occurrence) for each clique across all snapshots
    min_indices = first_occurrence_indices.groupby(first_occurrence_indices.index).min()

    # Sort the cliques based on their minimum indices to get the desired order
    cliques_order = min_indices.sort_values().index.tolist()

    # Reindex clique size to df_presence 
    df = df.reindex(cliques_order)

    # Create a mask for cells where 'clique_size' is greater than 1
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
    ax.set_yticklabels(df.index, fontsize=FONT_SIZE_TICK, va='center', linespacing=LINE_SPACING)

    # Annotate the chart with 'clique_size' values, skipping 'NaN' values
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, (str(round(value))), ha='center', va='center', color='white', fontsize=FONT_SIZE_VALUE)

    # Set labels and title
    plt.xlabel('Timestamps', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Cliques', fontsize=FONT_SIZE_LABEL)
    plt.title(f'[{method.upper()}]: Clique Size Over Time', fontsize=FONT_SIZE_TITLE)

    # Adjust spacing for vertical axis labels
    plt.tight_layout()
    
    # Save the plot to the specified output path
    if save == True:
        
        plt.savefig(join(output_path, f'clique_size_over_time_{method}_{group}.png'))
        
    if show != True:
        
        plt.close(fig)
    
    else: 
        
        plt.show() 
        
#####################################
######### Clique Growth #############
#####################################

def plot_clique_growth_over_time(metric_dataframes, method='upper_bound', group='sample', output_path="output/cliques/", save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (12, 8)

    # Extract data
    df = metric_dataframes[method][group]['size_clique']

    # Sort cliques by their average size
    cliques_order = df.mean(axis=1).sort_values(ascending=False).index
    df = df.reindex(cliques_order)

    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Plot each clique's size over time
    for clique in cliques_order:
        ax.plot(df.columns, df.loc[clique], marker='o', linestyle='-', label=f'Clique {clique}')

    # Labels and Title
    ax.set_xlabel("Time")
    ax.set_ylabel("Clique Size")
    ax.set_title("Clique Size Over Time")

    # Set tick positions and labels
    ax.set_xticks(np.arange(len(df.columns)))  # Set tick positions
    ax.set_xticklabels(df.columns, rotation=90)  # Set tick labels and rotate for better readability

    # Add legend
    ax.legend(title="Cliques", loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}/clique_growth_over_time_{method}_{group}.png", bbox_inches='tight')

    if show:
        plt.show()

        
        

#####################################
###### Clique Growth Rate ###########
#####################################

def plot_clique_growth_rate_over_time(metric_dataframes, method='upper_bound', group='sample', output_path="output/cliques/", save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (12, 8)
    MEDIAN_LINE_STYLE = {'color': 'black', 'linewidth': 2, 'linestyle': '--', 'label': 'Median Growth Rate'}

    # Extract data
    df = metric_dataframes[method][group]['size_clique']

    # Ensure columns are datetime objects and sort them
    df.columns = pd.to_datetime(df.columns)
    df = df.sort_index(axis=1)

    # Prepare a DataFrame to store growth rates
    growth_rate_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Calculate growth rates for each clique based on available values
    for clique, values in df.iterrows():
        available_values = values.dropna()
        if len(available_values) > 1:
            growth_rates = available_values.pct_change().dropna()
            growth_rate_df.loc[clique, growth_rates.index] = growth_rates

    # Filter cliques that occur less than 4 times
    valid_cliques = growth_rate_df.dropna(thresh=4).index
    filtered_growth_rate_df = growth_rate_df.loc[valid_cliques]

    # Calculate median growth rates over time
    median_growth_rate = filtered_growth_rate_df.median(axis=0).dropna()

    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Plot available growth rates for each clique
    for clique in valid_cliques:
        clique_growth_rates = filtered_growth_rate_df.loc[clique].dropna()
        ax.plot(clique_growth_rates.index, clique_growth_rates, marker='o', linestyle='-', label=f'Clique {clique}')

    # Plot median growth rate over time
    ax.plot(median_growth_rate.index, median_growth_rate, **MEDIAN_LINE_STYLE)

    # Labels and Title
    ax.set_xlabel("Time")
    ax.set_ylabel("Growth Rate")
    ax.set_title("Growth Rate of Cliques Over Time")

    # Set tick positions and labels
    ax.set_xticks(median_growth_rate.index)  # Set tick positions
    ax.set_xticklabels(median_growth_rate.index.strftime('%Y-%m-%d'), rotation=90)  # Set tick labels and rotate for better readability

    # Add legend
    ax.legend(title="Cliques", loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}/growth_rate_over_time.png", bbox_inches='tight')

    if show:
        plt.show()

        
#####################################
#### Stability vs. No. of Tokens ####
#####################################

def plot_clique_stability_vs_no_of_tokens(metric_dataframes, method='upper_bound', group='sample', output_path="output/cliques/", save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (10, 6)

    # Extract data
    df = metric_dataframes[method][group]['size_clique']

    # Calculate the mean size and stability (variance) for each clique
    no_of_tokens = np.array(([len(ast.literal_eval(clique)) for clique in df.index]))
    
    stability = np.array(df.notna().astype(int).mean(axis=1)) # note binary measure of stability

    # Prepare figure and axis for plotting
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Create a scatter plot of stability versus mean size
    ax.scatter(no_of_tokens, stability)
    
    # Labels and Title
    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("Clique Stability")
    ax.set_title("Clique Stability vs. Number of Tokens")

    # Show the correlation value on the plot
    correlation = np.corrcoef(no_of_tokens, stability)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

    plt.tight_layout()

    # Save or show the figure
    if save:
        plt.savefig(f"{output_path}/clique_stability_vs_size_{method}_{group}.png", bbox_inches='tight')
    if show:
        plt.show()

        
#####################################
###### Stability vs Size ############
#####################################
        
        
def plot_clique_stability_vs_size(metric_dataframes, method='upper_bound', group='sample', output_path="output/cliques/", save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (10, 6)

    # Extract data
    df = metric_dataframes[method]['sample']['size_clique']

    # Calculate the mean size and stability (variance) for each clique
    mean_size = df.mean(axis=1)
    
    stability = df.notna().astype(int).var(axis=1) # note binary measure of stability

    # Prepare figure and axis for plotting
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Create a scatter plot of stability versus mean size
    ax.scatter(mean_size, stability)
    
    # Labels and Title
    ax.set_xlabel("Mean Clique Size")
    ax.set_ylabel("Clique Stability (Variance)")
    ax.set_title("Clique Stability vs. Size")

    # Show the correlation value on the plot
    correlation = mean_size.corr(stability)
    ax.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

    plt.tight_layout()

    # Save or show the figure
    if save:
        plt.savefig(f"{output_path}/clique_stability_vs_size_{method}_{group}.png", bbox_inches='tight')
    if show:
        plt.show()
        
        
        
        
        
#####################################
###### Key heatmap chart ############
#####################################
        
        
def plot_heatmap_chart(metric_dataframes, metric_name, pct=True, log=True, output_path="./output/cliques/", save=False, show=True):

    if pct==True: 
        multiplier = 100 
        unit='%'
    else: 
        multiplier = 1
        unit=''
        
    
    # Define df
    
    if log == True: 
        df_u = np.log10(metric_dataframes['upper_bound']['sample'][metric_name]) * multiplier
        df_l = np.log10(metric_dataframes['upper_bound']['sample'][metric_name]) * multiplier
        
    else: 
        df_u = metric_dataframes['upper_bound']['sample'][metric_name] * multiplier
        df_l = metric_dataframes['lower_bound']['sample'][metric_name] * multiplier

    df_pv_u = metric_dataframes['upper_bound']['pvalues'][metric_name] 
    df_pv_l = metric_dataframes['lower_bound']['pvalues'][metric_name]
    
    # reindex
    df_index = metric_dataframes['upper_bound']['sample'][metric_name] 

        # Find the index of the first occurrence (value > 1) in each column (snapshot)
    first_occurrence_indices = (df_index.T > 0).idxmax()
        # Determine the minimum index (earliest occurrence) for each clique across all snapshots
    min_indices = first_occurrence_indices.groupby(first_occurrence_indices.index).min()
        # Sort the cliques based on their minimum indices to get the desired order
    cliques_order = min_indices.sort_values().index.tolist()

    # Reindex clique size to df
    df_u = df_u.reindex(cliques_order[::-1])
    df_pv_u =df_pv_u.reindex(cliques_order)

    fig, ax = plt.subplots(figsize=FIG_SIZE)


    # Create colormap
    # cmap = mcolors.LinearSegmentedColormap.from_list("CMRmap", plt.get_cmap("CMRmap")(np.linspace(0, 1, 256)))
    scaling_factor = df_u.max().max()*1.2
    cmap = plt.get_cmap("magma", lut=128)
    norm = mcolors.Normalize(vmin=0, vmax=scaling_factor)

    cell_width = 1  # as we're not using imshow, we can stick with unit width and height
    cell_height = 1

    for clique in df_u.index:
        for date in df_u.columns:
            value_u = df_u.loc[clique, date]
            pval_u = df_pv_u.loc[clique, date]
            pval_u = pval_to_significance(pval_u)

            try: 
                value_l = df_l.loc[clique, date]
                pval_l = df_pv_l.loc[clique, date]
                pval_l = pval_to_significance(pval_l)
            except: 
                pval_l = ''
                value_l = np.nan

            x_pos = df_u.columns.get_loc(date) * cell_width
            y_pos = df_u.index.get_loc(clique) * cell_height

            # Plotting the triangles
            if not np.isnan(value_u):
                # Calculate triangle vertices for 90-degree rotation
                x1, y1 = x_pos, y_pos + cell_height
                x2, y2 = x_pos, y_pos
                x3, y3 = x_pos + cell_width, y_pos + cell_height

                # Draw triangle
                upper_triangle = patches.Polygon([(x1, y1), (x2, y2), (x3, y3)], closed=True, color=cmap(norm(value_u)), zorder=1)  
                ax.add_patch(upper_triangle)

                # Calculate centroid
                C_x = (x1 + x2 + x3) / 3 + 0.1 #last term manual adjustment 
                C_y = (y1 + y2 + y3) / 3 + 0.1 #last term manual adjustment

                # Place text at centroid
                ax.text(C_x, C_y, f'U:{round(value_u,1) if not np.isnan(value_u) else ""}{unit}{pval_u}', ha='center', va='center', color='white', fontsize=FONT_SIZE_VALUE)


            if not np.isnan(value_l):
                # Calculate triangle vertices for 90-degree rotation
                x1, y1 = x_pos + cell_width, y_pos
                x2, y2 = x_pos + cell_width, y_pos + cell_height
                x3, y3 = x_pos, y_pos 

                # Draw triangle
                lower_triangle = patches.Polygon([(x1, y1), (x2, y2), (x3, y3)], closed=True, color=cmap(norm(value_l)), zorder=2)
                ax.add_patch(lower_triangle)

                # Calculate centroid
                C_x = (x1 + x2 + x3) / 3 - 0.1 #last term manual adjustment 
                C_y = (y1 + y2 + y3) / 3 - 0.1 # last term manual adjustment 

                # Place text at centroid
                ax.text(C_x, C_y, f'L:{round(value_l,1) if not np.isnan(value_l) else ""}{unit}{pval_l}', ha='center', va='center', color='white', fontsize=FONT_SIZE_VALUE)
     
    #CBAR 
    ## Adding the colorbar for the colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)  # '2%' determines the width of the colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
    ## add pct sign to tick vals of cbar 
    tick_vals = np.array(cbar.get_ticks()) 
    cbar.set_ticklabels([f'{round(val)}%' for val in tick_vals])
    
    # Significance box
    plt.text(1.16, 0.98, 'Relative to Control:\n* = 0.1\n** = 0.05\n*** = 0.01', 
         transform=ax.transAxes, fontsize=FONT_SIZE_TEXT, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='lightyellow', alpha=1, pad=12))

    
    ax.set_xlim(0, len(df_u.columns))
    ax.set_ylim(0, len(df_u.index))
    ax.set_xlabel('Date', size=FONT_SIZE_LABEL)
    ax.set_ylabel('Cliques', size=FONT_SIZE_LABEL)
    ax.set_title(f'{metric_name.replace("_", " ").title()}', size=FONT_SIZE_TITLE)
    ax.set_xticks(np.arange(len(df_u.columns))+0.5)
    ax.set_xticklabels(df_u.columns, rotation=90, ha='center', size=FONT_SIZE_TEXT)
    ax.set_yticks(np.arange(len(df_u.index))+0.5)
    ax.set_yticklabels(df_u.index, size=FONT_SIZE_TEXT)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}/{metric_name}_clique.png", bbox_inches='tight')

    if show:
        plt.show()
        
        
        
        
        
#####################################
###### Key Boxplot Chart ############
#####################################

        
def plot_boxplot_with_significance(metric_dataframes, metric, unit, method='upper_bound', group='sample', output_path="output/cliques/", save=True, show=True):
    # Constants for aesthetics
    FIG_SIZE = (12, 8)
    COLOR_MAP = {'non-significant': 'lightgray', '0.05': 'yellow', '0.01': 'orange', '0.001': 'red'}

    # Extract data
    df = metric_dataframes[method][group][metric]
    
    # load p_values to control
    df_pvalues = metric_dataframes[method]['pvalues'][metric]      

    # Sort cliques by average influence
    cliques_order = df.mean(axis=1).sort_values(ascending=False).index
    df = df.reindex(cliques_order)
    df_pvalues = df_pvalues.reindex(cliques_order)

    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Create boxplot data
    boxplot_data = [df.loc[clique].dropna() for clique in cliques_order]
    boxplot = ax.boxplot(boxplot_data, vert=False, patch_artist=True)

    # Color and annotate based on significance
    for i, clique in enumerate(cliques_order):
        p_values = df_pvalues.loc[clique].dropna()
        # Determine the most common significance level based on mode
        try: 
            significance = p_values.apply(lambda x: '0.001' if x < 0.001 else ('0.01' if x < 0.01 else ('0.05' if x < 0.05 else 'non-significant'))).mode()[0]
        except: 
            significance = 'non-significant'
        boxplot['boxes'][i].set_facecolor(COLOR_MAP[significance])

    # Adding legend for significance
    if metric == 'size_clique':
        
        pass
        
    else:
        legend_patches = [patches.Patch(color=color, label=significance) for significance, color in COLOR_MAP.items()]
        ax.legend(handles=legend_patches, title="Significance Levels", loc='upper left', bbox_to_anchor=(1, 1))

    # Labels and Title
    metric_name_formatted = ' '.join(metric.split('_')).title()
    naming_convention_grouping = 'Weakly' if method == 'upper_bound' else 'Strongly'

    ax.set_yticks(np.arange(1, len(cliques_order) + 1))
    ax.set_yticklabels(cliques_order)
    ax.set_ylabel('Cliques')
        
    ax.set_xlabel(f"{metric_name_formatted} {unit}")
 
    ax.set_title(f"{metric_name_formatted} of {naming_convention_grouping} Clique-defining Wallets")
    
        

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}/{metric}_significance_boxplot_{method}_{group}.png", bbox_inches='tight')

    if show:
        plt.show()
        
        
        
#####################################
###### Influence Labels #############
#####################################


def create_and_normalize_matrix(dataframe, label_column='Clique Name', short_labels=None):
    """
    Create a normalized matrix from a dataframe with string-encoded dictionaries of influence labels.

    Args:
        dataframe (pd.DataFrame): The original dataframe.
        label_column (str): The name of the column to use as labels (default: 'Clique Name').
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

    return result_df



def plot_heatmap_labels(metric_dataframes, method='upper_bound', group='sample', colormap='magma', output_path='output/cliques'):
    """
    Plot a heatmap from a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        title (str): The title of the heatmap.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        colormap (str): The colormap to use for the heatmap.
        output_path (str): The directory to save the heatmap.

    Returns:
        None
    """
    
    short_readable_labels = {
    'EMOA': 'Retail Wallets',
    'IEMOA': 'Institutional Wallets',
    'PCV': 'Protocol Wallets',
    'bridge_contract': 'Bridge Contracts',
    'other_contracts': 'Other Contracts',
    'vesting_contract': 'Vesting Contracts',
    'external_staking_contracts': 'Staking Contracts',
    'lp_amm': 'Liquidity Pools',
    'lending_borrowing_contract': 'Lending/Borrowing'
    }
    
    df_raw = metric_dataframes[method][group]['max_influence_label_distribution']
    df_raw.reset_index(inplace=True)

    
    df = create_and_normalize_matrix(df_raw, label_column='Clique Name', short_labels=short_readable_labels)
    
    
    fig, ax = plt.subplots(figsize=(20,16))
    data = df.to_numpy() * 100

    cax = ax.matshow(data, cmap=colormap, aspect='auto')
    fig.colorbar(cax, ax=ax).ax.tick_params(labelsize=14)

    ax.set_xlabel('Labels', fontsize=18)
    ax.set_ylabel('Cliques', fontsize=18)
    
    naming_convention_grouping = 'Weakly' if method == 'upper_bound' else 'Strongly'
    ax.set_title(f'Relative Control for {naming_convention_grouping}-defining Wallets of Total Influence by Label per Clique', fontsize=22)

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90, ha='right', size=16)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, size=16)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)

    text_color_threshold = np.max(data) / 2
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                color='white' if val < text_color_threshold else 'black', fontsize=14)

    plt.savefig(f"{output_path}/label_plot_cliques_{method}.png", bbox_inches='tight')
    plt.show()