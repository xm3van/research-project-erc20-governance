import pandas as pd 
import dotenv
from dotenv import load_dotenv
import os
import numpy as np
import h5py
from os.path import join 
import matplotlib.pyplot as plt
import requests 
import datetime

from statsmodels.stats.multitest import multipletests as m_tests
from data_loading.load_data import load_preparations, load_snapshot_frame
from data_processing.token_price import convert_blockheight_to_date, usd_value_decimals_token



load_dotenv()

path = os.environ["PROJECT_PATH"]
covalent_api_key = os.environ["COVALENTHQ_API_KEY"]
etherscan_api =  os.environ["ETHERSCAN_API_KEY"]

# Snapshot selection
df_snapshot = pd.read_csv("../assets/snapshot_selection.csv")


for snapshot in df_snapshot[df_snapshot['Block Height'] > 11547458]['Block Height']: #11547458
    print(f"Snapshot for Block Height: {snapshot}") 
    
    
    ### BUILD p_value df from .h5 file
    file_path = f'wallet_projection_output/output_f-none/output_snapshot_{snapshot}.h5'

    # Open the HDF5 file for reading
    with h5py.File(join(path,file_path), 'r') as file:
        # Access the p_values_group
        p_values_group = file[f'snapshot_{snapshot}/p_values']
    
        # Create an empty list to store the batches
        batches = []
    
        # Iterate over the datasets in p_values_group
        for dataset_name in p_values_group:
            dataset = p_values_group[dataset_name]
    
            # Access the data in the dataset
            data = dataset[:]
    
            # Append the batch data to the list
            batches.append(data)
    
        # Concatenate the batches into a single DataFrame
        p_values = np.concatenate(batches)
        
    df_p_values = pd.DataFrame(p_values)

    # Rename the columns
    df_p_values = df_p_values.rename(columns={0: 'address1', 1: 'address2', 2: 'p_value'})

    # Convert the 'p_value' column to float
    df_p_values['p_value'] = df_p_values['p_value'].astype(float)

    df_p_values['address1'] = df_p_values['address1'].str.decode('utf-8')

    df_p_values['address2'] = df_p_values['address2'].str.decode('utf-8')
    
    ## Correct for bonferri correcton 
    m_test = m_tests(pvals=df_p_values.p_value.values, alpha=0.01, method='bonferroni')
    
    df_p_values['m_test_result'] = m_test[1]
    
    
    ### CALCULATE WEIGHTED PORTFOLIO
    snapshots, assets = load_preparations()
    
    ## load data frame
    ddf = load_snapshot_frame(snapshot, path, assets)
    
    
    #### CREATE HISTORIC PRICE LIST 
    price_list = {}

    for token_address in ddf.token_address.unique(): 
        token_price_usd = usd_value_decimals_token(snapshot=snapshot, contractAddress=token_address, api_key=covalent_api_key, chainName='eth-mainnet',quoteCurrency='USD')['token_price_usd'] 

        price_list[token_address] = token_price_usd
        
    #### MODIFY DDF
    
    # Drop the 'Unnamed: 0' column
    ddf = ddf.drop('Unnamed: 0', axis=1)
    
    ddf['contract_decimals'] = int
    ddf['token_units'] = float
    ddf['token_price_usd'] = float
    
    data = []

    for contract_address in ddf.token_address.unique(): 

        response = usd_value_decimals_token(snapshot=snapshot, contractAddress=contract_address, api_key=covalent_api_key, chainName='eth-mainnet',quoteCurrency='USD')

        ddf.loc[ddf.token_address == contract_address, 'contract_decimals'] = response['contract_decimals']
        ddf.loc[ddf.token_address == contract_address, 'token_price_usd'] = response['token_price_usd']


    ddf['token_units'] = ddf['value']/(10**ddf['contract_decimals'])
    ddf['token_units'] = ddf['token_units'].astype(float)

    ddf['value_usd'] = ddf['token_units']*ddf['token_price_usd']
    ddf['value_usd'] = ddf['value_usd'].astype(float)
    
    
    ddf['address_value_usd'] = ddf.groupby('address')['value_usd'].transform('sum')
    ddf['degree'] = ddf.groupby('address')['token_address'].transform('count')
    ddf['inv_degree'] = 1/ddf.groupby('address')['token_address'].transform('count')
    ddf['wi(t)'] = ddf['value_usd'] / ddf['address_value_usd']
    
    ### load token validation
    df_token_projections = pd.read_csv(join(path, 'token_projection_output/output_f-none_cutoff/pvalues_11659570_none.csv'), index_col=0)
    df_token_projections_validated = df_token_projections[df_token_projections.m_test_result == True]
    df_token_projections_validated.reset_index(inplace=True)
    df_token_projections_validated[['token1', 'token2']] = df_token_projections_validated.combination.str.strip("()").str.replace("'", "").str.split(", ", expand=True)
    
    
    # Get unique token addresses from 'token1' column
    unique_token1_addresses = np.unique(df_token_projections_validated['token1'])

    # Get unique token addresses from 'token2' column
    unique_token2_addresses = np.unique(df_token_projections_validated['token2'])

    # Combine the unique token addresses from both columns
    unique_token_addresses = np.unique(np.concatenate((unique_token1_addresses, unique_token2_addresses)))
    
    ddf['holds_validated_security'] = ddf.groupby('address', group_keys=False)['token_address'].apply(lambda x: x.isin(unique_token_addresses))
    
    
    ### FIGURE 6

    # Calculate the average inverse degree
    average_inverse_degree = 1 / np.mean(ddf['degree'])

    # Separate the data based on the "holds_validated_security" column
    non_overlapping = ddf[ddf['holds_validated_security']]
    overlapping = ddf[~ddf['holds_validated_security']]

    # Create scatter plot for overlapping portfolios
    plt.scatter(np.log10(overlapping['inv_degree']), np.log10(overlapping['wi(t)']),
                label='Overlapping Portfolios', s=1, color='red')

    # Create scatter plot for non-overlapping portfolios
    plt.scatter(np.log10(non_overlapping['inv_degree']), np.log10(non_overlapping['wi(t)']),
                label='Non-Overlapping Portfolios', s=1, color='blue')

    # Plot the average inverse degree line
    x = np.linspace(np.min(np.log10(ddf['inv_degree'])), np.max(np.log10(ddf['inv_degree'])), 100)
    y = average_inverse_degree * x
    plt.plot(x, y, linestyle='--', color='black', linewidth=1, label='Average Inverse Degree')

    # Set labels and title
    plt.xlabel('Inverse Portfolio Diversification (1/di(t))')
    plt.ylabel('Average Share of Securities Market Value (wi(t))')
    plt.title('Portfolio Diversification vs. Avg. Share Securities Market Value')

    # Add legend
    plt.legend()

    # Save the figure
    plt.savefig(join('output', f"F6_PDvsSMV_{snapshot}.png"))

    # Clear the figure
    plt.clf()
    
    
    
    
    ### FIGURE 7 
    
    #### GET ASSET MCAP

    # get coingecko id
    df_token = pd.read_csv("../assets/df_final_token_selection_20221209.csv")
    address_cgId_dict = dict(zip(df_token.address,df_token.coingecko_id))

    # convert snapshot to DD-MM_YYYY using Etherscan API
    r = requests.get(f'https://api.etherscan.io/api?module=block&action=getblockreward&blockno={snapshot}&apikey={etherscan_api}')
    snapshot_timeStamp = int(r.json()['result']['timeStamp'])
    snapshot_date = str(datetime.datetime.fromtimestamp(snapshot_timeStamp).strftime('%d-%m-%Y'))

    # get historical mcap
    mcap_dict ={}
    for token_address in np.unique(ddf.token_address): 
        r = requests.get(url=f'https://api.coingecko.com/api/v3/coins/{address_cgId_dict[token_address]}/history?date={snapshot_date}')
        mcap = r.json()['market_data']['market_cap']['usd']
        mcap_dict[token_address] = mcap
    
    # mcap
    ddf['mcap'] = ddf.token_address.apply(lambda x: mcap_dict[x])
    # ddf['mcap'] = ddf.groupby('address')['value_usd'].transform('sum')
    
    
    ####  MODIFY DDF
    ddf['circulating_supply'] = ddf.groupby('token_address').token_units.transform('sum')
    ddf['degree_i'] = ddf.groupby('address')['token_address'].transform('count')
    ddf['degree_s'] = ddf.groupby('token_address')['address'].transform('count')
    ddf['cis(t)/d(t)'] = ddf['pct_supply'] /ddf['degree_s']
    ddf['nuniq_token_holder'] = ddf.groupby('token_address')['address'].transform('count')
    ddf['possible_links'] = (ddf['nuniq_token_holder']*(ddf['nuniq_token_holder']-1))/2
    
    #### fs(t)
    df_val = df_p_values[df_p_values.m_test_result == 0]
    observed_links_count = df_val.groupby('address1')['address2'].nunique().add(df_val.groupby('address2')['address1'].nunique(), fill_value=0)
    ddf = ddf.merge(observed_links_count.rename('observed_links').groupby(level=0).sum(), left_on='address', right_index=True)
    ddf['fs(t)'] = ddf['observed_links']/ddf['possible_links']
    
    
    #### RECREATE FIG 7
    # Create a figure with three subplots arranged horizontally
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left subplot - Fraction fs(t) vs. Security Capitalization
    axes[0].scatter(np.log10(ddf['mcap']), ddf['fs(t)'])
    axes[0].set_xlabel('Log10 Security Capitalization')
    axes[0].set_ylabel('Fraction of Val. Overlapping Portfolio Pairs (fs(t))')
    axes[0].set_title('Scatter Plot: Fraction fs(t) vs. Security Capitalization')

    # Middle subplot - Fraction fs(t) vs. Concentration
    axes[1].scatter(ddf['cis(t)/d(t)'], ddf['fs(t)'])
    axes[1].set_xlabel('Concentration (cis(t)/d(t))')
    axes[1].set_ylabel('Fraction of Val. Overlapping Portfolio Pairs (fs(t))')
    axes[1].set_title('Scatter Plot: Fraction fs(t) vs. Concentration')

    # Right subplot - Fraction fs(t) vs. Number of Owners
    axes[2].scatter(np.log10(ddf['degree_s']), ddf['fs(t)'])
    axes[2].set_xlabel('Log10 Number of Owners (degree_s)')
    axes[2].set_ylabel('Fraction of Val. Overlapping Portfolio Pairs (fs(t))')
    axes[2].set_title('Scatter Plot: Fraction fs(t) vs. Number of Owners')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(join('output', f"F7_CCO_{snapshot}.png"))

    # Clear the figure
    plt.clf()

    
    


    


        
    
    
    


