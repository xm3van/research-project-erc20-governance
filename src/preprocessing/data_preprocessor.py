import pandas as pd
import numpy as np
import os
from os.path import join

from dotenv import load_dotenv
load_dotenv()  

from web3 import Web3

from src.preprocessing.data_processing_utils import * 


###################################
### CREATE TOKEN BALANCE TABLES ###
###################################

def process_token_transfers(snapshot_selection_csv_path, token_transfers_csv_path, output_dir):
    """
    Processes token transfers and snapshot block heights to compute token holder balances per snapshot.

    Args:
        snapshot_selection_csv_path (str): Path to the CSV file containing snapshot selections.
        token_transfers_csv_path (str): Path to the CSV file containing token transfers data.
        output_dir (str): Directory where output CSV files will be stored.
    """

    # Load all token transfers
    ddf = pd.read_csv(token_transfers_csv_path)

    # Set value as float64
    ddf['value'] = ddf['value'].astype("float64")

    # Load all snapshots block heights
    df_snapshots = pd.read_csv(snapshot_selection_csv_path)

    # Iterate over snapshots
    for snapshot_block_height in df_snapshots["Block Height"]:
        print(f"Snapshot Block Height: {snapshot_block_height}")

        # Filter ddf to snapshot height
        ddf_snapshot = ddf[ddf.block_number <= snapshot_block_height]

        # Compute sums for outflows and inflows
        ddf_outflows = (
            ddf_snapshot.groupby(["token_address", "from_address"])['value']
            .sum()
            .compute()
            .reset_index()
            .assign(value=lambda x: x['value'] * -1)  # Negate the outflow values
            .rename(columns={'from_address': 'address'})
        )

        ddf_inflows = (
            ddf_snapshot.groupby(["token_address", "to_address"])['value']
            .sum()
            .compute()
            .reset_index()
            .rename(columns={'to_address': 'address'})
        )

        # Stack DataFrames
        df_all = pd.concat([ddf_inflows, ddf_outflows], ignore_index=True)

        # Sum to get final balance snapshot
        df_token_holder_balance_snapshot = df_all.groupby(["address", "token_address"])['value'].sum()

        # Store tokenholder df as csv file
        output_path = join(output_dir, f"token_holder_snapshot_balance_{snapshot_block_height}.csv")
        df_token_holder_balance_snapshot.to_csv(output_path)
        print(f"Output stored: {output_path}")
        
        
###################################
### ENRICH TOKEN BALANCE TABLES ###
###################################


def enrich_token_balance_tables(snapshot_selection_csv_path, 
                                token_list_csv_path, 
                                token_balance_dir, 
                                output_dir, 
                                from_block_height=11547458):
    
    # Load all snapshots block heights
    df_snapshot_points = pd.read_csv(snapshot_selection_csv_path, index_col=0)
    
    df_token_list = pd.read_csv(token_list_csv_path, index_col=0)
    
    df_kaggle, cefi_labels, df_lp_pairs, df_label = load_label_datasets()
    
    reference_bytecode_dict = get_reference_bytecode()
    
    w3 = Web3(Web3.HTTPProvider(PUBLIC_RPC))
    
    
    known_burner_addresses = ['0x0000000000000000000000000000000000000000',
                        '0x0000000000000000000000000000000000000000',
                        '0x0000000000000000000000000000000000000001',
                        '0x0000000000000000000000000000000000000002',
                        '0x0000000000000000000000000000000000000003',
                        '0x0000000000000000000000000000000000000004',
                        '0x0000000000000000000000000000000000000005',
                        '0x0000000000000000000000000000000000000006',
                        '0x0000000000000000000000000000000000000007',
                        '0x000000000000000000000000000000000000dead']



    
    for snapshot_block_height in df_snapshot_points[df_snapshot_points['Block Height'] >= from_block_height]['Block Height']:
        
        print(f"Commencing enriching of Token Balance Table {snapshot_block_height}")
        
        # Load relevant token balances for a given snapshot date
        df_token_balance = pd.read_csv(join(token_balance_dir, f'token_holder_snapshot_balance_{snapshot_block_height}.csv'))
        df_token_balance = df_token_balance[df_token_balance.token_address.isin(df_token_list.address)]
        
        # Filter negative balances 
        df_token_balance = df_token_balance[df_token_balance.value > 0].copy()
        
        # remove know burner addresses - Note: We remove them before we calculated the supply
        df_token_balance = df_token_balance[~df_token_balance.address.isin(known_burner_addresses)]
        
        # Update checksum address
        df_token_balance['address_checksum'] = df_token_balance.address.apply(lambda x: w3.to_checksum_address(x))
        
        # Label address check - get smart contract code for addresses in question

        df_sc = pd.read_csv(join(token_balance_dir, '../unique_address_with_code.csv'), index_col=[0]).rename(columns={'address': 'address_checksum'})
        df_tb_sc = df_token_balance.merge(df_sc, how='left', on='address_checksum')
        
        def reassign_labels(label_df, new_label=None):
            # to-do: remove in the future
            for index, row in label_df.iterrows():
                condition = (df_tb_sc['address_checksum'] == row.get('checksum_address')) | \
                            (df_tb_sc['address_checksum'] == row.get('address_checksum'))
                df_tb_sc.loc[condition, 'label'] = row.standardised_labelling if new_label is None else new_label

        # Labeling
        df_tb_sc['label'] = df_tb_sc.code.apply(lambda x: bytcode_comparison(x, reference_bytecode_dict))
        
        reassign_labels(cefi_labels, 'IEMOA')
        reassign_labels(df_kaggle)
        reassign_labels(df_lp_pairs, 'lp_amm')
        reassign_labels(df_label)

        
        # Calculate percentage of accessible supply
        df_tb_sc['pct_supply'] = df_tb_sc.groupby('token_address')['value'].transform(lambda x: x / x.sum())
        
        # Save output
        df_tb_sc.to_csv(join(output_dir,       
        f'token_holder_snapshot_balance_labelled_{snapshot_block_height}.csv'))
        
        print(f"Completed enriching of Token Balance Table {snapshot_block_height}")


        
