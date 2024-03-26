import pandas as pd
import numpy as np

from os.path import join



def load_preparations():
    
    # Snapshot selection
    df_snapshot = pd.read_csv("../assets/snapshot_selection.csv")
    snapshots = df_snapshot[df_snapshot['Block Height'] >= 11547458]['Block Height'].values

    # Address selection
    df_addresses = pd.read_csv("../assets/df_final_token_selection_20221209.csv")
    token_addresses = df_addresses.address.values
    
    return snapshots, token_addresses

def load_snapshot_frame(snapshot, path, token_addresses, cut_off=0.000005): 
    # Burner addresses (remove burner addresses)
    known_burner_addresses = [
        "0x0000000000000000000000000000000000000000",
        "0x0000000000000000000000000000000000000000",
        "0x0000000000000000000000000000000000000001",
        "0x0000000000000000000000000000000000000002",
        "0x0000000000000000000000000000000000000003",
        "0x0000000000000000000000000000000000000004",
        "0x0000000000000000000000000000000000000005",
        "0x0000000000000000000000000000000000000006",
        "0x0000000000000000000000000000000000000007",
        "0x000000000000000000000000000000000000dead",
    ]
    
    # Load data
    df = pd.read_csv(join(path, f"token_balance_lookup_tables_labelled/df_token_balenace_labelled_greater_01pct_bh{snapshot}_v2.csv"))
    
    # format 
    df = df.rename(columns={'address_x': 'address'})
    
    # filter
    df = df[df.value > 0]
    df = df[df.token_address.isin(token_addresses)]
    df = df[~df.address.isin(known_burner_addresses)]
    
    # pct_supply filter 
    dict_ts = dict(df.groupby('token_address').value.sum())
    
    df['pct_supply'] = 0
    
    for t in dict_ts.keys(): 
        
        df.loc[df.token_address == t, 'pct_supply'] = df[df.token_address == t].value / dict_ts[t]
        
    df = df[df.pct_supply > cut_off] 
    
    
    return df
    
    