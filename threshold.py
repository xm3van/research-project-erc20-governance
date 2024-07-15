import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from dotenv import load_dotenv
import datetime 

from src.utilities.metrics_and_tests import gini

load_dotenv()  

path = os.environ['DATA_DIRECTORY']
df_snapshots = pd.read_csv('data/snapshot_selection.csv')
df_tokens = pd.read_csv("data/final_token_selection.csv")
df_token_price = pd.read_csv("data/price_table.csv", index_col=0)

TOKEN_BALANCE_TABLE_INPUT_PATH = join(path, "data/snapshot_token_balance_tables_enriched")
START_BLOCK_HEIGHT = 11659570
SUPPLY_THRESHOLDS = [0.1, 0.001, 0.0001, 0.000001, 0.0000001, 0.00000001, 0]
SUPPLY_THRESHOLDS = np.concatenate(([0], np.logspace(-8, -1, num=70)))

# Remove burner addresses 
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

def gini_for_threshold():
    gini_results = []

    for _, row in df_snapshots[df_snapshots['Block Height'] >= START_BLOCK_HEIGHT].iterrows():
        snapshot_date = row['Date']
        snapshot_block_height = row['Block Height']

        print(f"Snapshot for Block Height: {snapshot_block_height} - {datetime.datetime.now()}")

        # Load data
        ddf = pd.read_csv(join(TOKEN_BALANCE_TABLE_INPUT_PATH, f'token_holder_snapshot_balance_labelled_{snapshot_block_height}.csv'))
        
        # Ensure we only check for tokens we want to analyze
        ddf = ddf[ddf.token_address.str.lower().isin(df_tokens.address.str.lower())]

        # Remove burner addresses
        ddf = ddf[~ddf.address.isin(known_burner_addresses)]

        for token in df_tokens['address']:
            ddf_token = ddf[ddf['token_address'].str.lower() == token.lower()]

            for SUPPLY_THRESHOLD in SUPPLY_THRESHOLDS:

                ddf_filtered = ddf_token[ddf_token.pct_supply >= SUPPLY_THRESHOLD]

                if len(ddf_filtered) == 0:
                    continue  # Skip if no addresses meet the threshold

                gini_value = gini(ddf_filtered['pct_supply'])
                gini_results.append((snapshot_date, token, SUPPLY_THRESHOLD, gini_value))

    return gini_results

if __name__ == "__main__":
    gini_results = gini_for_threshold()  # Store the Gini results

    # Save the Gini results and metrics
    gini_df = pd.DataFrame(gini_results, columns=['Date', 'Token', 'Threshold', 'Gini Coefficient'])
    gini_df.to_csv(join(path, 'output/threshold/gini_coefficients.csv'), index=False)
    