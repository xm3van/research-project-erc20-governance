from src.preprocessing.data_preprocessor import process_token_transfers, enrich_token_balance_tables, enrich_token_balance_tables_v2

import os
from os.path import join


from dotenv import load_dotenv
load_dotenv()  


def main():
    DATA_DIRECTORY = os.environ["DATA_DIRECTORY"] 
    snapshot_selection_csv_path = "./data/snapshot_selection.csv" 
    token_list_csv_path = 'data/final_token_selection.csv'
    token_transfers_csv_path = join(DATA_DIRECTORY, "data/token_transfers_22021209.csv")
    snapshot_token_balance_tables_dir = join(DATA_DIRECTORY, "data/snapshot_token_balance_tables")
    snapshot_token_balance_tables_enriched_dir = join(DATA_DIRECTORY, "data/snapshot_token_balance_tables_enriched_v2")
    
    # Generates tokenholder lookup tables
    # process_token_transfers(snapshot_selection_csv_path, token_transfers_csv_path, snapshot_token_balance_tables_dir)
    
    # Enriches lookup tables with labels and pct_supply 
    # enrich_token_balance_tables(snapshot_selection_csv_path=snapshot_selection_csv_path, 
    #                              token_list_csv_path=token_list_csv_path, 
    #                              token_balance_dir=snapshot_token_balance_tables_dir,
    #                              output_dir=snapshot_token_balance_tables_enriched_dir,
    #                              from_block_height=11659570) #11547458
 
    enrich_token_balance_tables_v2(snapshot_selection_csv_path=snapshot_selection_csv_path, 
                                token_list_csv_path=token_list_csv_path, 
                                label_list_csv_path='./data/address_labels/final_labels.csv', 
                                token_balance_dir=snapshot_token_balance_tables_dir,
                                output_dir=snapshot_token_balance_tables_enriched_dir,
                                from_block_height=11659570) #11547458


    
if __name__ == "__main__":
    main()
