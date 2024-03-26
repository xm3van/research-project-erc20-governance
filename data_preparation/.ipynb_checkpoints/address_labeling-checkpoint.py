
import pandas as pd
import numpy as np
import dask.dataframe as dd
from os.path import join
import os

import dotenv
env_var = dotenv.dotenv_values()
from web3 import Web3, HTTPProvider


######## Environment set up 


path = env_var['PROJECT_PATH']
w3 = Web3(Web3.HTTPProvider(env_var['INFURA_API_ENDPOINT']))

# load tokenlist
df_token_list = pd.read_csv('./assets/df_final_token_selection_20221209.csv')

# load snapshot dates 
df_snapshot_points = pd.read_csv('./assets/snapshot_selection.csv')


####### get diff contracts types
code_gnosis = w3.eth.getCode('0xDaB5dc22350f9a6Aff03Cf3D9341aAD0ba42d2a6')
code_eoa = w3.eth.getCode('0x000018bbb8Df8de9e3eaf772dB1c4EEc228EF06c')

code_uni_pairv2 = w3.eth.getCode('0x06da0fd433C1A5d7a4faa01111c044910A184553')
code_uni_pairv3 = w3.eth.getCode('0x8f8EF111B67C04Eb1641f5ff19EE54Cda062f163')
code_balancer = w3.eth.getCode('0xD5e10e8513E33a2867e20ddfc35Ee081CBA57769')

## check if aave v2 token 
code_aave_v0 = w3.eth.getCode('0x7D2D3688Df45Ce7C552E19c27e007673da9204B8')
code_aave_v1 = w3.eth.getCode('0xfC1E690f61EFd961294b3e1Ce3313fBD8aa4f85d')
code_aave_v2 = w3.eth.getCode('0x8dAE6Cb04688C62d939ed9B68d32Bc62e49970b1')
#code_aave_debt_v0 = w3.eth.getCode('') #not did not have debt token - https://docs.aave.com/developers/v/1.0/
code_aave_debt_v1 = w3.eth.getCode('0xfC1E690f61EFd961294b3e1Ce3313fBD8aa4f85d')
code_aave_debt_v2 = w3.eth.getCode('0xD939F7430dC8D5a427f156dE1012A56C18AcB6Aa') # note


######## Functions 

import numpy as np 

def bytcode_comparison(bytecode): 
    
    ## eoa 
    if bytecode == str(code_eoa): 
        return 'EMOA'
    
    elif bytecode == str(code_gnosis): 
        # we treat multisig as EOAs
        return 'EMOA'
    
    
    ## aave ## TASK double check this (I am not sure if collateral rest in these contracts)    
    elif bytecode == str(code_aave_debt_v1): 
        return 'lending_borrowing_contract'
    
    elif bytecode == str(code_aave_debt_v2): 
        return 'lending_borrowing_contract'
    
    elif bytecode == str(code_aave_v0): 
        return 'lending_borrowing_contract'
    
    elif bytecode == str(code_aave_v1): 
        return 'lending_borrowing_contract'
    
    elif bytecode == str(code_aave_v2): 
        return 'lending_borrowing_contract'
    
    
    
    ## lp 
    elif bytecode == str(code_uni_pairv2): 
        return 'lp_amm'
    
    elif bytecode == str(code_balancer): 
        return 'lp_amm' #separation as to lp ratio is customisable we may want to account for this 
    
    
######## load reference data sets 

# kaggle
df_kaggle = pd.read_csv('assets/address_labels/address_labels_kaggle.csv')
## filter misplaced strings
df_kaggle = df_kaggle[df_kaggle['Address'].str.contains('0x') == True]
## standardise to checksum address 
df_kaggle['checksum_address'] = df_kaggle['Address'].apply(lambda x: w3.toChecksumAddress(x))

# cefi 
cefi_labels = pd.read_csv('./assets/address_labels/address_labels_cefi.csv')
## stamdardise to checksum 
cefi_labels['checksum_address'] = cefi_labels.Address.apply(lambda address_str: w3.toChecksumAddress(address_str.replace('\u200b', '').replace('.', '').strip()))


# dex pairs
df_lp_pairs = pd.read_csv('assets/address_labels/dex_lp_pair_addresses.csv')
## standardise to checksum address 
df_lp_pairs['checksum_address'] = df_lp_pairs['hex(pair_address)'].apply(lambda x: w3.toChecksumAddress('0x' + x))

# doxx
df_label = pd.read_csv('assets/address_labels/address_labels_targeted_dox.csv', index_col=[0])


######## Execution 

# select snapshot - start at 10664157 as before that we do not have any visualisation || Note only true for token to token
for snapshot_height in df_snapshot_points[df_snapshot_points['Block Height'] >= 11547458]['Block Height']: 
    
    ## Prepration

    # load relevant token balances for a given snapshot date
    df_tb = pd.read_csv(join(path, f'token_balance_lookup_tables/token_holder_snapshot_balance_{snapshot_height}.csv'))
    
    # check if in tokenlist
    df_tb = df_tb[df_tb.token_address.isin(list(df_token_list.address))]
    
    ## update checksum address
    df_tb['address_checksum'] = df_tb.address.apply(lambda x: w3.toChecksumAddress(x))
    
    # label address check - for this get smart contract code for addresses in question 
    df_sc = pd.read_csv(join(path, 'unique_address_with_code.csv'),index_col=[0])
    df_tb_sc = df_tb.merge(df_sc, how='left', left_on='address_checksum', right_on='address') 
    df_tb_sc.drop(columns=['address_y'], inplace=True)
    
    
    
    ## labeling 
    
    df_tb_sc['label'] = df_tb_sc.code.apply(lambda x: bytcode_comparison(x))

    # <---------- cefi------------> 

    ## iterate over shared values and re-assign label 
    for index, row in cefi_labels[cefi_labels.checksum_address.isin(df_tb_sc.address_checksum) == True].iterrows(): 

        df_tb_sc.loc[df_tb_sc.address_checksum == row.checksum_address, 'label'] = 'IEMOA'
        
    #< -------------kaggle -----------> 
    

    ## iterate over shared values and re-assign label 
    for index, row in df_kaggle[df_kaggle.checksum_address.isin(df_tb_sc.address_checksum) == True].iterrows(): 

        df_tb_sc.loc[df_tb_sc.address_checksum == row.checksum_address, 'label'] = row.standardised_labelling
        
        
    #< -------------lp  -----------> 

    ## iterate over shared values and re-assign label 
    for index, row in df_lp_pairs[df_lp_pairs.checksum_address.isin(df_tb_sc.address_checksum) == True].iterrows(): 

        df_tb_sc.loc[df_tb_sc.address_checksum == row.checksum_address, 'label'] = 'lp_amm'
    
    #< -------------Doxx-----------> 
    ## iterate over shared values and re-assign label 
    for index, row in df_label.iterrows(): 
    
        df_tb_sc.loc[df_tb_sc.address_checksum == row.address_checksum, 'label'] = row.standardised_labelling

        
    # PCT SUPPLY 
    dict_ts = dict(df_tb_sc.groupby('token_address').value.sum())
    df_tb_sc['pct_supply'] = int
    for t in dict_ts.keys(): 
        df_tb_sc.loc[df_tb_sc.token_address == t, 'pct_supply'] = df_tb_sc[df_tb_sc.token_address == t].value / dict_ts[t]
        
        
    #save output 
    output_path = '/local/scratch/exported/governance-erc20/project_erc20_governanceTokens_data/token_balance_lookup_tables_labelled'
    df_tb_sc.to_csv(join(output_path,f'df_token_balenace_labelled_greater_01pct_bh{snapshot_height}_v2.csv'))