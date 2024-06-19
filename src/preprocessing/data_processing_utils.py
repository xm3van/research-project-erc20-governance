import pandas as pd
import numpy as np
import os
from os.path import join

from dotenv import load_dotenv
load_dotenv()  

from web3 import Web3


PUBLIC_RPC = os.environ["PUBLIC_RPC"] 


def get_reference_bytecode():
    """
    Add addtional references 
    """
    
    w3 = Web3(Web3.HTTPProvider(PUBLIC_RPC))

    reference_bytecode = {
        'gnosis_multisig': w3.eth.get_code('0xDaB5dc22350f9a6Aff03Cf3D9341aAD0ba42d2a6'),
        'eoa': w3.eth.get_code('0x000018bbb8Df8de9e3eaf772dB1c4EEc228EF06c'),
        'uni_pairv2': w3.eth.get_code('0x06da0fd433C1A5d7a4faa01111c044910A184553'),
        'uni_pairv3': w3.eth.get_code('0x8f8EF111B67C04Eb1641f5ff19EE54Cda062f163'),
        'balancer': w3.eth.get_code('0xD5e10e8513E33a2867e20ddfc35Ee081CBA57769'),
        'aave_v0': w3.eth.get_code('0x7D2D3688Df45Ce7C552E19c27e007673da9204B8'),
        'aave_v1': w3.eth.get_code('0xfC1E690f61EFd961294b3e1Ce3313fBD8aa4f85d'),
        'aave_v2': w3.eth.get_code('0x8dAE6Cb04688C62d939ed9B68d32Bc62e49970b1'),
        'aave_debt_v2': w3.eth.get_code('0xD939F7430dC8D5a427f156dE1012A56C18AcB6Aa')
    }
    return reference_bytecode


def bytcode_comparison(bytecode, reference_bytecode):
    # eoa
    if bytecode == reference_bytecode['eoa']:
        return 'EMOA'
    elif bytecode == reference_bytecode['gnosis_multisig']:
        # we treat multisig as EOAs
        return 'EMOA'
    
    # aave
    elif bytecode == reference_bytecode['aave_debt_v2']:
        return 'lending_borrowing_contract'
    elif bytecode == reference_bytecode['aave_v0']:
        return 'lending_borrowing_contract'
    elif bytecode == reference_bytecode['aave_v1']:
        return 'lending_borrowing_contract'
    elif bytecode == reference_bytecode['aave_v2']:
        return 'lending_borrowing_contract'
    
    # lp
    elif bytecode == reference_bytecode['uni_pairv2']:
        return 'lp_amm'
    elif bytecode == reference_bytecode['balancer']:
        return 'lp_amm'  


def load_label_datasets():
    
    w3 = Web3(Web3.HTTPProvider(PUBLIC_RPC))

    # Load Kaggle dataset
    df_kaggle = pd.read_csv('./data/address_labels/address_labels_kaggle.csv')
    # Filter misplaced strings
    df_kaggle = df_kaggle[df_kaggle['Address'].str.contains('0x')]
    # Standardize to checksum address
    df_kaggle['checksum_address'] = df_kaggle['Address'].apply(lambda x: w3.to_checksum_address(x))

    # Load CEFI dataset
    cefi_labels = pd.read_csv('./data/address_labels/address_labels_cefi.csv', index_col=[0])
    # Standardize to checksum
    cefi_labels['checksum_address'] = cefi_labels['Address'].apply(lambda address_str: w3.to_checksum_address(address_str.replace('\u200b', '').replace('.', '').strip()))

    # Load DEX pairs dataset
    df_lp_pairs = pd.read_csv('./data/address_labels/dex_lp_pair_addresses.csv')
    # Standardize to checksum address
    df_lp_pairs['checksum_address'] = df_lp_pairs['hex(pair_address)'].apply(lambda x: w3.to_checksum_address('0x' + x))

    # Load Doxx dataset
    df_label = pd.read_csv('./data/address_labels/address_labels_targeted_dox.csv', index_col=[0])

    return df_kaggle, cefi_labels, df_lp_pairs, df_label

# Call the function to load the datasets
# df_kaggle, cefi_labels, df_lp_pairs, df_label = load_label_dataset()




