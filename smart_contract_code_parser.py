"""
The purpose of this script is to collect smart contract code which is neede to classify
a) if we are dealing with an EOA addresss
b) if we are dealing with a multi-sig 
c) potentially other identifiable contract
"""

import dotenv
import pandas as pd
from os.path import join
import csv
from web3 import Web3, HTTPProvider

# store environment variables
env_var = dotenv.dotenv_values()

print("### set-up ###")
# Set up authentication
path = env_var["PROJECT_PATH"]
w3 = Web3(Web3.HTTPProvider(env_var["ETHEREUM_NODE_ENDPOINT"]))

# set input & output path
input_path = join(path, "df_unique_addresses2.csv")
output_path = join(path, "smart_contract_code.csv")

# load input file
df_ua = pd.read_csv(input_path)

# Script
print("### Start work ###")

# Open the CSV file for writing
with open(output_path, "w", newline="") as csvfile:
    fieldnames = ["address", "code"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for a in df_ua.unique_addresses:

        address = w3.toChecksumAddress(a)

        code = w3.eth.getCode(address)

        writer.writerow({"address": address, "code": code})

print(f"Data written to {output_path}")
print("### End work ###")
