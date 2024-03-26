### put into script ###

# import
import dask.dataframe as dd

import pandas as pd
import numpy as np

import os
from os.path import join

from dotenv import load_dotenv

load_dotenv()

path = os.environ["PROJECT_PATH"]


# load all token transfers
ddf = dd.read_csv(join(path, "token_transfers_22021209.csv"))

# set value as int
ddf.value = ddf.value.astype("float64")

# load all snapshots blockheights
df_snapshots = pd.read_csv("./assets/snapshot_selection.csv")

# iterate over snapshots
for snapshot_block_height in df_snapshots["Block Height"]:

    print(f"Snapshot Block Height: {snapshot_block_height}")

    # filter df to snapshot height
    ddf_snapshot = ddf[ddf.block_number <= snapshot_block_height]

    # compute sums
    ddf_outflows = (
        ddf_snapshot.groupby(["token_address", "from_address"])
        .value.sum()
        .compute()
        .reset_index()
    )
    ddf_inflows = (
        ddf_snapshot.groupby(
            [
                "token_address",
                "to_address",
            ]
        )
        .value.sum()
        .compute()
        .reset_index()
    )

    # format
    ddf_outflows.value = ddf_outflows.value * -1
    ddf_outflows.columns = ["token_address", "address", "value"]
    ddf_inflows.columns = ["token_address", "address", "value"]

    # stack df's
    df_all = pd.concat([ddf_inflows, ddf_outflows])

    # sum
    df_token_holder_balance_snapshot = df_all.groupby(
        ["address", "token_address"]
    ).value.sum()

    # store tokenholder df as csv file
    df_token_holder_balance_snapshot.to_csv(
        join(
            path,
            f"token_balance_lookup_tables/token_holder_snapshot_balance_{snapshot_block_height}.csv",
        )
    )
