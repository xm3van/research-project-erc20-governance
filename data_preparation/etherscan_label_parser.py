import pandas as pd 
import requests

df_tb_sc = pd.read_csv('df_tb_sc.csv', index_col=0)

for index, row in df_tb_sc.iterrows(): 
    
    try: 
        r = requests.get(f'https://octal.art/etherscan-labels/addresses/{row.address_x}.json')
        row['etherscan_label'] = list(r.json()['Labels'].keys())
        
    except: 
        
        pass
    
    
df_tb_sc.to_csv('df_tb_sc_esl.csv')

