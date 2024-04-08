import requests

def get_ticker(coingecko_id): 
    print(coingecko_id)
    data = requests.get(f'https://api.coingecko.com/api/v3/coins/{coingecko_id}')
    ticker = data.json()['symbol']
    time.sleep(int(60/4))
    return ticker.upper()

def convert_blockheight_to_date(snapshot,api_key): 
    
    """Takes block_height and returns date.

    Args:
        snapshot (str): block height

    Returns:
        str: date of format YYYY-MM-DD

    Raises:
        None
    """
    
    url = f"https://api.covalenthq.com/v1/eth-mainnet/block_v2/{snapshot}/?key={api_key}"

    response = requests.get(url)
    
    date = response.json()['data']['items'][0]['signed_at']
    
    date_YMD = date.split('T')[0]
    
    return date_YMD


def usd_value_decimals_token(snapshot, contractAddress, api_key, chainName='eth-mainnet',quoteCurrency='USD'): 
    
    """Takes blockheight and contract address of token and
    returns contract_decimals and value in usd.

    Args:
        snapshot (str): block height.
        contractAddress (str): token contract address

    Returns:
        dict with keys and value pairs: 
        'contract_address': contractAddress (str)
        'token_price_usd': price_usd_per_token (int)
        'contract_decimals': decimals (int)

    Raises:
        None
    """
    

    
    
    date = convert_blockheight_to_date(snapshot, api_key)
    
    
    url = f"https://api.covalenthq.com/v1/pricing/historical_by_addresses_v2/{chainName}/{quoteCurrency}/{contractAddress}/?key={api_key}"
    
    params = {
    "from": date, # (YYYY-MM-DD)
    "to": date # (YYYY-MM-DD)
    }

    response = requests.get(url, params=params)
    
    # token price 
    price_usd_per_token = response.json()['data'][0]['prices'][0]['price']
    
    # decimals
    decimals = response.json()['data'][0]['contract_decimals']
    
    return {'contract_address': contractAddress, 'token_price_usd': price_usd_per_token, 'contract_decimals': decimals}
    
    
    

