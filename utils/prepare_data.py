import pandas as pd

def prepare_short_data(filename:str='data/returns_RU.xlsx') -> pd.DataFrame:
    """
    reads data from excel file, fills weekday indices, percent changes for stocks
    returns filled dataframe
    """
    df = pd.read_excel(filename, parse_dates=[0])
    df['Weekday'] = df['Date'].dt.dayofweek
    for col in df.columns[1:-1]:
        df[col+'_pct'] = df[col].pct_change()
    
    return df