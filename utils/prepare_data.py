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
        df.rename(columns={col:col+'_close'}, inplace=True)
    
    return df

def prepare_full_data(filename:str='data/Data_RU.xlsx') -> pd.DataFrame:
    """
    reads data from excel file, fills weekday indices, percent changes for stocks
    returns filled dataframe
    """
    df = pd.read_excel(filename, skiprows=1, parse_dates=[0])
    df = df.loc[:, (df.isnull().sum(axis=0) <= 0.1*df.shape[0])]
    
    df.drop([0], inplace=True)
    df.dropna(axis='index', inplace=True)
    df['Date'] = pd.to_datetime(df['SECID'], format="%Y-%m-%d %H:%M:%S")
    df.drop(columns=['SECID'], inplace=True)
    df['Weekday'] = df['Date'].dt.dayofweek
    
    for col in df.columns[:-2]: # except Date
        if col[-2] == '.':
            if col[-1] == '1':
                df.rename(columns={col:col[:-2]+'_high'}, inplace=True)
            elif col[-1] == '2':
                df.rename(columns={col:col[:-2]+'_low'}, inplace=True)
            elif col[-1] == '3':
                df.rename(columns={col:col[:-2]+'_open'}, inplace=True)
            else:
                df.rename(columns={col:col[:-2]+'_volume'}, inplace=True)
        else:
            df.rename(columns={col:col+'_close'}, inplace=True)
#     print(df.columns)
    for col in df.columns[:(len(df.columns)-2)//5]:
        col_name = col[:-6]
        df[col_name+'_D1'] = (df[col_name+'_open']-df[col_name+'_close'].shift(periods=-1))/df[col_name+'_close'].shift(periods=-1)
        df[col_name+'_D2'] = (df[col_name+'_close']-df[col_name+'_open'])/df[col_name+'_open']
        df[col_name+'_D3'] = (df[col_name+'_high']-df[col_name+'_open'])/df[col_name+'_open']
        df[col_name+'_D4'] = (df[col_name+'_open']-df[col_name+'_low'])/df[col_name+'_low']
    
    return df.dropna(axis='index')
        