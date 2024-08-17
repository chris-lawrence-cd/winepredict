import pandas as pd
import requests
from datetime import datetime
import openpyxl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def download_fred_data(series_id, api_key):
    """Downloads data from FRED for a given series ID.

    Args:
        series_id (str): The FRED series ID.
        api_key (str): The API key for accessing FRED data.

    Returns:
        pd.DataFrame: A DataFrame containing the downloaded data with dates as the index.
    """
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    response = requests.get(url)
    data = response.json()  # Corrected this line
    df = pd.DataFrame(data['observations'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Clean data: replace '.' with NaN and convert to float
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    return df[['value']]

def process_fred_data(api_key, start_date='1992-01-01', output_file='FRED_Data.xlsx'):
    """Processes FRED data by downloading, filtering, resampling, and saving it to an Excel file.

    Args:
        api_key (str): The API key for accessing FRED data.
        start_date (str): The start date for filtering the data. Defaults to '1992-01-01'.
        output_file (str): The name of the output Excel file. Defaults to 'FRED_Data.xlsx'.
    """
    # Define series IDs for FRED data
    fred_series_ids = {
        'Average Wine Price': 'APU0000720311',
        'GDP': 'GDP',
        'Unemployment Rate': 'UNRATE',
        'Retail Sales': 'RSAFS',
        'Durable Goods Orders': 'DGORDER',
        'Money Supply (M1)': 'WM1NS',
        'Federal Funds Rate': 'FEDFUNDS',
        'Consumer Price Index (CPI)': 'CPIAUCSL',
        'S&P 500 Index': 'SP500',
        'Personal Consumption Expenditures': 'PCE',
        'Disposable Personal Income': 'DPI',
        'Consumer Confidence Index': 'UMCSENT',
        'Producer Price Index': 'PPIACO',
        'Industrial Production Index': 'INDPRO',
        'Total Nonfarm Payrolls': 'PAYEMS',
        'Housing Starts': 'HOUST',
        '10-Year Treasury Constant Maturity Rate': 'GS10',
        'Corporate Profits After Tax': 'CP',
        'Personal Savings Rate': 'PSAVERT'
    }

    # Download FRED data
    fred_data = {}
    for name, series_id in fred_series_ids.items():
        fred_data[name] = download_fred_data(series_id, api_key)

    # Combine all data into a single DataFrame
    combined_df = pd.concat(fred_data.values(), axis=1)
    combined_df.columns = fred_data.keys()

    # Filter data to include only rows on or after the start date
    filtered_df = combined_df[combined_df.index >= start_date]

    # Resample data to monthly frequency
    monthly_df = filtered_df.resample('M').mean()

    # Create data dictionary
    data_dict = pd.DataFrame({
        'Variable Name': list(fred_series_ids.keys()),
        'FRED Series ID': list(fred_series_ids.values()),
        'Description': [
            'Average Wine Price',
            'Gross Domestic Product',
            'Unemployment Rate',
            'Retail Sales',
            'Durable Goods Orders',
            'Money Supply (M1)',
            'Federal Funds Rate',
            'Consumer Price Index',
            'S&P 500 Index',
            'Personal Consumption Expenditures',
            'Disposable Personal Income',
            'Consumer Confidence Index',
            'Producer Price Index',
            'Industrial Production Index',
            'Total Nonfarm Payrolls',
            'Housing Starts',
            '10-Year Treasury Constant Maturity Rate',
            'Corporate Profits After Tax',
            'Personal Savings Rate'
        ]
    })

    # Save data to an Excel file with two tabs
    with pd.ExcelWriter(output_file) as writer:
        monthly_df.to_excel(writer, sheet_name='Data')
        data_dict.to_excel(writer, sheet_name='Data Dictionary', index=False)

    print("Data downloaded, filtered, resampled to monthly frequency, and saved to {}.".format(output_file))

def preprocess_and_analyze_data(file_path, sheet_name='Data'):
    """Loads, preprocesses, and performs correlation analysis on the data.

    Args:
        file_path (str): The path to the Excel file containing the data.
        sheet_name (str): The name of the sheet in the Excel file to load. Defaults to 'Data'.

    Returns:
        pd.DataFrame: A DataFrame containing the scaled data with the 'Average Wine Price' column.
    """
    # Load data
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Data preprocessing
    # Handle missing values by filling with the mean of each column
    data.fillna(data.mean(), inplace=True)

    # Drop the datetime column if it exists
    if 'date' in data.columns:
        data.drop(columns=['date'], inplace=True)

    # Feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop(['Average Wine Price'], axis=1))
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns[1:])
    scaled_df['Average Wine Price'] = data['Average Wine Price'].values

    # Correlation analysis
    plt.figure(figsize=(12, 8))
    sns.heatmap(scaled_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    return scaled_df
