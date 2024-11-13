
### Step 1: Import Required Libraries
import datetime
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os
import matplotlib.dates as mdates
from copy import deepcopy

def convert_to_bps_ticks(df,bps_per_instrument=8/0.99,ticks_per_instrument=32/0.99):
    '''
    0.99 (ZN price pts) change = 32 ticks or 8bps; This is what I understand for ZN.
    So 1 bp is 0,12375 ZN price points.
    So 1ZN price pts change= 32/0.99 ticks per zn
    So 1ZN price pts change= 8/0.99  bps per zn
    '''
    b=bps_per_instrument
    t=ticks_per_instrument
    df_b=df*bps_per_instrument
    df_t=df*ticks_per_instrument
    return pd.concat((df_b,df_t),axis=1)
    
### Step 2: Fetching and Preprocessing Data
def get_market_data(symbols, start_date, end_date, interval="1d", period='Not Defined',return_change=False): #Adj Close data for a lot many days
    closing_prices = yf.download(
        symbols, start=start_date, end=end_date, interval=interval
    )["Adj Close"] #or close same thing.
    #---Above code gives pandas dataframe in general. ["Adj Close"] gives adjusted close price after dividend.
    
    closing_prices.columns = (
        closing_prices.columns.str.replace("=F", "")
        .str.replace("DX-Y.NYB", "DXY")
        .str.replace("^", "")
    )
    if return_change:
        return closing_prices.pct_change().dropna()
    return closing_prices.dropna()


def get_intraday_data(symbol, interval="5m", period="60d", start_date=None):  # Same as get_market_data
    if start_date:
        end_date = start_date - datetime.timedelta(days=int(period.replace("d", "")))
        df = yf.download(symbol, start=end_date, end=start_date, interval=interval)
    else:
        df = yf.Ticker(symbol).history(period=period, interval=interval)

    # Modify column names to remove '=F' or other extensions
    simplified_name = (
        symbol.replace("=F", "").replace("DX-Y.NYB", "DXY").str.replace("^", "")
    )
    df.rename(columns={"Close": simplified_name}, inplace=True)
    return df[[simplified_name]]


def split_periods(data):
    periods = {
        "us_open": data.between_time("07:00:00", "11:00:00").sort_index(),
        "us_mid": data.between_time("11:00:01", "15:00:00").sort_index(),
        "us_close": data.between_time("15:00:01", "17:00:00").sort_index(),
        "us_apac": data.between_time("17:00:01", "02:00:00").sort_index(),
        "us_emea": data.between_time("02:00:01", "07:00:00").sort_index(),
    }
    return periods


def get_company_name(ticker):
    try:
        company = yf.Ticker(ticker)
        return company.info["longName"]
    except:
        try:
            company = yf.Ticker(ticker)
            return company.info["shortName"]
        except:
            return ticker  # Return the original ticker if unable to fetch the name
#Tickers and corresponding symbols
dict_tickers = {
    "ZN=F": "10-Year T-Note Futures",
    "DX-Y.NYB": "US Dollar Index",
    "CL=F": "Crude Oil futures",
    "GC=F": "Gold futures",
    "NQ=F": "Nasdaq 100 futures",
    "^DJI": "Dow Jones Industrial Average",
    "^GSPC": "S&P 500"}
    # "FGBL=F":"German 10-Year Bund", #Not sure
	# "FOAT=F":"French 10-Year OAT", #Not sure
	# "G=F":"UK 10-Year Gilt"} #Not sure

dict_symbols = {
     "ZN=F":["ZN","10-Year T-Note Futures"],
    "DX-Y.NYB":["DXY","US Dollar Index"],
     "CL=F":["CL","Crude Oil futures"],
     "GC=F":["GC","Gold futures"],
     "NQ=F":["NQ","Nasdaq 100 futures"],
    "^DJI":["DJI","Dow Jones Industrial Average"],
     "^GSPC":["GSPC","S&P 500"]}
    # "FGBL":"German 10-Year Bund",
    # "FOAT":"French 10-year OAT",
    # "G": "UK 10-Year Gilt"
print(dict_symbols)
#{'ZN=F': ['ZN', '10-Year T-Note Futures'], 'DX-Y.NYB': ['DXY', 'US Dollar Index'], 'CL=F': ['CL', 'Crude Oil futures'], 'GC=F': ['GC', 'Gold futures'], 'NQ=F': ['NQ', 'Nasdaq 100 futures'], '^DJI': ['DJI', 'Dow Jones Industrial Average'], '^GSPC': ['GSPC', 'S&P 500']}
 
#Getting intervals for last 1 month intra-day data i.e interval=1m
# Get today's date
today = (datetime.datetime.now())
latest_intraday=(today-timedelta(days=1)).strftime("%Y-%m-%d")#doing this to get complete data of previous day
# Calculate the date one month ago
one_month_ago = today - timedelta(days=29)
last_intraday = one_month_ago.strftime("%Y-%m-%d")
start_intraday=latest_intraday
end_intraday=last_intraday
print(f'Fetching Intraday data (1m interval) from {start_intraday} to {last_intraday}')
# Make List of all possible Start-End dates between today and today-30
alldates=[]
for i in range(29,0,-7):
    alldates.append((today-timedelta(days=i)).strftime("%Y-%m-%d"))
grouped_start_last_dates=[[a,b] for a,b in zip(alldates,alldates[1:])]
dates=grouped_start_last_dates
print(dates)

###Step5: Intraday returns (High - Low)
def combine_all_data(list_of_first_last,list_of_symbols): # Append the data for all the Start-End dates
    datalist=[]
    for data in list_of_first_last:
        start=last_intraday=data[0]
        end=latest_intraday=data[1]
        interval='1m'
        data = yf.download(list_of_symbols, start=start, end=end, interval=interval) #Returns a dataframe
        datalist.append(data)
    return(pd.concat(datalist))

print("Today's date:", start_intraday)
print("Date one month ago:", end_intraday)

# Save distribution of daily returns
# Create output directory if it doesn't exist
output_dir = "output_monthly_intraday"
os.makedirs(output_dir, exist_ok=True)
name = f"{end_intraday}_to_{start_intraday}"


alltickers=list(dict_symbols.keys())
allsymbols=[i[0] for i in list(dict_symbols.values())]

print('Symbols',allsymbols)
print('Tickers',alltickers)
# Fetch Data
data = combine_all_data(grouped_start_last_dates,list_of_symbols=alltickers)
tickers_as_columns=data.stack(level=0)
for col in tickers_as_columns.columns:
    col_data=tickers_as_columns[col].unstack()
    dict_symbols[col].append(col_data)

#Get 1month Intraday data for ZN in bps and ticks
zn_intraday_data_pts=deepcopy(dict_symbols['ZN=F'][2])
zn_returns=zn_intraday_data_pts['Close'] # or Adj Close. Both are same in options and futures.
zn_returns_change=abs(zn_returns.diff(-1))#next-initial row absolute difference
zn_returns_change=zn_returns_change.dropna() #removing the last null row.
zn_returns_change_bps_ticks=convert_to_bps_ticks(zn_returns_change,
                                           bps_per_instrument=8/0.99,
                                           ticks_per_instrument=32/0.99)

df_zn_returns_change_bps_ticks=pd.DataFrame(zn_returns_change_bps_ticks)
df_zn_returns_change_bps_ticks.columns=pd.Index(['bps','ticks'],name='ZN Returns')
print(df_zn_returns_change_bps_ticks)

# Save distribution of ZN returns
# Create output directory if it doesn't exist
zn_output_dir = "output_monthly_intraday_bps_ticks_ZN"
os.makedirs(zn_output_dir, exist_ok=True)
zn_name = f"{end_intraday}_to_{start_intraday}"
zn_output_file_0= os.path.join(
            zn_output_dir, "ZN_Intraday_returns_data_"+f"{zn_name}.csv"
        )
df_zn_returns_change_bps_ticks.to_csv(zn_output_file_0)

 
#Data Stats for all tickers
#1. All tickers intraday data in pts
print(f"\nLast one month - Intraday data Summary {start_intraday} to {end_intraday} for all tickers:")
output_file_all = os.path.join(
            output_dir, "All_tickers_Intraday_monthly_data_summary_"+f"{name}.csv"
        )
data.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_csv(output_file_all)
data.describe()

#2. ZN returns data in bps and ticks
print(f"\nLast one month - Intraday returns data Summary {start_intraday} to {end_intraday} for ZN (bps and ticks):")
zn_output_file= os.path.join(
            zn_output_dir, "ZN_Intraday_returns_data_summary_"+f"{zn_name}.csv"
        )
df_zn_returns_change_bps_ticks.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_csv(zn_output_file)
df_zn_returns_change_bps_ticks.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

#Plotting intra day Returns data for all tickers with X-Axis having all dates from start to end
for key_of_col in list(dict_symbols.keys()):
    sy=(dict_symbols[key_of_col][0])
    print(sy)

# 1 Returns for all tickers in pts
for key_of_col in list(dict_symbols.keys()):
    sy=dict_symbols[key_of_col][0]#symbol to be displayed as filename to prevent errors later
    try:
        data = dict_symbols[key_of_col][2] #Data for that symbol.
        if data.empty:
            raise ValueError("No data available for the specified date range.")
        # Print data
        # Print some statistics
        print(f"\nLast one month - Intraday returns data Summary {start_intraday} to {end_intraday} for ZN (bps and ticks):")
      
        output_file = os.path.join(
            output_dir, "Intraday_monthly_summary_" + str(sy) + '_'+f"{name}.csv"
        )
        output_file2 = os.path.join(
            output_dir, "Intraday_monthly_data_" + str(sy) + '_'+f"{name}.csv"
        )
        data.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_csv(output_file)
        data.to_csv(output_file2)
        # Plot the data
        plt.figure(figsize=(12, 6))
    
        # Plot Adj Close as a line
        print(data)
        data["Close"] = data["Close"].interpolate()
        data["High"] = data["High"].interpolate()
        data["Low"] = data["Low"].interpolate()
        plt.plot(data.index, data["Close"], label="Close", color="green")
       


        # Set the date format for x-axis
        ax = plt.gca()  # Get current axis
        ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks for each day
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format ticks to show date only
       
        plt.xticks(rotation=90)
        # Shade the area between High and Low
       
        plt.fill_between(data.index, data["Low"],data["High"],alpha=0.1,color="red",label="High-Low Range")
    
        plt.title(
            f"{str(sy)} Price Data ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})"
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_file3 = os.path.join(
            output_dir, "Monthly_intraday_returns_All_Dates_" + str(sy) +'_'+ f"{name}.jpg"
        )
        plt.savefig(output_file3)
        plt.show()
    
        # Print the actual date range of the data
        print(f"\nActual date range of data:")
        print(f"Start: {data.index[0]}")
        print(f"End: {data.index[-1]}")
        print(f"Total periods: {len(data)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Try adjusting the date range or check the availability of data for this symbol."
        )

try:
    zndata = df_zn_returns_change_bps_ticks.copy()
    if zndata.empty:
        raise ValueError("No data available for the specified date range.")
    
    zn_output_file2 = os.path.join(
        zn_output_dir, "ZN_Intraday_returns_bps_ticks_All_Dates_" + f"{zn_name}.jpg"
    )

    # Set up the figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot ZN Returns in bps on the first subplot
    x = zndata.index
    y1 = zndata['bps']
    ax1.plot(x, y1, 'b-', label='ZN Returns in bps')
    ax1.set_ylabel('ZN Returns in bps', color='b')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Plot ZN Returns in ticks on the second subplot
    y2 = zndata['ticks']
    ax2.plot(x, y2, 'g-', label='ZN Returns in ticks')
    ax2.set_ylabel('ZN Returns in ticks', color='g')
    ax2.set_xlabel('Dates')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Set date format for x-axis (shared)
    ax2.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks for each day
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format ticks to show date only
    plt.xticks(rotation=90)
    
    # Add an overall title
    fig.suptitle(
        f"ZN Intraday Returns Data ({zndata.index[0].strftime('%Y-%m-%d')} to {zndata.index[-1].strftime('%Y-%m-%d')})"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title

    # Save the plot
    plt.savefig(zn_output_file2)
    
    # Display the plot
    plt.show()

    # Print the actual date range of the data
    print(f"\nActual date range of data:")
    print(f"Start: {zndata.index[0]}")
    print(f"End: {zndata.index[-1]}")
    print(f"Total periods: {len(zndata)}")

except Exception as e:
    print(f"An error occurred: {e}")
    print(
        "Try adjusting the date range or check the availability of data for this symbol."
    )
