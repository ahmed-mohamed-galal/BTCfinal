import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
import category_encoders
import joblib
import plotly.express as px 

###########################################

df = pd.read_csv('binance_btc_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df.drop('timestamp' , inplace=True , axis=1)
df.drop(['ignore' ,'close_time', 'Unnamed: 0'] , inplace=True , axis=1)
st.set_page_config(layout='wide')

###########################################

st.title("Welcome to My BTC Predection for the nest day")

with st.sidebar:
    st.title('BTC Dashboard')
select_ss = st.sidebar.selectbox('Select',  
                                    ('Select','Data Analysis' , 'Machine Learning'))

if select_ss=='Select':
    st.title('Please Select')

elif select_ss == 'Data Analysis':
    st.write("BTC Dashboard")

    #  the highest daily price range
    st.title('the highest daily price range')
    highest_day = df.groupby('day')['close'].sum().reset_index()
    q1 = px.line(highest_day , x ='day' , y ='close')
    st.plotly_chart(q1)

    # daily trades
    st.title('daily trades')
    trade_day = df.groupby('day')['trades'].sum().reset_index()
    q2 = px.line(trade_day , x ='day' , y ='trades')
    st.plotly_chart(q2)

    # the higest year of price
    st.title('the higest year of price')
    highest_day = df.groupby('year')['close'].max().reset_index()
    q3 = px.bar(highest_day , x ='year' , y ='close' , text_auto=True)
    st.plotly_chart(q3)

    # the lowet year of price
    st.title('the lowet year of price')
    lowest_year = df.groupby('year')['close'].min().reset_index()
    q4 = px.bar(lowest_year , x ='year' , y ='close' , text_auto=True)
    st.plotly_chart(q4)

    # The highest price for top 5 months
    st.title('The highest price for top 5 months')
    q5_filter = df.groupby('month')['close'].max().reset_index()
    q5_filter = q5_filter.sort_values(by='close', ascending=False).head(5)
    q5_filter['highlight'] = q5_filter['month'] == q5_filter.iloc[0]['month']
    q5 = px.pie(q5_filter,
                 names='month',
                 values='close',
                 title='The highest price for top 5 months',
                 color='highlight',
                 color_discrete_map={True: 'gold', False: 'lightgray'},
                 hole=0.2)
    q5.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(q5)



elif select_ss == 'Machine Learning':
    st.write("Welcome to the Machine Learning section")
    def get_input():
        open = st.number_input("Enter a number of Start price")
    
        low = st.number_input("Enter a number of Lowest price" )
    
        close = st.number_input("Enter a number of End price")
    
        volume = st.number_input("Enter a Total BTC Traded (Volume)")
    
        quote_asset_volume = st.number_input("Enter a 	Total USDT Traded")
    
        trades = st.number_input("Enter a number Number of Trades")
    
        taker_buy_base = st.number_input("Enter a number of BTC Bought via Market Buys (Taker Buy Volume)")
    
        taker_buy_quote = st.number_input("Enter a number of USDT Spent via Market Buys (Taker Buy Value)")
    
        year = st.slider("Enter Year", min_value=2025, max_value=2030)
    
        month = st.slider("Enter month", min_value=0, max_value=12)
    
        day = st.slider("Enter day", min_value=0, max_value=31)
    
        return pd.DataFrame(data=[[open , low , close ,volume , quote_asset_volume , trades ,taker_buy_base,taker_buy_quote, year , month ,day]],
            columns=[ 'open' , 'low' , 'close' ,'volume' , 'quote_asset_volume' , 'trades','taker_buy_base','taker_buy_quote', 'year' , 'month' ,'day'])
    
    predct = get_input()
    
    if st.button("🔮 Predict Next Day's BTC Close Price"):       
        pl = joblib.load('btc.h5')
        st.write(pl.predict(predct))
