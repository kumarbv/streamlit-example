import streamlit as st
import pandas as pd
from thematicnifty import tn
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from fastquant import get_stock_data, backtest
from nseinfopackage import nseinfo
import plotly.graph_objects as go


# Initializer
bmi_group = (' - ','NIFTY_MIDSMALLCAP_400','NIFTY_200','NIFTY_LARGEMIDCAP_250','NIFTY_TOTAL_MARKET','NIFTY_50','NIFTY_SMALLCAP_50','NIFTY_SMALLCAP_250','NIFTY_SMALLCAP_100','NIFTY_100','NIFTY_MIDCAP_150','NIFTY_MIDCAP_SELECT','NIFTY_NEXT_50','NIFTY500_MULTICAP_50_25_25','NIFTY_MICROCAP_250','NIFTY_MIDCAP_50','NIFTY_MIDCAP_100')
secti_group = (' - ','NIFTY_PHARMA','NIFTY_BANK','NIFTY_PSU_BANK','NIFTY_FMCG','NIFTY_CONSUMER_DURABLES','NIFTY_PRIVATE_BANK','NIFTY_AUTO','NIFTY_HEALTHCARE_INDEX','NIFTY_ENERGY','NIFTY_METAL','NIFTY_OIL_and_GAS','NIFTY_REALTY','NIFTY_FINANCIAL_SERVICES_25_50','NIFTY_FINANCIAL_SERVICES','NIFTY_MEDIA','NIFTY_IT')
strati_group = (' - ','NIFTY100_LOW_VOLATILITY_30','NIFTY100_QUALITY_30','NIFTY50_VALUE_20','NIFTY_MIDCAP150_QUALITY_50','NIFTY_ALPHA_50','NIFTY200_QUALITY_30','NIFTY100_EQUAL_WEIGHT','NIFTY_ALPHA_LOW_VOLATILITY_30','NIFTY200_MOMENTUM_30','NIFTY_DIVIDEND_OPPORTUNITIES_50')
themei_group = (' - ','NIFTY100_ESG','NIFTY_INDIA_CONSUMPTION','NIFTY_SERVICES_SECTOR','NIFTY_INFRASTRUCTURE','NIFTY_INDIA_DIGITAL','NIFTY_PSE','NIFTY100_LIQUID_15','NIFTY_INDIA_MANUFACTURING','NIFTY_CPSE','NIFTY_MIDCAP_LIQUID_15','NIFTY_GROWTH_SECTORS_15','NIFTY_COMMODITIES','NIFTY_MNC')

start = "2017-07-08"
end = "2022-07-08"

st.title('Portfolio Optimizer')
st.write('Start Date: {st} | End Date: {ed}'.format(st=start, ed=end))

def getStocksList(main_choice, sub_choice):
    if main_choice == 'BMI':
        return tn.getThematicNiftyStocks(group_name='bmi_group', group_item=str(sub_choice), return_type='as_list')
    elif main_choice == 'STRATI':
        return tn.getThematicNiftyStocks(group_name='strati_group', group_item=str(sub_choice), return_type='as_list')
    elif main_choice == 'SECTI':
        return tn.getThematicNiftyStocks(group_name='secti_group', group_item=str(sub_choice), return_type='as_list')
    elif main_choice == 'THEMEI':
        return tn.getThematicNiftyStocks(group_name='themei_group', group_item=str(sub_choice), return_type='as_list')
    else:
        return 'No Choice'

def list_to_string(primary_list):
    filtered_list = []
    isin_list = nseinfo.getISINNumbers(primary_list)
    primary_list = nseinfo.getSymbols(isin_list)
    for share in primary_list:
        if pd.to_datetime(start) < pd.to_datetime(nseinfo.listedSince('SYMBOL',share)):
            filtered_list.append(share)
    final_list = list(set(primary_list) - set(filtered_list))
    final_list_1 = [item + '.NS' for item in final_list]
    return ' '.join(final_list_1)

@st.cache
def optimize_portfolio(shares_string, start, end):
    df = yf.download(shares_string, start=start, end=end)['Adj Close']
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)
    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ear, av, sr = ef.portfolio_performance(verbose=True)
    opt_data = [[round(ear*100, 2), round(av*100,2), round(sr, 4)]]
    df0 = pd.DataFrame(opt_data, columns = ['Expected Annual Returns (%)', 'Annual Volatality (%)', 'Sharpe Ratio'])
    data = []
    # Processing Cleaned Weights
    for key, value in cleaned_weights.items():
        if value >= 0.01:
            print(key, round(100 * value, 2))
            sub_data = [key, round(100 * value, 2)]
            data.append(sub_data)
    df1 = pd.DataFrame(data, columns=['Stock', 'Pct'])
    return df0, df, df1

@st.cache
def perform_backtest(df):
    # Backtesting
    backtest_data = []
    backtest_basic_data = []
    backtest_rvalue_data = []
    backtest_drawdowns_data = []
    backtest_wins_data = []
    backtest_losses_data = []

    for index, row in df.iterrows():
        stock = get_stock_data(row['Stock'], start, end)
        print(row['Stock'], '------->')
        res = backtest('smac', stock, fast_period=50, slow_period=200, plot=False)

        backtest_basic = [row['Stock'], round(res['pnl'][0], 2), res['sharperatio'][0]]
        backtest_rvalue = [row['Stock'], round(res['rtot'][0], 2), round(res['ravg'][0], 2), round(res['rnorm'][0], 2), round(res['rnorm100'][0], 2) ]
        backtest_drawdowns = [row['Stock'], round(res['drawdown'][0], 2), round(res['moneydown'][0], 2), round(res['maxdrawdown'][0], 2), res['maxdrawdownperiod'][0]]
        backtest_wins = [row['Stock'], res['won'][0], round(res['win_rate'][0], 2) ,round(res['won_avg'][0], 2), round(res['won_avg_prcnt'][0], 2), round(res['won_max'][0], 2), round(res['won_max_prcnt'][0], 2)]
        backtest_losses = [row['Stock'], res['lost'][0], round(res['lost_avg'][0], 2), round(res['lost_avg_prcnt'][0], 2), round(res['lost_max'][0], 2), round(res['lost_max_prcnt'][0], 2) ]

        backtest_basic_data.append(backtest_basic)
        backtest_rvalue_data.append(backtest_rvalue)
        backtest_drawdowns_data.append(backtest_drawdowns)
        backtest_wins_data.append(backtest_wins)
        backtest_losses_data.append(backtest_losses)

        backtest_subdata = [row['Stock'], res['pnl'][0], res['sharperatio'][0], res['lost'][0], res['lost_max'][0],
                            res['lost_max_prcnt'][0], res['won'][0], res['won_max'][0], res['won_max_prcnt'][0],
                            res['won_avg'][0], res['won_avg_prcnt'][0]]
        backtest_data.append(backtest_subdata)

    backtest_basic_df = pd.DataFrame(backtest_basic_data, columns = ['Stock', 'pnl', 'sharperatio'])
    backtest_rvalue_df = pd.DataFrame(backtest_rvalue_data, columns = ['Stock', 'rtot', 'ravg','rnorm', 'rnorm100'])
    backtest_drawdowns_df = pd.DataFrame(backtest_rvalue_data, columns = ['Stock', 'drawdown', 'moneydown', 'maxdrawdown', 'maxdrawdownperiod'])
    backtest_wins_df = pd.DataFrame(backtest_wins_data, columns = ['Stock', 'won', 'win_rate', 'won_avg', 'won_avg_prcnt', 'won_max', 'won_max_prcnt'])
    backtest_losses_df = pd.DataFrame(backtest_losses_data, columns = ['Stock', 'lost', 'lost_avg', 'lost_avg_prcnt', 'lost_max', 'lost_max_prcnt'])

    return backtest_basic_df, backtest_rvalue_df, backtest_drawdowns_df, backtest_wins_df, backtest_losses_df


def do_analysis(basic_df, rvalue_df, drawdowns_df, wins_df, losses_df):
    df0, df, df1 = optimize_portfolio(shares_string, start, end)
    st.markdown(''' ## Performance of Optimized Stocks''')
    st.table(df0)

    stocks = df1['Stock']
    share = df1['Pct']
    go_fig = go.Figure()
    obj = go.Pie(labels=stocks, values=share)
    go_fig.add_trace(obj)
    st.plotly_chart(go_fig)

    st.markdown(''' ## Backtesting Results''')
    st.text('Backtest Strategy: SMAC\nYears: last 5 years\nParameters chosen: fast_period=50, slow_period=200')
    backtest_val = st.selectbox("Choose Backtest Results Details:", [' - ','Basic', 'R Value', 'Drawdowns', 'Wins', 'Losses'])
    if backtest_val == 'Basic':
        st.markdown(''' ### Basic Data''')
        st.dataframe(basic_df)
    elif backtest_val == 'R Value':
        st.markdown(''' ### R Value Data''')
        st.dataframe(rvalue_df)
    elif backtest_val == 'Drawdowns':
        st.markdown(''' ### Drawdowns Data''')
        st.dataframe(drawdowns_df)
    elif backtest_val == 'Wins':
        st.markdown(''' ### Wins Data''')
        st.dataframe(wins_df)
    elif backtest_val == 'Losses':
        st.markdown(''' ### Losses Data''')
        st.dataframe(losses_df)
    return 'Done'


main_choice = st.sidebar.selectbox("Choose Main Theme", (' - ','BMI', 'STRATI', 'SECTI','THEMEI'))


if main_choice == 'BMI':
    sub_choice = st.sidebar.selectbox("Choose Sub Theme", bmi_group)
    stocks_list = getStocksList(main_choice, sub_choice)
    shares_string = list_to_string(stocks_list)
    df0, df, df1 = optimize_portfolio(shares_string, start, end)
    basic_df, rvalue_df, drawdowns_df, wins_df, losses_df = perform_backtest(df1)
    do_analysis(basic_df, rvalue_df, drawdowns_df, wins_df, losses_df)
elif main_choice == 'STRATI':
    sub_choice = st.sidebar.selectbox("Choose Sub Theme", strati_group)
    stocks_list = getStocksList(main_choice, sub_choice)
    shares_string = list_to_string(stocks_list)
    df0, df, df1 = optimize_portfolio(shares_string, start, end)
    basic_df, rvalue_df, drawdowns_df, wins_df, losses_df = perform_backtest(df1)
    do_analysis(basic_df, rvalue_df, drawdowns_df, wins_df, losses_df)
elif main_choice == 'SECTI':
    sub_choice = st.sidebar.selectbox("Choose Sub Theme", secti_group)
    stocks_list = getStocksList(main_choice, sub_choice)
    shares_string = list_to_string(stocks_list)
    df0, df, df1 = optimize_portfolio(shares_string, start, end)
    basic_df, rvalue_df, drawdowns_df, wins_df, losses_df = perform_backtest(df1)
    do_analysis(basic_df, rvalue_df, drawdowns_df, wins_df, losses_df)
elif main_choice == 'THEMEI':
    sub_choice = st.sidebar.selectbox('Choose Sub Theme', themei_group)
    stocks_list = getStocksList(main_choice, sub_choice)
    shares_string = list_to_string(stocks_list)
    df0, df, df1 = optimize_portfolio(shares_string, start, end)
    basic_df, rvalue_df, drawdowns_df, wins_df, losses_df = perform_backtest(df1)
    do_analysis(basic_df, rvalue_df, drawdowns_df, wins_df, losses_df)
