import pandas as pd
import statsmodels.api as sm

def rolling_mean(series, window): return series.rolling(window).mean()
def rolling_std(series, window): return series.rolling(window).std()
def rolling_sum(series, window): return series.rolling(window).sum()
def ewma(series, span, min_periods): return series.ewm(
    span=span, min_periods=min_periods).mean()


def get_value(df, idx, col): return df.iloc[idx][col]


def MA(df, n):
    MA = pd.Series(rolling_mean(df['Close'], n), name='MA_' + str(n))
    df['MA'] = MA
    return df


def EMA(df, n):
    EMA = pd.Series(
        ewma(df['Close'], span=n, min_periods=n - 1), name='EMA_' + str(n))
    df['EMA'] = EMA
    return df


def MOM(df, n):
    M = pd.Series(df['Close'].diff(n), name='Momentum_' + str(n))
    df['MOM'] = M
    return df


def STOK(df):
    SOk = pd.Series((df['Close'] - df['Low']) /
                    (df['High'] - df['Low']), name='SO%k')
    df['SOk'] = SOk
    return df


def STO(df,  nK, nD, nS=1):
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (
        df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name='SO%k'+str(nK))
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD,
                    min_periods=nD-1, adjust=True).mean(), name='SO%d'+str(nD))
    SOk = SOk.ewm(ignore_na=False, span=nS,
                  min_periods=nS-1, adjust=True).mean()
    SOd = SOd.ewm(ignore_na=False, span=nS,
                  min_periods=nS-1, adjust=True).mean()
    df['SOk'] = SOk
    df['SOd'] = SOd
    return df


def RSI(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= len(df) - 1:
        UpMove = get_value(df, i + 1, 'High') - get_value(df, i, 'High')
        DoMove = get_value(df, i, 'Low') - get_value(df, i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(ewma(UpI, span=n, min_periods=n - 1))
    NegDI = pd.Series(ewma(DoI, span=n, min_periods=n - 1))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df['RSI'] = RSI
    return df


def MACD(df, n_fast, n_slow):
    EMAfast = pd.Series(ewma(df['Close'], span=n_fast, min_periods=n_slow - 1))
    EMAslow = pd.Series(ewma(df['Close'], span=n_slow, min_periods=n_slow - 1))
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' +
                     str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(ewma(MACD, span=9, min_periods=8),
                         name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' +
                         str(n_fast) + '_' + str(n_slow))
    df['MACD'] = MACD
    df['MACDsign'] = MACDsign
    df['MACDdiff'] = MACDdiff
    return df


def CCI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - rolling_mean(PP, n)) /
                    rolling_std(PP, n), name='CCI_' + str(n))
    df['CCI'] = CCI
    return df


def train_and_forecast_SARIMA(data):
    model = sm.tsa.statespace.SARIMAX(data, trend='n', order=(1,1,0), seasonal_order=(1,0,0,16)).fit()
    return model.forecast(1)

def train_and_forecast_AR1(data, ar_order):

    model = ARIMA(data, order=(
        ar_order, 0, 0)).fit(method="yule_walker")

    forecasts = model.forecast(1)

    return forecasts
