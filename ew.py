import pandas as pd
import numpy as np

# --- Utility & Analytics Functions ---
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

# --- Strategy Definitions (Verified & Robust) ---

def strat_bwx(df):
    df = df.copy()

    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = close.pct_change()
    month, day, dow = df.index.month, df.index.day, df.index.dayofweek
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()
    def std(s, n): return s.rolling(n).std()

    # Core Technicals
    sma10, sma20, sma50, sma200 = sma(close, 10), sma(close, 20), sma(close, 50), sma(close, 200)
    std20, rsi14, rsi2 = std(close, 20), get_rsi(close, 14), get_rsi(close, 2)
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    
    # MACD Block
    m12 = ema(close, 12) - ema(close, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    macd_h = m12 - s9
    
    m3_7 = ema(close, 3) - ema(close, 7)
    s3_7 = m3_7.ewm(span=2, adjust=False).mean()

    # CCI and MAD
    tp = (high + low + close) / 3
    sma_tp = sma(tp, 20)
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # Volatility / Volume Technicals
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    ha20, ha5 = rng.rolling(20).mean(), rng.rolling(5).mean()
    vol_avg20 = vol.rolling(20).mean()
    vol_std20 = rets.rolling(20).std()
    v_norm = vol / sma(vol, 50)
    range_climax = rng / rng.rolling(50).mean().replace(0, np.nan)
    
    # MFI
    pmf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    nmf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (pmf / (nmf + 1e-6))))
    
    # UO
    uo_bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4 * (uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2 * (uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    l24_52 = ema(close, 24) - ema(close, 52)
    h3 = l24_52 - ema(l24_52, 18)
    
    rv5, rv20 = std(rets, 5) * 15.87, std(rets, 20) * 15.87
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (vol_std20 + 1e-10)
    dom, dow, month = df.index.day, df.index.dayofweek, df.index.month
    vol20 = rets.rolling(20).std()

    def get_macd_local(s, fast=12, slow=26, signal=9):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    vm_p, vm_m = (high - low.shift(1)).abs(), (low - high.shift(1)).abs()
    vi_p, vi_m = vm_p.rolling(14).sum() / tr.rolling(14).sum(), vm_m.rolling(14).sum() / tr.rolling(14).sum()

    m_s, dy_s, dw_s = df.index.month, df.index.day, df.index.dayofweek
    dy_s, m_s, dw_s = pd.Series(dy_s, index=df.index), pd.Series(m_s, index=df.index), pd.Series(dw_s, index=df.index)

    gap = (open_p / close.shift(1)) - 1
    std10, std50 = close.rolling(10).std(), close.rolling(50).std()
    ma20, ema100 = close.rolling(20).mean(), close.ewm(span=100).mean()
    m5, s5, _ = get_macd_local(close, 5, 13, 1)

    ha20, ha5 = (high - low).rolling(20).mean(), (high - low).rolling(5).mean()
    m12, s12, _ = get_macd_local(close, 12, 26, 9)
    h12 = m12 - s12
    vol_proxy = (high - low).rolling(14).std()

    m, dy, dw = df.index.month, df.index.day, df.index.dayofweek
    ret60 = close.pct_change(60)
    returns = close.pct_change()
    semi_down = returns.clip(upper=0).rolling(20).std()
    v20, v60 = rets.rolling(20).std(), rets.rolling(60).std()
    ploc = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
    vsma20 = vol.rolling(20).mean()

    tp = (high + low + close) / 3
    mf = tp * vol
    # Use the df.index to ensure alignment in the MFR calculation
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    mfr = pos_mf / (neg_mf + 1e-10)
    d_mid = (high.rolling(20).max() + low.rolling(20).min()) / 2
    atr20 = (high - low).rolling(20).mean()
    vwap20 = (vol * close).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)
    emv = (((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)) / (vol / 1000000 / (high - low + 1e-10) + 1e-10)
    emv_z = (emv - emv.rolling(20).mean()) / (emv.rolling(20).std() + 1e-10)
    mv, sv, hv = get_macd_local(vol, 5, 15, 5)

    df['Vol_SMA20'] = vol.rolling(20).mean()
    df['ibs'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    pk = 100 * (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min())
    rsi = get_rsi(close, 14)

    s = pd.DataFrame(index=df.index)

    s['s4'] = np.where(ibs < 0.2, 1.5, np.where(ibs > 0.8, -1.0, 0.5))
    s['s5'] = np.where(close > close.shift(1) + (np.maximum(high-low, np.maximum(abs(high-close.shift(1)), abs(low-close.shift(1))))).rolling(14).mean(), 1.0, -0.5)
    s['s6'] = np.where(pk > pk.rolling(3).mean(), 1.0, -0.5)
    s['s7'] = np.where((df.index.day >= 10) & (df.index.day <= 15), 1.0, 0.1)
    s['s10'] = np.where(close > high.rolling(20).max().shift(1), 1.0, -0.5)
    s['c6'] = np.where(close > close.shift(20), 1.0, -1.0)
    s['i2_c6'] = np.where(vol < vol.shift(1), 0.5, -0.5)
    s['i3_c3'] = np.where(ibs.rolling(5).mean() < 0.3, 1.2, 0.0)
    s['i5_c12'] = np.where(rsi < 20, 1.5, -0.5)
    s['L1_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L2_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L3_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L4_C12'] = np.where(vol.pct_change() > 0.5, -1.0, 0.5)
    s['IBS_Donch_High_Fail'] = np.where((df['ibs'] > 0.8) & (df['High'] < df['High'].rolling(20).max()), -0.5, 0.0)
    s['VWAP_Dist'] = np.where(close > vwap20, 0.2, -0.4)
    s['EMV_Z'] = np.where(emv_z > 1.0, 0.3, -0.3)
    s['Vortex_Trend'] = np.where(vi_p > vi_m, 0.4, -0.2)
    s['Std_Bias'] = np.where(ibs.rolling(10).mean() / (ibs.rolling(10).std() + 1e-10) > 1.5, -0.4, 0.1)
    s['VolHedge'] = np.where(v20 * np.sqrt(252) > v60 * np.sqrt(252) * 1.3, -1.0, 0.0)
    s['VolATR'] = np.where((vol / (vsma20 + 1e-10) > 1.5) & (rets < 0), 1.0, 0.0)
    s['IBS_Extreme_S'] = np.where(ibs > 0.98, -1.0, 0.0)
    s['Vol_Vol_Conviction'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['DonchMid'] = np.where((close < d_mid) & (close > d_mid * 0.99), 0.5, 0.0)
    s['AugExit'] = np.where(month == 8, -1.0, 0.1)
    s['Tail_Risk'] = returns.rolling(60).quantile(0.05) / (semi_down + 1e-10) 
    s['Asym_Conv'] = semi_down.rolling(20).corr(returns.rolling(20).std())
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['Payday'] = np.where(dy_s.isin([1, 15, 30]), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((m_s == 11) & (dy_s >= 20), 0.3, -0.1)
    s['Weekend_Gap'] = np.where((dw_s == 0) & (gap < -0.005), 0.4, -0.1)
    s['Stdev_Ratio'] = np.where(std10 / (std50 + 1e-10) > 1.2, 0.3, -0.1)
    s['VIX_Proxy'] = np.where((close / (close.rolling(20).max() + 1e-10) - 1) < -0.05, 0.5, -0.1)
    s['ZScore_Price'] = np.where(((close - close.rolling(20).mean()) / (close.rolling(20).std()+1e-10)) > 1.0, 0.4, np.where(((close - close.rolling(20).mean()) / (close.rolling(20).std()+1e-10)) < -1.0, -0.4, 0))
    s['Intra_Vol_Conc'] = np.where(vol / (high-low+1e-10) > (vol/(high-low+1e-10)).rolling(20).mean(), -0.2, 0.1)
    s['MACD_Dual_Confirm'] = np.where((m5 > s5) & (m12 > s12), 0.5, -0.2)
    s['Up_Down_Ratio'] = np.where(vol.where(rets > 0).rolling(10).sum() / vol.where(rets < 0).rolling(10).sum() > 1.5, 0.3, -0.1)
    s['v2'] = np.where((ret60 > 0) & (ibs < 0.2), 1.5, 0.2)
    s['v12'] = np.where((range_climax > 3.0) & (rets < 0), 1.5, 0.2)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['opt3_mag_vol'] = - ema(rets.abs() / (vol20 + 1e-10), 5)
    s['it1_sig20'] = -(rng / close).rolling(20).std() / (vol20 + 1e-10)
    s['mr51'] = (ema(close, 60) - close) / close
    s['vol_skew_2'] = np.where(vol.rolling(60).skew() > 2.0, -0.4, 0.1)
    s['rsi14_low'] = np.where(rsi14 < 30, 1.4, 0.0)
    s['ibs_ma20_low'] = np.where(ibs < ibs.rolling(20).mean(), 1.3, 0.0)
    s['h_accel_50'] = np.where((m12-s9).diff(50) > 0, 1.0, 0.0)
    s['Payday_v2'] = np.where(dom.isin([1, 15]), 0.8, 0.0)
    s['Day15_v2'] = np.where(dom == 15, 1.0, 0.0)
    s['v5'] = np.where(((rv5 / rv20) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['FriDeRisk_Old'] = np.where(df.index.dayofweek == 4, -0.3, 0.1)
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['sig41'] = -((rets * vol).rolling(20).mean() / ((rets * vol).rolling(20).std() + 1e-6))
    s['FriDeRisk'] = np.where(df.index.dayofweek == 4, -1.0, 0.0)
    s['OpExMon'] = np.where((df.index.day >= 15) & (df.index.day <= 21) & (df.index.dayofweek == 0), 1.0, 0.0)
    s['Payday_1_15'] = np.where(df.index.day.isin([1, 15]), 1.0, 0.0)
    s['s100'] = np.where((open_p / close.shift(1) - 1 > 0.01) & (close < open_p), -1.0, 0.0)
    s['BB_Squeeze'] = np.where(((4 * std20) / (sma20 + 1e-10)) < ((4 * std20) / (sma20 + 1e-10)).rolling(100).quantile(0.1), 0.5, 0.0)
    s['HV_Break'] = np.where((close < sma20) & (vol / (sma(vol, 20) + 1e-10) > 1.5), -0.4, 0.0)
    s['S_Iter3'] = np.where((rng > sma(rng, 20)) & (close < sma20), -0.3, 0.0)
    s['S_Iter4'] = np.where((close > close.shift(4)).astype(int).rolling(9).sum() == 9, -0.5, 0.0)
    s['MR_8'] = np.where(cci < -150, 0.3, 0.0)
    s['MR_6'] = np.where(get_rsi(close, 20) < 30, 0.2, 0.0)
    s['LoVol_PB'] = np.where((m12 < s9) & (vol < sma(vol, 20)), 0.4, -0.2)
    s['Thanksgiving'] = np.where((df.index.month == 11) & (df.index.day >= 20) & (df.index.day <= 28), 0.4, -0.1)
    s['ibs_mean_revert_wide'] = np.where((ibs < 0.2) & ((close - sma20)/sma20 < -0.05), 1.5, 0.0)
    s['MeanDev10'] = np.where((close - sma10) / (sma10 + 1e-10) < -0.07, 1.5, 0.0)
    s['RangeExpOS'] = np.where((rng > rng.shift(1) * 1.5) & (ibs < 0.05), 1.5, 0.0)
    s['MACD_Zero_Cross'] = np.where(m12 > 0, 0.2, -0.2)
    s['MACD_RSI_OS'] = np.where((macd_h > macd_h.shift(1)) & (rsi2 < 10), 0.8, 0.0)
    s['MACD_Quick'] = np.where(m3_7 > s3_7, 0.4, -0.3)
    s['IBS_Mean_3'] = np.where(ibs.rolling(3).mean() < 0.2, 1.3, 0.0)
    s['IBS_Return_Pos'] = np.where((ibs < 0.2) & (rets > 0), 1.2, 0.0)
    s['IBS_Bounce'] = np.where((ibs.shift(1) < 0.1) & (ibs > 0.3), 1.1, 0.0)
    s['IBS_RSI2_Cross'] = np.where((ibs < 0.3) & (rsi2 > 20) & (rsi2.shift(1) <= 20), 1.1, 0.0)
    s['IBS_DoubleHigh'] = np.where((ibs > 0.8) & (ibs.shift(1) > 0.8), -0.6, 0.0)
    s['MACD_EMA_Filt'] = np.where((close > sma50) & (macd_h > 0), 0.5, -0.2)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_dbc(df_input):
    df = df_input.copy()
    close, vol, high, low, op = df['Close'], df['Volume'], df['High'], df['Low'], df['Open']
    rets = close.pct_change()

    def calculate_macd(series, fast, slow, signal_span):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        return macd_line, signal_line

    # Pre-computations
    m12, s12 = calculate_macd(close, 12, 26, 9)
    m5, s5 = calculate_macd(close, 5, 13, 1)
    m24, s24 = calculate_macd(close, 24, 52, 18)
    h12 = m12 - s12
    vol_std20 = rets.rolling(20).std()
    gap = (op / close.shift(1)) - 1
    m_s, dy_s, dw_s = df.index.month, df.index.day, df.index.dayofweek
    dy_s, m_s, dw_s = pd.Series(dy_s, index=df.index), pd.Series(m_s, index=df.index), pd.Series(dw_s, index=df.index)
    std10, std50 = close.rolling(10).std(), close.rolling(50).std()
    ma20, ema100 = close.rolling(20).mean(), close.ewm(span=100).mean()
    ha20, ha5 = (high - low).rolling(20).mean(), (high - low).rolling(5).mean()

    s = pd.DataFrame(index=df.index)
    
    # 1-18 Baseline Signals
    s['Vol_Conv'] = np.where((vol.pct_change().rolling(20).std() / (vol_std20 + 1e-10)) > 1.2, 1.5, 0.5)
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['Payday'] = np.where(dy_s.isin([1, 15, 30]), 0.5, -0.1)
    s['FriDeRisk'] = np.where(dw_s == 4, -0.3, 0.1)
    s['Holiday_Frontrun'] = np.where((m_s == 11) & (dy_s >= 20), 0.3, -0.1)
    s['WinDressing'] = np.where((m_s.isin([6,12])) & (dy_s >= 28), 0.5, -0.1)
    s['Weekend_Gap'] = np.where((dw_s == 0) & (gap < -0.005), 0.4, -0.1)
    s['MR_EMA_Top'] = np.where(close > close.ewm(span=20).mean() * 1.05, -1.0, 0.0)
    s['V_SellerExhaust'] = np.where(((close - low) / (high - low + 1e-10) < 0.1) & (vol > vol.rolling(20).mean() * 1.5) & (get_rsi(close, 14) < 25), 0.5, -0.1)
    s['B_Donchian'] = np.where(close > high.rolling(50).max().shift(1), 0.5, -0.1)
    s['S_ThursdayTurn'] = np.where(dw_s == 3, 0.2, -0.1)
    s['Stdev_Ratio'] = np.where(std10 / (std50 + 1e-10) > 1.2, 0.3, -0.1)
    s['Price_Stretch'] = np.where(close / ema100 > 1.05, 0.4, -0.1)
    s['VIX_Proxy'] = np.where((close / (close.rolling(20).max() + 1e-10) - 1) < -0.05, 0.5, -0.1)
    s['ZScore_Price'] = np.where(((close - close.rolling(20).mean()) / (close.rolling(20).std()+1e-10)) > 1.0, 0.4, np.where(((close - close.rolling(20).mean()) / (close.rolling(20).std()+1e-10)) < -1.0, -0.4, 0))
    s['Intra_Vol_Conc'] = np.where(vol / (high-low+1e-10) > (vol/(high-low+1e-10)).rolling(20).mean(), -0.2, 0.1)
    s['MACD_Vola_Bands'] = np.where(m12 > m12.rolling(20).mean() + 2*m12.rolling(20).std(), -0.5, 0.1)
    s['MACD_Dual_Confirm'] = np.where((m5 > s5) & (m12 > s12), 0.5, -0.2)
    s['MACD_Fractal_D'] = np.where(m12.diff(10).abs() / m12.diff().abs().rolling(10).sum() > 0.6, 0.3, -0.1)
    s['Up_Down_Ratio'] = np.where(vol.where(rets > 0).rolling(10).sum() / vol.where(rets < 0).rolling(10).sum() > 1.5, 0.3, -0.1)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1, 1.5)
    return exposure

def strat_ewj(df):
    df = df.copy()

    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = close.pct_change()
    month, day, dow = df.index.month, df.index.day, df.index.dayofweek
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()
    def std(s, n): return s.rolling(n).std()
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # Core Technicals
    sma10, sma20, sma50, sma200 = sma(close, 10), sma(close, 20), sma(close, 50), sma(close, 200)
    std20, rsi14, rsi2 = std(close, 20), get_rsi(close, 14), get_rsi(close, 2)
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    
    # MACD Block
    m12 = ema(close, 12) - ema(close, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    macd_h = m12 - s9
    
    m3_7 = ema(close, 3) - ema(close, 7)
    s3_7 = m3_7.ewm(span=2, adjust=False).mean()

    # CCI and MAD
    tp = (high + low + close) / 3
    sma_tp = sma(tp, 20)
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # Volatility / Volume Technicals
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    ha20, ha5 = rng.rolling(20).mean(), rng.rolling(5).mean()
    vol_avg20 = vol.rolling(20).mean()
    vol_std20 = rets.rolling(20).std()
    v_norm = vol / sma(vol, 50)
    range_climax = rng / rng.rolling(50).mean().replace(0, np.nan)
    
    # MFI
    pmf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    nmf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (pmf / (nmf + 1e-6))))
    
    # UO
    uo_bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4 * (uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2 * (uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    l24_52 = ema(close, 24) - ema(close, 52)
    h3 = l24_52 - ema(l24_52, 18)
    
    rv5, rv20 = std(rets, 5) * 15.87, std(rets, 20) * 15.87
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (vol_std20 + 1e-10)
    dom, dow, month = df.index.day, df.index.dayofweek, df.index.month
    vol20 = rets.rolling(20).std()

    def calculate_macd(series, fast, slow, signal_span):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        return macd_line, signal_line

    def get_macd_local(s, fast=12, slow=26, signal=9):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    vm_p, vm_m = (high - low.shift(1)).abs(), (low - high.shift(1)).abs()
    vi_p, vi_m = vm_p.rolling(14).sum() / tr.rolling(14).sum(), vm_m.rolling(14).sum() / tr.rolling(14).sum()

    m_s, dy_s, dw_s = df.index.month, df.index.day, df.index.dayofweek
    dy_s, m_s, dw_s = pd.Series(dy_s, index=df.index), pd.Series(m_s, index=df.index), pd.Series(dw_s, index=df.index)

    gap = (open_p / close.shift(1)) - 1
    std10, std50 = close.rolling(10).std(), close.rolling(50).std()
    ma20, ema100 = close.rolling(20).mean(), close.ewm(span=100).mean()
    m5, s5 = calculate_macd(close, 5, 13, 1)

    ha20, ha5 = (high - low).rolling(20).mean(), (high - low).rolling(5).mean()
    m12, s12 = calculate_macd(close, 12, 26, 9)
    h12 = m12 - s12
    vol_proxy = (high - low).rolling(14).std()

    m, dy, dw = df.index.month, df.index.day, df.index.dayofweek
    ret60 = close.pct_change(60)
    returns = close.pct_change()
    semi_down = returns.clip(upper=0).rolling(20).std()
    v20, v60 = rets.rolling(20).std(), rets.rolling(60).std()
    ploc = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
    vsma20 = vol.rolling(20).mean()

    tp = (high + low + close) / 3
    mf = tp * vol
    # Use the df.index to ensure alignment in the MFR calculation
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    mfr = pos_mf / (neg_mf + 1e-10)
    d_mid = (high.rolling(20).max() + low.rolling(20).min()) / 2
    atr20 = (high - low).rolling(20).mean()
    vwap20 = (vol * close).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)
    emv = (((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)) / (vol / 1000000 / (high - low + 1e-10) + 1e-10)
    emv_z = (emv - emv.rolling(20).mean()) / (emv.rolling(20).std() + 1e-10)
    mv, sv, hv = get_macd_local(vol, 5, 15, 5)

    # --- Feature Engineering ---
    df['Ret'] = df['Close'].pct_change()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Vol10'] = df['Ret'].rolling(10).std() * np.sqrt(252)
    df['Vol20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
    df['Vol60'] = df['Ret'].rolling(60).std() * np.sqrt(252)
    df['ER10'] = df['Close'].diff(10).abs() / (df['Close'].diff().abs().rolling(10).sum() + 1e-10)
    
    vol = df['Volume']
    df['Vol_SMA20'] = vol.rolling(20).mean()
    df['RelVol'] = vol / (df['Vol_SMA20'] + 1e-10)
    df['ibs'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    pk = 100 * (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min())
    rsi = get_rsi(close, 14)

    s = pd.DataFrame(index=df.index)

    s['i2_c6'] = np.where(vol < vol.shift(1), 0.5, -0.5)
    s['i3_c3'] = np.where(ibs.rolling(5).mean() < 0.3, 1.2, 0.0)
    s['i5_c12'] = np.where(rsi < 20, 1.5, -0.5)
    s['L1_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L2_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L3_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L4_C12'] = np.where(vol.pct_change() > 0.5, -1.0, 0.5)

    s['IBS_Donch_High_Fail'] = np.where((df['ibs'] > 0.8) & (df['High'] < df['High'].rolling(20).max()), -0.5, 0.0)
    s['IBS_Short_Vol_Conviction'] = np.where((df['ibs'] > 0.8) & (vol > df['Vol_SMA20']), -0.4, 0.0)
    s['Low_Vol_Pullback'] = np.where((df['Ret'] < 0) & (df['RelVol'] < 0.7), 0.8, 0.0)
    s['Vol_Dry_Up'] = np.where(df['Volume'] < df['Volume'].rolling(50).min().shift(1), -0.3, 0.0)
    s['MACD_Vol_Mom'] = np.where(hv > 0, -0.3, 0.1)
    s['Kelt_Pos'] = np.where(close > (sma20 + 2*atr20), -0.4, np.where(close < (sma20 - 2*atr20), 0.4, 0.1))
    s['Std_Bias'] = np.where(ibs.rolling(10).mean() / (ibs.rolling(10).std() + 1e-10) > 1.5, -0.4, 0.1)
    s['Resistance_Prox'] = np.where((close > high.rolling(60).max() * 0.995) & (vol < vol.rolling(20).mean()), -0.5, 0.0)
    s['VolHedge'] = np.where(v20 * np.sqrt(252) > v60 * np.sqrt(252) * 1.3, -1.0, 0.0)
    s['Price_Location'] = np.where(ploc > 1, -0.5, 0.2)
    s['VolATR'] = np.where((vol / (vsma20 + 1e-10) > 1.5) & (rets < 0), 1.0, 0.0)
    s['IBS_Extreme_S'] = np.where(ibs > 0.98, -1.0, 0.0)
    s['IBS_Ultra_S'] = np.where(ibs > 0.99, -1.0, 0.0)
    s['IBS_Trend_L'] = np.where((ibs < 0.1) & (close > sma200), 1.5, 0.0)
    s['High_IBS_Fade'] = np.where(ibs > 0.95, -1.0, 0.0)
    s['Vol_Vol_Conviction'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['Gap_Ratio'] = np.where((open_p - close.shift(1)).abs() > (close - open_p).abs(), -0.5, 0.5)
    s['Gap_Fade'] = np.where((open_p - close.shift(1)) / (close.shift(1) + 1e-10) > 0.01, -0.5, 0.2)
    s['DonchMid'] = np.where((close < d_mid) & (close > d_mid * 0.99), 0.5, 0.0)
    s['Tail_Risk'] = returns.rolling(60).quantile(0.05) / (semi_down + 1e-10) 
    s['Asym_Conv'] = semi_down.rolling(20).corr(returns.rolling(20).std())
    s['MR_EMA_Top'] = np.where(close > close.ewm(span=20).mean() * 1.05, -1.0, 0.0)
    s['S_ThursdayTurn'] = np.where(dw_s == 3, 0.2, -0.1)
    s['MACD_Vola_Bands'] = np.where(m12 > m12.rolling(20).mean() + 2*m12.rolling(20).std(), -0.5, 0.1)
    s['macd_slow_hist_pos'] = np.where(h3 > 0, 0.3, -0.3)
    s['Week_3_Bearish'] = np.where((dom >= 15) & (dom <= 21), -0.3, 0.0)
    s['IBS'] = np.where(ibs < 0.2, 0.3, -0.1)
    s['ibs_ma20_low'] = np.where(ibs < ibs.rolling(20).mean(), 1.3, 0.0)
    s['h_accel_50'] = np.where((m12-s9).diff(50) > 0, 1.0, 0.0)
    s['Payday_v2'] = np.where(dom.isin([1, 15]), 0.8, 0.0)
    s['Sept_Mid_v2'] = np.where((month == 9) & (dom >= 15) & (dom <= 25), -1.0, 0.0)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['v12'] = np.where((range_climax > 3.0) & (rets < 0), 1.5, 0.2)
    s['it1_sig20'] = -(rng / close).rolling(20).std() / (vol20 + 1e-10)
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['mr51'] = (ema(close, 60) - close) / close
    s['c40'] = np.where(uo > 70, -1.0, 0.0)
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['CrashProtector'] = np.where((df.index.month == 9) | ((df.index.month == 10) & (df.index.day <= 25)), -0.5, 0.0)
    s['OpExMon'] = np.where((df.index.day >= 15) & (df.index.day <= 21) & (df.index.dayofweek == 0), 1.0, 0.0)
    s['s100'] = np.where((open_p / close.shift(1) - 1 > 0.01) & (close < open_p), -1.0, 0.0)
    s['DeathCross'] = np.where(sma50 < sma200, -0.5, 0.0)
    s['SMA200_Slope'] = np.where(sma200.diff(5) > 0, 0.2, -0.2)
    s['BB_Squeeze'] = np.where(((4 * std20) / (sma20 + 1e-10)) < ((4 * std20) / (sma20 + 1e-10)).rolling(100).quantile(0.1), 0.5, 0.0)
    s['HV_Break'] = np.where((close < sma20) & (vol / (sma(vol, 20) + 1e-10) > 1.5), -0.4, 0.0)
    s['S_Iter3'] = np.where((rng > sma(rng, 20)) & (close < sma20), -0.3, 0.0)
    s['S_Iter4'] = np.where((close > close.shift(4)).astype(int).rolling(9).sum() == 9, -0.5, 0.0)
    s['AV_GARCH_Extreme'] = np.where(rets**2 > (rets**2).rolling(252).mean() * 10, -1.0, 0.0)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['MeanDev10'] = np.where((close - sma10) / (sma10 + 1e-10) < -0.07, 1.5, 0.0)
    s['MACD_Zero_Cross'] = np.where(m12 > 0, 0.2, -0.2)
    s['MACD_RSI_OS'] = np.where((macd_h > macd_h.shift(1)) & (rsi2 < 10), 0.8, 0.0)
    s['MACD_Quick'] = np.where(m3_7 > s3_7, 0.4, -0.3)
    s['IBS_Return_Pos'] = np.where((ibs < 0.2) & (rets > 0), 1.2, 0.0)
    s['IBS_Bounce'] = np.where((ibs.shift(1) < 0.1) & (ibs > 0.3), 1.1, 0.0)
    s['IBS_QuietFear'] = np.where((ibs < 0.1) & (rets.abs() < 0.005), 1.1, 0.0)
    s['IBS_BullSupp'] = np.where((ibs < 0.3) & (close > sma20), 1.0, 0.0)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['ADX_Proxy'] = np.where((m12 > s12) & (vol_proxy > vol_proxy.rolling(20).mean()), 0.5, -0.2)
    s['Payday'] = np.where(pd.Series(dy, index=df.index).isin([1, 15, 30]), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((m == 11) & (dy >= 20), 0.3, -0.1)
    s['SpringFever'] = np.where(m == 4, 0.3, -0.1)
    s['WinDressing'] = np.where((pd.Series(m, index=df.index).isin([6,12])) & (dy >= 28), 0.5, -0.1)
    s['Thanksgiving'] = np.where((m == 11) & (dy >= 20) & (dy <= 28), 0.4, -0.1)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_eww(df):
    """
    Final optimized systematic strategy for EWW.
    Developed via 5 iterations of drawdown analysis and volume-volatility-short signal integration.
    Exposure: -1.0 to 1.5. 
    Timing: Close-to-Close (Signals generated at Close of T, executed for period T to T+1).
    """
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    rets = close.pct_change().fillna(0)
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def std(s, n): return s.rolling(n).std()

    # --- Pre-computations ---
    vol20 = rets.rolling(20).std()
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    ret60 = close.pct_change(60)
    rv5, rv20 = rets.rolling(5).std() * 15.87, vol20 * 15.87
    tp = (high + low + close) / 3
    mfi_raw = tp * vol
    mfi = 100 - (100 / (1 + (mfi_raw.where(tp > tp.shift(1), 0).rolling(14).sum() / 
                            mfi_raw.where(tp < tp.shift(1), 0).rolling(14).sum().replace(0, 1e-6))))
    m12 = ema(close, 12) - ema(close, 26)
    s12 = m12.ewm(span=9, adjust=False).mean()
    h12 = m12 - s12
    range_climax = rng / rng.rolling(50).mean().replace(0, np.nan)
    gap_v = (df['Open'] / close.shift(1) - 1).rolling(20).std()
    er = rets.rolling(10).sum().abs() / rets.abs().rolling(10).sum()

    s = pd.DataFrame(index=df.index)
    
    # --- Ensemble Signals ---
    # Core Systematic
    s['v2'] = np.where((ret60 > 0) & (ibs < 0.2), 1.5, 0.2)
    s['v5'] = np.where(((rv5 / rv20) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['v10'] = np.where((close < close.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['v11'] = np.where(0.5 * vol.rolling(20).mean() * (close.diff(3)**2) > \
                        (0.5 * vol.rolling(20).mean() * (close.diff(3)**2)).rolling(100).quantile(0.95), -0.5, 0.2)
    s['v12'] = np.where((range_climax > 3.0) & (rets < 0), 1.5, 0.2)
    s['FriDeRisk'] = np.where(df.index.dayofweek == 4, -0.3, 0.1)
    s['Vol_Conv'] = np.where(vol.pct_change().rolling(20).std() / (vol20 + 1e-10) > 1.2, 1.5, 0.5)
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    
    # Volatility and Reversion Filters
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['opt3_mag_vol'] = - ema(rets.abs() / (vol20 + 1e-10), 5)
    s['it1_sig20'] = -(rng / close).rolling(20).std() / (vol20 + 1e-10)
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['mr51'] = (ema(close, 60) - close) / close
    s['v4_10'] = (close - df['Open']) / (rng + 1e-6)
    
    # Short-Only Optimization Iterations
    pmf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    nmf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi_s = 100 - (100 / (1 + (pmf / (nmf + 1e-6))))
    s['s32'] = np.where(mfi_s > 80, -1.0, 0.0)
    s['s41'] = np.where(rets.rolling(252).kurt() > 10, -0.5, 0.0)
    s['c2'] = np.where(close > close.rolling(20).mean() + 2.5 * std(close, 20), -1.0, 0.0)
    
    # Ultimate Oscillator Proxy Short
    def bp(df): return close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    def tr(df): return pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4 * (bp(df).rolling(7).sum()/tr(df).rolling(7).sum()) + 2 * (bp(df).rolling(14).sum()/tr(df).rolling(14).sum()) + (bp(df).rolling(28).sum()/tr(df).rolling(28).sum())) / 7
    s['c40'] = np.where(uo > 70, -1.0, 0.0)
    
    # Multi-Frequency RSI Exhaustion
    delta = close.diff()
    up = delta.where(delta > 0, 0); down = delta.where(delta < 0, 0).abs()
    rsi14 = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean().replace(0, 1e-6)))
    rsi7 = 100 - (100 / (1 + up.rolling(7).mean() / down.rolling(7).mean().replace(0, 1e-6)))
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['c91'] = np.where(rsi7 > 85, -1.0, 0.0)

    # --- Weighting & Execution ---
    # Shifted by 1 to execute based on T-closing data for the T to T+1 interval
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_fxi(df):
    """
    Consolidated Final Strategy for FXI.
    Ensemble of original signals + 5 iterative additions.
    """
    # 1. Inputs
    c, h, l, v, o = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = c.pct_change().fillna(0)
    
    # Helpers
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()
    def std(s, n): return s.rolling(n).std()

    # 2. Indicators for Base Signals
    rng = (h - l).replace(0, 1e-6)
    ibs = (c - l) / rng
    tp = (h + l + c) / 3
    mfi_raw = tp * v
    vol20 = rets.rolling(20).std()
    m12 = ema(c, 12) - ema(c, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    gap_v = (o / c.shift(1) - 1).rolling(20).std()
    dom, month = df.index.day, df.index.month
    v_norm = v / sma(v, 50)
    std20 = std(rets, 20)
    cmf = (((c - l) - (h - c)) / rng * v).rolling(20).sum() / v.rolling(20).sum().replace(0, 1e-6)
    vol_proxy = rng.rolling(14).std()
    ma50 = sma(c, 50)
    r_neg = rets.where(rets < 0, 0)
    vov = std(vol20, 20)

    s = pd.DataFrame(index=df.index)
    
    # Original Ensemble
    s['ADX_Proxy'] = np.where((m12 > s9) & (vol_proxy > vol_proxy.rolling(20).mean()), 0.5, -0.2)
    s['Vol_Conv'] = np.where(v.pct_change().rolling(20).std() / (vol20 + 1e-10) > 1.2, 1.5, 0.5)
    s['quiet_acc_3'] = np.where((v_norm > 1.2) & (std20 < std(rets, 100) * 0.8), 1.0, 0.0)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['cmf_rev'] = np.where(cmf < -0.2, 1.4, 0.0)
    s['ibs_rank'] = np.where(ibs < ibs.rolling(100).quantile(0.05), 1.5, 0.0)
    s['q_end_wd'] = np.where(pd.Series(month, index=df.index).isin([3, 6, 9, 12]) & (dom >= 25), 0.6, 0.0)
    s['last_day_q'] = np.where(dom == df.index.days_in_month, 0.7, 0.0)
    
    # Fibonacci Signals
    roll_h13, roll_l13 = h.rolling(13).max(), l.rolling(13).min()
    fib_786 = roll_l13 + 0.786 * (roll_h13 - roll_l13)
    s['fib_786_res_13'] = np.where(c > fib_786, -0.6, 0.0)
    roll_h21, roll_l21 = h.rolling(21).max(), l.rolling(21).min()
    fib_618_21 = roll_l21 + 0.618 * (roll_h21 - roll_l21)
    fib_382_21 = roll_l21 + 0.382 * (roll_h21 - roll_l21)
    roll_h34, roll_l34 = h.rolling(34).max(), l.rolling(34).min()
    fib_618_34 = roll_l34 + 0.618 * (roll_h34 - roll_l34)
    s['fib_618_fail_34'] = np.where((c < fib_618_34) & (h > fib_618_34), -0.7, 0.0)
    s['S8_IBS_Trend_Rej'] = np.where((ibs > 0.8) & (c < sma(c, 200)), -1.0, 0.0)
    s['S3_4_VolPrDiv'] = np.where((c > c.shift(1)) & (v < v.shift(1)*0.8), -0.7, 0.0)
    s['S4_10_SeqHH'] = np.where((c > c.shift(1)).astype(int).rolling(4).sum() == 4, -0.6, 0.0)

    # Final Exposure Logic
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_fxu(df):
    # Preprocessing makes 'Date' the index, so we access columns directly
    close, low, high, vol, open_p = df['Close'], df['Low'], df['High'], df['Volume'], df['Open']
    
    # Access date components from the index instead of a column
    day = df.index.day
    month = df.index.month
    
    # Pre-calculations
    returns = close.pct_change()
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    vol20 = returns.rolling(20).std() * np.sqrt(252)
    # ibs = (close - low) / (high - low + 1e-10) # Defined but unused in original snippet, kept for reference
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (returns.rolling(20).std() + 1e-10)
    semi_down = returns.clip(upper=0).rolling(20).std()
    
    s = pd.DataFrame(index=df.index)
    
    s['Golden_Vol'] = np.where((sma50 > sma200) & (vol20 < 0.20), 1.5, 0.0)
    s['Vol_Vol_Conviction'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['Gap_Ratio'] = np.where((open_p - close.shift(1)).abs() > (close - open_p).abs(), -0.5, 0.5)
    s['Gap_Fade'] = np.where((open_p - close.shift(1)) / (close.shift(1) + 1e-10) > 0.01, -0.5, 0.2)
    
    tp = (high + low + close) / 3
    mf = tp * vol
    # Use the df.index to ensure alignment in the MFR calculation
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    mfr = pos_mf / (neg_mf + 1e-10)
    s['MFI5'] = np.where(100 - (100 / (1 + mfr)) < 10, 1.0, 0.0)
    
    d_mid = (high.rolling(20).max() + low.rolling(20).min()) / 2
    s['DonchMid'] = np.where((close < d_mid) & (close > d_mid * 0.99), 0.5, 0.0)
    s['MidMonthRev'] = np.where((day >= 18) & (day <= 22), -0.3, 0.1)
    s['AugExit'] = np.where(month == 8, -1.0, 0.1)
    atr20 = (high - low).rolling(20).mean()
    s['Range_Hedge'] = np.where((high - low).rolling(5).mean() > atr20, -0.5, 0.5)
    
    # Asymmetric Volatility Signals
    # er = close.diff(10).abs() / returns.abs().rolling(10).sum() # Defined but unused, kept for reference
    s['Down_Beta'] = returns.rolling(20).corr(semi_down) 
    s['Tail_Risk'] = returns.rolling(60).quantile(0.05) / (semi_down + 1e-10) 
    s['Asym_Conv'] = semi_down.rolling(20).corr(returns.rolling(20).std()) 
    
    # Ensemble with Shift 1 to prevent lookahead bias
    ensemble_exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1, 1.5)
    return ensemble_exposure

def strat_gld(df):
    df = df.copy()

    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = close.pct_change()
    month, day, dow = df.index.month, df.index.day, df.index.dayofweek
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()
    def std(s, n): return s.rolling(n).std()
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # Core Technicals
    sma10, sma20, sma50, sma200 = sma(close, 10), sma(close, 20), sma(close, 50), sma(close, 200)
    std20, rsi14, rsi2 = std(close, 20), get_rsi(close, 14), get_rsi(close, 2)
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    
    # MACD Block
    m12 = ema(close, 12) - ema(close, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    macd_h = m12 - s9
    
    m3_7 = ema(close, 3) - ema(close, 7)
    s3_7 = m3_7.ewm(span=2, adjust=False).mean()

    # CCI and MAD
    tp = (high + low + close) / 3
    sma_tp = sma(tp, 20)
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # Volatility / Volume Technicals
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    ha20, ha5 = rng.rolling(20).mean(), rng.rolling(5).mean()
    vol_avg20 = vol.rolling(20).mean()
    vol_std20 = rets.rolling(20).std()
    v_norm = vol / sma(vol, 50)
    range_climax = rng / rng.rolling(50).mean().replace(0, np.nan)
    
    # MFI
    pmf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    nmf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (pmf / (nmf + 1e-6))))
    
    # UO
    uo_bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4 * (uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2 * (uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    l24_52 = ema(close, 24) - ema(close, 52)
    h3 = l24_52 - ema(l24_52, 18)
    
    rv5, rv20 = std(rets, 5) * 15.87, std(rets, 20) * 15.87
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (vol_std20 + 1e-10)
    dom, dow, month = df.index.day, df.index.dayofweek, df.index.month
    vol20 = rets.rolling(20).std()

    def get_macd_local(s, fast=12, slow=26, signal=9):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    vm_p, vm_m = (high - low.shift(1)).abs(), (low - high.shift(1)).abs()
    vi_p, vi_m = vm_p.rolling(14).sum() / tr.rolling(14).sum(), vm_m.rolling(14).sum() / tr.rolling(14).sum()

    m_s, dy_s, dw_s = df.index.month, df.index.day, df.index.dayofweek
    dy_s, m_s, dw_s = pd.Series(dy_s, index=df.index), pd.Series(m_s, index=df.index), pd.Series(dw_s, index=df.index)

    gap = (open_p / close.shift(1)) - 1
    std10, std50 = close.rolling(10).std(), close.rolling(50).std()
    ma20, ema100 = close.rolling(20).mean(), close.ewm(span=100).mean()
    m5, s5, _ = get_macd_local(close, 5, 13, 1)

    ha20, ha5 = (high - low).rolling(20).mean(), (high - low).rolling(5).mean()
    m12, s12, _ = get_macd_local(close, 12, 26, 9)
    h12 = m12 - s12
    vol_proxy = (high - low).rolling(14).std()

    m, dy, dw = df.index.month, df.index.day, df.index.dayofweek
    ret60 = close.pct_change(60)
    returns = close.pct_change()
    semi_down = returns.clip(upper=0).rolling(20).std()
    v20, v60 = rets.rolling(20).std(), rets.rolling(60).std()
    ploc = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
    vsma20 = vol.rolling(20).mean()

    tp = (high + low + close) / 3
    mf = tp * vol
    # Use the df.index to ensure alignment in the MFR calculation
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    mfr = pos_mf / (neg_mf + 1e-10)
    d_mid = (high.rolling(20).max() + low.rolling(20).min()) / 2
    atr20 = (high - low).rolling(20).mean()
    vwap20 = (vol * close).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)
    emv = (((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)) / (vol / 1000000 / (high - low + 1e-10) + 1e-10)
    emv_z = (emv - emv.rolling(20).mean()) / (emv.rolling(20).std() + 1e-10)
    mv, sv, hv = get_macd_local(vol, 5, 15, 5)

    # --- Feature Engineering ---
    df['Ret'] = df['Close'].pct_change()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Vol10'] = df['Ret'].rolling(10).std() * np.sqrt(252)
    df['Vol20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
    df['Vol60'] = df['Ret'].rolling(60).std() * np.sqrt(252)
    df['ER10'] = df['Close'].diff(10).abs() / (df['Close'].diff().abs().rolling(10).sum() + 1e-10)
    
    vol = df['Volume']
    df['Vol_SMA20'] = vol.rolling(20).mean()
    df['RelVol'] = vol / (df['Vol_SMA20'] + 1e-10)
    df['ibs'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    pk = 100 * (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min())
    rsi = get_rsi(close, 14)

    s = pd.DataFrame(index=df.index)

    s['s6'] = np.where(pk > pk.rolling(3).mean(), 1.0, -0.5)
    s['s7'] = np.where((df.index.day >= 10) & (df.index.day <= 15), 1.0, 0.1)
    s['s10'] = np.where(close > high.rolling(20).max().shift(1), 1.0, -0.5)
    s['c6'] = np.where(close > close.shift(20), 1.0, -1.0)
    s['L1_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L2_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L3_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L4_C12'] = np.where(vol.pct_change() > 0.5, -1.0, 0.5)
    s['Low_Vol_Pullback'] = np.where((df['Ret'] < 0) & (df['RelVol'] < 0.7), 0.8, 0.0)
    s['EMV_Z'] = np.where(emv_z > 1.0, 0.3, -0.3)
    s['Kelt_Pos'] = np.where(close > (sma20 + 2*atr20), -0.4, np.where(close < (sma20 - 2*atr20), 0.4, 0.1))
    s['Vortex_Trend'] = np.where(vi_p > vi_m, 0.4, -0.2)
    s['Std_Bias'] = np.where(ibs.rolling(10).mean() / (ibs.rolling(10).std() + 1e-10) > 1.5, -0.4, 0.1)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['Price_Location'] = np.where(ploc > 1, -0.5, 0.2)
    s['Bear_Range_Expansion'] = np.where((high - low) > ha20 * 1.5, -0.3, 0.0)
    s['Golden_Vol'] = np.where((sma50 > sma200) & (vol20 < 0.20), 1.5, 0.0)
    s['Vol_Vol_Conviction'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['DonchMid'] = np.where((close < d_mid) & (close > d_mid * 0.99), 0.5, 0.0)
    s['Range_Hedge'] = np.where((high - low).rolling(5).mean() > atr20, -0.5, 0.5)
    s['Down_Beta'] = returns.rolling(20).corr(semi_down)
    s['Weekend_Gap'] = np.where((dw_s == 0) & (gap < -0.005), 0.4, -0.1)
    s['MR_EMA_Top'] = np.where(close > close.ewm(span=20).mean() * 1.05, -1.0, 0.0)
    s['B_Donchian'] = np.where(close > high.rolling(50).max().shift(1), 0.5, -0.1)
    s['S_ThursdayTurn'] = np.where(dw_s == 3, 0.2, -0.1)
    s['Stdev_Ratio'] = np.where(std10 / (std50 + 1e-10) > 1.2, 0.3, -0.1)
    s['VIX_Proxy'] = np.where((close / (close.rolling(20).max() + 1e-10) - 1) < -0.05, 0.5, -0.1)
    s['MACD_Dual_Confirm'] = np.where((m5 > s5) & (m12 > s12), 0.5, -0.2)
    s['Up_Down_Ratio'] = np.where(vol.where(rets > 0).rolling(10).sum() / vol.where(rets < 0).rolling(10).sum() > 1.5, 0.3, -0.1)
    s['macd_slow_hist_pos'] = np.where(h3 > 0, 0.3, -0.3)
    s['vol_skew_2'] = np.where(vol.rolling(60).skew() > 2.0, -0.4, 0.1)
    s['m_cross_ext'] = np.where((m12 < -1) & (m12 > s9), 1.5, 0.0)
    s['Payday_v2'] = np.where(dom.isin([1, 15]), 0.8, 0.0)
    s['Day15_v2'] = np.where(dom == 15, 1.0, 0.0)
    s['B_Climax'] = np.where(((vol / sma(vol, 50)) * (rng / sma(rng, 50))) > 3.0, -1.0, 0.0)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['v12'] = np.where((range_climax > 3.0) & (rets < 0), 1.5, 0.2)
    s['FriDeRisk_Old'] = np.where(df.index.dayofweek == 4, -0.3, 0.1)
    s['opt3_mag_vol'] = - ema(rets.abs() / (vol20 + 1e-10), 5)
    s['it1_sig20'] = -(rng / close).rolling(20).std() / (vol20 + 1e-10)
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['mr51'] = (ema(close, 60) - close) / close
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['FriDeRisk'] = np.where(df.index.dayofweek == 4, -1.0, 0.0)
    s['CrashProtector'] = np.where((df.index.month == 9) | ((df.index.month == 10) & (df.index.day <= 25)), -0.5, 0.0)
    s['Payday_1_15'] = np.where(df.index.day.isin([1, 15]), 1.0, 0.0)
    s['s100'] = np.where((open_p / close.shift(1) - 1 > 0.01) & (close < open_p), -1.0, 0.0)
    s['Gene_Recessive_Dip'] = np.where((close < sma50) & (close > sma200), 1.5, 0.0)
    s['DeathCross'] = np.where(sma50 < sma200, -0.5, 0.0)
    s['S_Iter4'] = np.where((close > close.shift(4)).astype(int).rolling(9).sum() == 9, -0.5, 0.0)
    s['AV_GARCH_Extreme'] = np.where(rets**2 > (rets**2).rolling(252).mean() * 10, -1.0, 0.0)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['LoVol_PB'] = np.where((m12 < s9) & (vol < sma(vol, 20)), 0.4, -0.2)
    s['RangeExpOS'] = np.where((rng > rng.shift(1) * 1.5) & (ibs < 0.05), 1.5, 0.0)
    s['MACD_RSI_OS'] = np.where((macd_h > macd_h.shift(1)) & (rsi2 < 10), 0.8, 0.0)
    s['MACD_Early_Turn'] = np.where((macd_h > 0) & (macd_h.shift(1) < 0) & (macd_h.shift(10) < 0), 1.0, 0.0)
    s['IBS_ATR_Panic'] = np.where((ibs < 0.1) & (std(close, 14) > std(close, 14).shift(20) * 1.5), 1.5, 0.0)
    s['IBS_Bounce'] = np.where((ibs.shift(1) < 0.1) & (ibs > 0.3), 1.1, 0.0)
    s['IBS_QuietFear'] = np.where((ibs < 0.1) & (rets.abs() < 0.005), 1.1, 0.0)
    s['MACD_EMA_Filt'] = np.where((close > sma50) & (macd_h > 0), 0.5, -0.2)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    s['ADX_Proxy'] = np.where((m12 > s12) & (vol_proxy > vol_proxy.rolling(20).mean()), 0.5, -0.2)
    s['Payday'] = np.where(pd.Series(dy, index=df.index).isin([1, 15, 30]), 0.5, -0.1)
    s['JanEff'] = np.where((m == 1) & (dy <= 15), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((m == 11) & (dy >= 20), 0.3, -0.1)
    s['SpringFever'] = np.where(m == 4, 0.3, -0.1)
    s['WinDressing'] = np.where((pd.Series(m, index=df.index).isin([6,12])) & (dy >= 28), 0.5, -0.1)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_gsy(df_input):
    df = df_input.copy()
    close, vol, high, low = df['Close'], df['Volume'], df['High'], df['Low']
    rets = df['Close'].pct_change()
    m, dy, dw = df.index.month, df.index.day, df.index.dayofweek
    v20, v60 = rets.rolling(20).std(), rets.rolling(60).std()
    vsma20 = vol.rolling(20).mean()
    ibs = (close - low) / (high - low + 1e-10)
    ha20 = (high - low).rolling(20).mean()
    ploc = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
    
    # RSI
    def get_rsi(s, n):
        d = s.diff()
        g = (d.where(d > 0, 0)).rolling(n).mean()
        l = (-d.where(d < 0, 0)).rolling(n).mean()
        return 100 - (100 / (1 + (g / (l + 1e-10))))
    
    rsi2 = get_rsi(close, 2)
    sma200 = close.rolling(200).mean()
    hi10 = high.rolling(10).max().shift(1)

    s = pd.DataFrame(index=df.index)
    
    # --- Baseline Components ---
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (rets.rolling(20).std() + 1e-10)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['VolHedge'] = np.where(v20 * np.sqrt(252) > v60 * np.sqrt(252) * 1.3, -1.0, 0.0)
    s['FriDeRisk'] = np.where(dw == 4, -0.3, 0.1)
    s['JanEff'] = np.where((m == 1) & (dy <= 15), 0.5, -0.1)
    s['Price_Location'] = np.where(ploc > 1, -0.5, 0.2)
    s['Bear_Range_Expansion'] = np.where((high - low) > ha20 * 1.5, -0.3, 0.0)
    s['VolATR'] = np.where((vol / (vsma20 + 1e-10) > 1.5) & (rets < 0), 1.0, 0.0)
    s['IBS_Extreme_S'] = np.where(ibs > 0.98, -1.0, 0.0)
    s['IBS_Ultra_S'] = np.where(ibs > 0.99, -1.0, 0.0)
    s['IBS_Trend_L'] = np.where((ibs < 0.1) & (close > sma200), 1.5, 0.0)
    s['High_IBS_Fade'] = np.where(ibs > 0.95, -1.0, 0.0)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_hyg(df_input):
    df = df_input.copy()
    close, vol, high, low, op = df['Close'], df['Volume'], df['High'], df['Low'], df['Open']
    rets = df['Close'].pct_change()
    
    def get_macd_local(s, fast=12, slow=26, signal=9):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    def zscore(s, n): 
        return (s - s.rolling(n).mean()) / (s.rolling(n).std() + 1e-10)

    # Base Features
    ibs = (close - low) / (high - low + 1e-10)
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (rets.rolling(20).std() + 1e-10)
    sma20 = close.rolling(20).mean()
    v20 = rets.rolling(20).std()
    atr20 = (high - low).rolling(20).mean()
    emv = (((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)) / (vol / 1000000 / (high - low + 1e-10) + 1e-10)
    emv_z = (emv - emv.rolling(20).mean()) / (emv.rolling(20).std() + 1e-10)
    vwap20 = (vol * close).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)
    m1, s1, h1 = get_macd_local(close, 12, 26, 9)
    mv, sv, hv = get_macd_local(vol, 5, 15, 5)
    vm_p, vm_m = (high - low.shift(1)).abs(), (low - high.shift(1)).abs()
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    vi_p, vi_m = vm_p.rolling(14).sum() / tr.rolling(14).sum(), vm_m.rolling(14).sum() / tr.rolling(14).sum()
    
    s = pd.DataFrame(index=df.index)
    # Original Alphas
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['FriDeRisk'] = np.where(df.index.dayofweek == 4, -0.3, 0.1)
    s['JanEff'] = np.where((df.index.month == 1) & (df.index.day <= 15), 0.5, -0.1)
    s['IBS_Extreme_S'] = np.where(ibs > 0.98, -1.0, 0.0)
    s['Gap_Regime_2'] = np.where((op - close.shift(1)).abs().rolling(10).sum() > atr20, -0.5, 0.2)
    s['Liquidity_Regime'] = np.where((vol / (atr20 + 1e-10)) < (vol / (atr20 + 1e-10)).rolling(50).mean(), -0.3, 0.1)
    s['VWAP_Dist'] = np.where(close > vwap20, 0.2, -0.4)
    s['EMV_Z'] = np.where(emv_z > 1.0, 0.3, -0.3)
    s['Body_Exp'] = np.where(abs(close - op) > (high - low).rolling(20).mean(), 0.3, -0.1)
    s['MACD_Vol_Mom'] = np.where(hv > 0, -0.3, 0.1)
    s['Kelt_Pos'] = np.where(close > (sma20 + 2*atr20), -0.4, np.where(close < (sma20 - 2*atr20), 0.4, 0.1))
    s['Vortex_Trend'] = np.where(vi_p > vi_m, 0.4, -0.2)
    s['Std_Bias'] = np.where(ibs.rolling(10).mean() / (ibs.rolling(10).std() + 1e-10) > 1.5, -0.4, 0.1)
    s['Resistance_Prox'] = np.where((close > high.rolling(60).max() * 0.995) & (vol < vol.rolling(20).mean()), -0.5, 0.0)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_ita(df):
    df = df.copy()
    c, h, l, v, o = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = c.pct_change()
    
    # --- 1. Technical Indicator Helpers ---
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / (loss + 1e-10))))

    def get_macd(s, fast, slow, signal):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    # Indicator Pre-calculations
    sma10, sma20, sma50, sma200 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    std10, std20, std50 = c.rolling(10).std(), c.rolling(20).std(), c.rolling(50).std()
    ema20, ema100 = c.ewm(span=20).mean(), c.ewm(span=100).mean()
    rsi2, rsi14, rsi20 = get_rsi(c, 2), get_rsi(c, 14), get_rsi(c, 20)
    
    rng = (h - l).replace(0, 1e-6)
    ha5, ha20 = rng.rolling(5).mean(), rng.rolling(20).mean()
    ibs = (c - l) / rng
    atr20 = (pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)).rolling(20).mean()
    
    m12, s12, h12 = get_macd(c, 12, 26, 9)
    m5, s5, h5 = get_macd(c, 5, 13, 1)
    m3_7, s3_7, _ = get_macd(c, 3, 7, 2)
    l24_52, _, h3 = get_macd(c, 24, 52, 18)
    mv, sv, hv = get_macd(v, 5, 15, 5)

    vol20 = rets.rolling(20).std()
    vol_vol_ratio = v.pct_change().rolling(20).std() / (vol20 + 1e-10)
    v_norm = v / v.rolling(50).mean().replace(0, 1e-10)
    pk = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-10)
    
    tp = (h + l + c) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True) + 1e-10)
    
    mfi_pmf = (tp * v).where(tp > tp.shift(1), 0).rolling(14).sum()
    mfi_nmf = (tp * v).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (mfi_pmf / (mfi_nmf + 1e-6))))
    
    uo_bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4*(uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2*(uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    month, day, dow = df.index.month, df.index.day, df.index.dayofweek
    s = pd.DataFrame(index=df.index)

    # --- 2. Signal Generation (144 Unique Signals) ---
    s['s4'] = np.where(ibs < 0.2, 1.5, np.where(ibs > 0.8, -1.0, 0.5))
    s['s6'] = np.where(pk > pk.rolling(3).mean(), 1.0, -0.5)
    s['s7'] = np.where((day >= 10) & (day <= 15), 1.0, 0.1)
    s['i2_c6'] = np.where(v < v.shift(1), 0.5, -0.5)
    s['i3_c3'] = np.where(ibs.rolling(5).mean() < 0.3, 1.2, 0.0)
    s['L1_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L4_C12'] = np.where(v.pct_change() > 0.5, -1.0, 0.5)
    s['IBS_Short_Vol_Conviction'] = np.where((ibs > 0.8) & (v > v.rolling(20).mean()), -0.4, 0.0)
    s['Low_Vol_Pullback'] = np.where((rets < 0) & (v / v.rolling(20).mean() < 0.7), 0.8, 0.0)
    s['Quiet_Accum'] = np.where((v / v.rolling(20).mean() < 0.8) & (c > c.shift(1)), 0.4, 0.0)
    s['FriDeRisk'] = np.where(dow == 4, -0.3, 0.1)
    s['JanEff'] = np.where((month == 1) & (day <= 15), 0.5, -0.1)
    s['IBS_Extreme_S'] = np.where(ibs > 0.98, -1.0, 0.0)
    s['Gap_Regime_2'] = np.where((o / c.shift(1) - 1).abs().rolling(10).sum() > atr20, -0.5, 0.2)
    s['Liquidity_Regime'] = np.where((v / (atr20 + 1e-10)) < (v / (atr20 + 1e-10)).rolling(50).mean(), -0.3, 0.1)
    s['VWAP_Dist'] = np.where(c > ((v * c).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)), 0.2, -0.4)
    emv = (((h+l)/2) - ((h.shift(1)+l.shift(1))/2)) / (v/1e6/(rng+1e-10) + 1e-10)
    s['EMV_Z'] = np.where(((emv - emv.rolling(20).mean()) / (emv.rolling(20).std() + 1e-10)) > 1.0, 0.3, -0.3)
    vm_p, vm_m = (h - l.shift(1)).abs(), (l - h.shift(1)).abs()
    s['Vortex_Trend'] = np.where(vm_p.rolling(14).sum() > vm_m.rolling(14).sum(), 0.4, -0.2)
    s['Resistance_Prox'] = np.where((c > h.rolling(60).max() * 0.995) & (v < v.rolling(20).mean()), -0.5, 0.0)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['VolHedge'] = np.where(vol20*15.87 > rets.rolling(60).std()*15.87*1.3, -1.0, 0.0)
    s['Price_Location'] = np.where(((c - sma20)/(std20+1e-10)) > 1, -0.5, 0.2)
    s['Bear_Range_Expansion'] = np.where(rng > ha20 * 1.5, -0.3, 0.0)
    s['IBS_Ultra_S'] = np.where(ibs > 0.99, -1.0, 0.0)
    s['IBS_Trend_L'] = np.where((ibs < 0.1) & (c > sma200), 1.5, 0.0)
    s['Gap_Ratio'] = np.where((o - c.shift(1)).abs() > (c - o).abs(), -0.5, 0.5)
    s['Gap_Fade'] = np.where((o / c.shift(1) - 1) > 0.01, -0.5, 0.2)
    mfr_p = (tp * v).where(tp > tp.shift(1), 0).rolling(5).sum()
    mfr_n = (tp * v).where(tp < tp.shift(1), 0).rolling(5).sum()
    s['DonchMid'] = np.where((c < (h.rolling(20).max() + l.rolling(20).min())/2) & (c > (h.rolling(20).max() + l.rolling(20).min())/2 * 0.99), 0.5, 0.0)
    s['MidMonthRev'] = np.where((day >= 18) & (day <= 22), -0.3, 0.1)
    s['AugExit'] = np.where(month == 8, -1.0, 0.1)
    s['Range_Hedge'] = np.where(ha5 > atr20, -0.5, 0.5)
    semi_down = rets.clip(upper=0).rolling(20).std()
    s['Down_Beta'] = rets.rolling(20).corr(semi_down)
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['Payday'] = np.where(day.isin([1, 15, 30]), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((month == 11) & (day >= 20), 0.3, -0.1)
    s['WinDressing'] = np.where((month.isin([6,12])) & (day >= 28), 0.5, -0.1)
    s['Weekend_Gap'] = np.where((dow == 0) & ((o / c.shift(1) - 1) < -0.005), 0.4, -0.1)
    s['MR_EMA_Top'] = np.where(c > ema20 * 1.05, -1.0, 0.0)
    s['S_ThursdayTurn'] = np.where(dow == 3, 0.2, -0.1)
    s['Stdev_Ratio'] = np.where(std10 / (std50 + 1e-10) > 1.2, 0.3, -0.1)
    s['Intra_Vol_Conc'] = np.where(v / (h-l+1e-10) > (v/(h-l+1e-10)).rolling(20).mean(), -0.2, 0.1)
    s['MACD_Vola_Bands'] = np.where(m12 > m12.rolling(20).mean() + 2*m12.rolling(20).std(), -0.5, 0.1)
    s['MACD_Dual_Confirm'] = np.where((m5 > s5) & (m12 > s12), 0.5, -0.2)
    s['v2'] = np.where((c.pct_change(60) > 0) & (ibs < 0.2), 1.5, 0.2)
    s['v11'] = np.where(0.5 * v.rolling(20).mean() * (c.diff(3)**2) > (0.5 * v.rolling(20).mean() * (c.diff(3)**2)).rolling(100).quantile(0.95), -0.5, 0.2)
    s['SummerSolstice'] = np.where((month == 6) & (day >= 15) & (day <= 25), -0.5, 0.1)
    s['m_cross_ext'] = np.where((m12 < -1) & (m12 > s12), 1.5, 0.0)
    s['h_accel_50'] = np.where(h12.diff(50) > 0, 1.0, 0.0)
    s['B_Climax'] = np.where((v_norm * (rng / rng.rolling(50).mean())) > 3.0, -1.0, 0.0)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['it1_sig20'] = -(rng / c).rolling(20).std() / (vol20 + 1e-10)
    s['v5'] = np.where(((rets.rolling(5).std() / (vol20 + 1e-10)) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v10'] = np.where((c < c.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['FriDeRisk_Old'] = np.where(dow == 4, -0.3, 0.1)
    gap_v = (o / c.shift(1) - 1).rolling(20).std()
    s['opt1_gap_vol'] = -(gap_v / (gap_v.ewm(span=100).mean() + 1e-10))
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['mr51'] = (c.ewm(span=60).mean() - c) / c
    s['v4_10'] = (c - o) / (rng + 1e-6)
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['sig41'] = -((rets * v).rolling(20).mean() / ((rets * v).rolling(20).std() + 1e-6))
    s['CrashProtector'] = np.where((month == 9) | ((month == 10) & (day <= 25)), -0.5, 0.0)
    s['OpExMon'] = np.where((day >= 15) & (day <= 21) & (dow == 0), 1.0, 0.0)
    s['Payday_1_15'] = np.where(day.isin([1, 15]), 1.0, 0.0)
    s['Third_Week_Exh'] = np.where((day >= 18) & (day <= 22), -1.0, 0.0)
    s['s100'] = np.where((o / c.shift(1) - 1 > 0.01) & (c < o), -1.0, 0.0)
    s['Gene_Recessive_Dip'] = np.where((c < sma50) & (c > sma200), 1.5, 0.0)
    s['DeathCross'] = np.where(sma50 < sma200, -0.5, 0.0)
    s['S_Iter4'] = np.where((c > c.shift(4)).astype(int).rolling(9).sum() == 9, -0.5, 0.0)
    s['MR_8'] = np.where(cci < -150, 0.3, 0.0)
    s['MR_6'] = np.where(rsi20 < 30, 0.2, 0.0)
    s['I1_14'] = np.where((c < c.shift(1)).rolling(5).sum()==5, 0.3, 0.0)
    s['Orbital_Decay'] = np.where((sma10 < sma50) & (sma10.diff() < 0), -0.2, 0.0)
    s['LoVol_PB'] = np.where((m12 < s12) & (v < v.rolling(20).mean()), 0.4, -0.2)
    s['Thanksgiving'] = np.where((month == 11) & (day >= 20) & (day <= 28), 0.4, -0.1)
    s['MeanDev10'] = np.where((c - sma10) / (sma10 + 1e-10) < -0.07, 1.5, 0.0)
    s['MACD_RSI_OS'] = np.where((h12 > h12.shift(1)) & (rsi2 < 10), 0.8, 0.0)
    s['MACD_Quick'] = np.where(m3_7 > s3_7, 0.4, -0.3)
    s['IBS_Bounce'] = np.where((ibs.shift(1) < 0.1) & (ibs > 0.3), 1.1, 0.0)
    s['IBS_RSI2_Cross'] = np.where((ibs < 0.3) & (rsi2 > 20) & (rsi2.shift(1) <= 20), 1.1, 0.0)
    s['IBS_GapUpFade'] = np.where((ibs > 0.8) & (o > c.shift(1)), -0.4, 0.0)
    s['MACD_EMA_Filt'] = np.where((c > sma50) & (h12 > 0), 0.5, -0.2)

    # --- 3. Final Average ---
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_iyh(df):
    df = df.copy()

    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = close.pct_change()
    month, day, dow = df.index.month, df.index.day, df.index.dayofweek
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()
    def std(s, n): return s.rolling(n).std()
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # Core Technicals
    sma10, sma20, sma50, sma200 = sma(close, 10), sma(close, 20), sma(close, 50), sma(close, 200)
    std20, rsi14, rsi2 = std(close, 20), get_rsi(close, 14), get_rsi(close, 2)
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    
    # MACD Block
    m12 = ema(close, 12) - ema(close, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    macd_h = m12 - s9
    
    m3_7 = ema(close, 3) - ema(close, 7)
    s3_7 = m3_7.ewm(span=2, adjust=False).mean()

    # CCI and MAD
    tp = (high + low + close) / 3
    sma_tp = sma(tp, 20)
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # Volatility / Volume Technicals
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    ha20, ha5 = rng.rolling(20).mean(), rng.rolling(5).mean()
    vol_avg20 = vol.rolling(20).mean()
    vol_std20 = rets.rolling(20).std()
    v_norm = vol / sma(vol, 50)
    range_climax = rng / rng.rolling(50).mean().replace(0, np.nan)
    
    # MFI
    pmf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    nmf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (pmf / (nmf + 1e-6))))
    
    # UO
    uo_bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4 * (uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2 * (uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    l24_52 = ema(close, 24) - ema(close, 52)
    h3 = l24_52 - ema(l24_52, 18)
    
    rv5, rv20 = std(rets, 5) * 15.87, std(rets, 20) * 15.87
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (vol_std20 + 1e-10)
    dom, dow, month = df.index.day, df.index.dayofweek, df.index.month
    vol20 = rets.rolling(20).std()

    def calculate_macd(series, fast, slow, signal_span):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        return macd_line, signal_line

    def get_macd_local(s, fast=12, slow=26, signal=9):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    vm_p, vm_m = (high - low.shift(1)).abs(), (low - high.shift(1)).abs()
    vi_p, vi_m = vm_p.rolling(14).sum() / tr.rolling(14).sum(), vm_m.rolling(14).sum() / tr.rolling(14).sum()

    m_s, dy_s, dw_s = df.index.month, df.index.day, df.index.dayofweek
    dy_s, m_s, dw_s = pd.Series(dy_s, index=df.index), pd.Series(m_s, index=df.index), pd.Series(dw_s, index=df.index)

    gap = (open_p / close.shift(1)) - 1
    std10, std50 = close.rolling(10).std(), close.rolling(50).std()
    ma20, ema100 = close.rolling(20).mean(), close.ewm(span=100).mean()
    m5, s5 = calculate_macd(close, 5, 13, 1)

    ha20, ha5 = (high - low).rolling(20).mean(), (high - low).rolling(5).mean()
    m12, s12 = calculate_macd(close, 12, 26, 9)
    h12 = m12 - s12
    vol_proxy = (high - low).rolling(14).std()

    m, dy, dw = df.index.month, df.index.day, df.index.dayofweek
    ret60 = close.pct_change(60)
    returns = close.pct_change()
    semi_down = returns.clip(upper=0).rolling(20).std()
    v20, v60 = rets.rolling(20).std(), rets.rolling(60).std()
    ploc = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
    vsma20 = vol.rolling(20).mean()

    tp = (high + low + close) / 3
    mf = tp * vol
    # Use the df.index to ensure alignment in the MFR calculation
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=df.index).rolling(5).sum()
    mfr = pos_mf / (neg_mf + 1e-10)
    d_mid = (high.rolling(20).max() + low.rolling(20).min()) / 2
    atr20 = (high - low).rolling(20).mean()
    vwap20 = (vol * close).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)
    emv = (((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)) / (vol / 1000000 / (high - low + 1e-10) + 1e-10)
    emv_z = (emv - emv.rolling(20).mean()) / (emv.rolling(20).std() + 1e-10)
    mv, sv, hv = get_macd_local(vol, 5, 15, 5)

    # --- Feature Engineering ---
    df['Ret'] = df['Close'].pct_change()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Vol10'] = df['Ret'].rolling(10).std() * np.sqrt(252)
    df['Vol20'] = df['Ret'].rolling(20).std() * np.sqrt(252)
    df['Vol60'] = df['Ret'].rolling(60).std() * np.sqrt(252)
    df['ER10'] = df['Close'].diff(10).abs() / (df['Close'].diff().abs().rolling(10).sum() + 1e-10)
    
    vol = df['Volume']
    df['Vol_SMA20'] = vol.rolling(20).mean()
    df['RelVol'] = vol / (df['Vol_SMA20'] + 1e-10)
    df['ibs'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    pk = 100 * (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min())
    rsi = get_rsi(close, 14)

    s = pd.DataFrame(index=df.index)

    s['s4'] = np.where(ibs < 0.2, 1.5, np.where(ibs > 0.8, -1.0, 0.5))
    s['s6'] = np.where(pk > pk.rolling(3).mean(), 1.0, -0.5)
    s['s7'] = np.where((df.index.day >= 10) & (df.index.day <= 15), 1.0, 0.1)
    s['i3_c3'] = np.where(ibs.rolling(5).mean() < 0.3, 1.2, 0.0)
    s['L3_C4'] = np.where(ibs < 0.1, 1.5, 0.1)
    s['L4_C12'] = np.where(vol.pct_change() > 0.5, -1.0, 0.5)
    s['MACD_Hist_Deccel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['Low_Vol_Pullback'] = np.where((df['Ret'] < 0) & (df['RelVol'] < 0.7), 0.8, 0.0)
    s['Quiet_Accum'] = np.where((df['RelVol'] < 0.8) & (df['Close'] > df['Close'].shift(1)), 0.4, 0.0)
    s['Vol_Dry_Up'] = np.where(df['Volume'] < df['Volume'].rolling(50).min().shift(1), -0.3, 0.0)
    s['Body_Exp'] = np.where(abs(close - open_p) > (high - low).rolling(20).mean(), 0.3, -0.1)
    s['Kelt_Pos'] = np.where(close > (sma20 + 2*atr20), -0.4, np.where(close < (sma20 - 2*atr20), 0.4, 0.1))
    s['Vortex_Trend'] = np.where(vi_p > vi_m, 0.4, -0.2)
    s['Std_Bias'] = np.where(ibs.rolling(10).mean() / (ibs.rolling(10).std() + 1e-10) > 1.5, -0.4, 0.1)
    s['Price_Location'] = np.where(ploc > 1, -0.5, 0.2)
    s['Bear_Range_Expansion'] = np.where((high - low) > ha20 * 1.5, -0.3, 0.0)
    s['IBS_Extreme_S'] = np.where(ibs > 0.98, -1.0, 0.0)
    s['IBS_Ultra_S'] = np.where(ibs > 0.99, -1.0, 0.0)
    s['High_IBS_Fade'] = np.where(ibs > 0.95, -1.0, 0.0)
    s['Golden_Vol'] = np.where((sma50 > sma200) & (vol20 < 0.20), 1.5, 0.0)
    s['Vol_Vol_Conviction'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['Gap_Ratio'] = np.where((open_p - close.shift(1)).abs() > (close - open_p).abs(), -0.5, 0.5)
    s['Gap_Fade'] = np.where((open_p - close.shift(1)) / (close.shift(1) + 1e-10) > 0.01, -0.5, 0.2)
    s['MidMonthRev'] = np.where((day >= 18) & (day <= 22), -0.3, 0.1)
    s['AugExit'] = np.where(month == 8, -1.0, 0.1)
    s['Range_Hedge'] = np.where((high - low).rolling(5).mean() > atr20, -0.5, 0.5)
    s['Down_Beta'] = returns.rolling(20).corr(semi_down) 
    s['Tail_Risk'] = returns.rolling(60).quantile(0.05) / (semi_down + 1e-10) 
    s['Asym_Conv'] = semi_down.rolling(20).corr(returns.rolling(20).std()) 
    s['Vol_Conv'] = np.where((vol.pct_change().rolling(20).std() / (vol_std20 + 1e-10)) > 1.2, 1.5, 0.5)
    s['Weekend_Gap'] = np.where((dw_s == 0) & (gap < -0.005), 0.4, -0.1)
    s['MR_EMA_Top'] = np.where(close > close.ewm(span=20).mean() * 1.05, -1.0, 0.0)
    s['B_Donchian'] = np.where(close > high.rolling(50).max().shift(1), 0.5, -0.1)
    s['VIX_Proxy'] = np.where((close / (close.rolling(20).max() + 1e-10) - 1) < -0.05, 0.5, -0.1)
    s['MACD_Vola_Bands'] = np.where(m12 > m12.rolling(20).mean() + 2*m12.rolling(20).std(), -0.5, 0.1)
    s['MACD_Dual_Confirm'] = np.where((m5 > s5) & (m12 > s12), 0.5, -0.2)
    s['MACD_Fractal_D'] = np.where(m12.diff(10).abs() / m12.diff().abs().rolling(10).sum() > 0.6, 0.3, -0.1)
    s['v2'] = np.where((ret60 > 0) & (ibs < 0.2), 1.5, 0.2)
    s['v11'] = np.where(0.5 * vol.rolling(20).mean() * (close.diff(3)**2) > (0.5 * vol.rolling(20).mean() * (close.diff(3)**2)).rolling(100).quantile(0.95), -0.5, 0.2)
    s['macd_slow_hist_pos'] = np.where(h3 > 0, 0.3, -0.3)
    s['vol_skew_2'] = np.where(vol.rolling(60).skew() > 2.0, -0.4, 0.1)
    s['IBS'] = np.where(ibs < 0.2, 0.3, -0.1)
    s['h_accel_50'] = np.where((m12-s9).diff(50) > 0, 1.0, 0.0)
    s['Payday_v2'] = np.where(dom.isin([1, 15]), 0.8, 0.0)
    s['Day15_v2'] = np.where(dom == 15, 1.0, 0.0)
    s['Sept_Mid_v2'] = np.where((month == 9) & (dom >= 15) & (dom <= 25), -1.0, 0.0)
    s['B_Climax'] = np.where(((vol / sma(vol, 50)) * (rng / sma(rng, 50))) > 3.0, -1.0, 0.0)
    s['AV_GARCH_Extreme'] = np.where(rets**2 > (rets**2).rolling(252).mean() * 10, -1.0, 0.0)
    s['v5'] = np.where(((rv5 / rv20) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['v12'] = np.where((range_climax > 3.0) & (rets < 0), 1.5, 0.2)
    s['opt3_mag_vol'] = - ema(rets.abs() / (vol20 + 1e-10), 5)
    s['it1_sig20'] = -(rng / close).rolling(20).std() / (vol20 + 1e-10)
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['c40'] = np.where(uo > 70, -1.0, 0.0)
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['sig41'] = -((rets * vol).rolling(20).mean() / ((rets * vol).rolling(20).std() + 1e-6))
    s['FriDeRisk'] = np.where(df.index.dayofweek == 4, -1.0, 0.0)
    s['CrashProtector'] = np.where((df.index.month == 9) | ((df.index.month == 10) & (df.index.day <= 25)), -0.5, 0.0)
    s['OpExMon'] = np.where((df.index.day >= 15) & (df.index.day <= 21) & (df.index.dayofweek == 0), 1.0, 0.0)
    s['Payday_1_15'] = np.where(df.index.day.isin([1, 15]), 1.0, 0.0)
    s['Third_Week_Exh'] = np.where((df.index.day >= 18) & (df.index.day <= 22), -1.0, 0.0)
    s['s100'] = np.where((open_p / close.shift(1) - 1 > 0.01) & (close < open_p), -1.0, 0.0)
    s['Gene_Recessive_Dip'] = np.where((close < sma50) & (close > sma200), 1.5, 0.0)
    s['SMA200_Slope'] = np.where(sma200.diff(5) > 0, 0.2, -0.2)
    s['BB_Squeeze'] = np.where(((4 * std20) / (sma20 + 1e-10)) < ((4 * std20) / (sma20 + 1e-10)).rolling(100).quantile(0.1), 0.5, 0.0)
    s['HV_Break'] = np.where((close < sma20) & (vol / (sma(vol, 20) + 1e-10) > 1.5), -0.4, 0.0)
    s['S_Iter3'] = np.where((rng > sma(rng, 20)) & (close < sma20), -0.3, 0.0)
    s['S_Iter4'] = np.where((close > close.shift(4)).astype(int).rolling(9).sum() == 9, -0.5, 0.0)
    s['MR_6'] = np.where(get_rsi(close, 20) < 30, 0.2, 0.0)
    s['Orbital_Decay'] = np.where((sma10 < sma50) & (sma10.diff() < 0), -0.2, 0.0)
    s['Payday'] = np.where(pd.Series(df.index.day, index=df.index).isin([1, 15, 30]), 0.5, -0.1)
    s['AV_GARCH_Extreme'] = np.where(rets**2 > (rets**2).rolling(252).mean() * 10, -1.0, 0.0)
    s['v10'] = np.where((close < close.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['MACD_Cross'] = np.where(m12 < s9, -0.5, 0.0)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['Thanksgiving'] = np.where((df.index.month == 11) & (df.index.day >= 20) & (df.index.day <= 28), 0.4, -0.1)
    s['MeanDev10'] = np.where((close - sma10) / (sma10 + 1e-10) < -0.07, 1.5, 0.0)
    s['RangeExpOS'] = np.where((rng > rng.shift(1) * 1.5) & (ibs < 0.05), 1.5, 0.0)
    s['MACD_Zero_Cross'] = np.where(m12 > 0, 0.2, -0.2)
    s['IBS_RSI2_Cross'] = np.where((ibs < 0.3) & (rsi2 > 20) & (rsi2.shift(1) <= 20), 1.1, 0.0)
    s['IBS_RSI14Rev'] = np.where((ibs < 0.2) & (rsi14 > rsi14.shift(1)), 1.1, 0.0)
    s['IBS_DoubleHigh'] = np.where((ibs > 0.8) & (ibs.shift(1) > 0.8), -0.6, 0.0)
    s['IBS_BullSupp'] = np.where((ibs < 0.3) & (close > sma20), 1.0, 0.0)
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['JanEff'] = np.where((m == 1) & (dy <= 15), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((m == 11) & (dy >= 20), 0.3, -0.1)
    s['SpringFever'] = np.where(m == 4, 0.3, -0.1)
    s['SummerSolstice'] = np.where((m == 6) & (dy >= 15) & (dy <= 25), -0.5, 0.1)
    s['WinDressing'] = np.where((pd.Series(m, index=df.index).isin([6,12])) & (dy >= 28), 0.5, -0.1)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_nlr(df):
    df = df.copy()
    c, h, l, v, o = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = c.pct_change()
    
    # --- 1. Technical Indicator Helpers ---
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / (loss + 1e-10))))

    def get_macd(s, fast, slow, signal):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    # Indicator Pre-calculations
    sma10, sma20, sma50, sma200 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    std10, std20, std50 = c.rolling(10).std(), c.rolling(20).std(), c.rolling(50).std()
    ema20, ema100 = c.ewm(span=20).mean(), c.ewm(span=100).mean()
    rsi2, rsi14, rsi20 = get_rsi(c, 2), get_rsi(c, 14), get_rsi(c, 20)
    
    rng = (h - l).replace(0, 1e-6)
    ha5, ha20 = rng.rolling(5).mean(), rng.rolling(20).mean()
    ibs = (c - l) / rng
    atr20 = (pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)).rolling(20).mean()
    
    m12, s12, h12 = get_macd(c, 12, 26, 9)
    m5, s5, h5 = get_macd(c, 5, 13, 1)
    m3_7, s3_7, _ = get_macd(c, 3, 7, 2)
    l24_52, _, h3 = get_macd(c, 24, 52, 18)
    mv, sv, hv = get_macd(v, 5, 15, 5)

    vol20 = rets.rolling(20).std()
    vol_vol_ratio = v.pct_change().rolling(20).std() / (vol20 + 1e-10)
    v_norm = v / v.rolling(50).mean().replace(0, 1e-10)
    pk = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-10)
    
    tp = (h + l + c) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True) + 1e-10)
    
    mfi_pmf = (tp * v).where(tp > tp.shift(1), 0).rolling(14).sum()
    mfi_nmf = (tp * v).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (mfi_pmf / (mfi_nmf + 1e-6))))
    
    uo_bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4*(uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2*(uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    month, day, dow = df.index.month, df.index.day, df.index.dayofweek
    s = pd.DataFrame(index=df.index)

    # --- 2. Signal Generation (144 Unique Signals) ---
    s['s6'] = np.where(pk > pk.rolling(3).mean(), 1.0, -0.5)
    s['i3_c3'] = np.where(ibs.rolling(5).mean() < 0.3, 1.2, 0.0)
    s['L4_C12'] = np.where(v.pct_change() > 0.5, -1.0, 0.5)
    s['FriDeRisk'] = np.where(dow == 4, -0.3, 0.1)
    s['Gap_Regime_2'] = np.where((o / c.shift(1) - 1).abs().rolling(10).sum() > atr20, -0.5, 0.2)
    s['VWAP_Dist'] = np.where(c > ((v * c).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)), 0.2, -0.4)
    emv = (((h+l)/2) - ((h.shift(1)+l.shift(1))/2)) / (v/1e6/(rng+1e-10) + 1e-10)
    s['Body_Exp'] = np.where(abs(c - o) > ha20, 0.3, -0.1)
    s['Kelt_Pos'] = np.where(c > (sma20 + 2*atr20), -0.4, np.where(c < (sma20 - 2*atr20), 0.4, 0.1))
    vm_p, vm_m = (h - l.shift(1)).abs(), (l - h.shift(1)).abs()
    s['Vortex_Trend'] = np.where(vm_p.rolling(14).sum() > vm_m.rolling(14).sum(), 0.4, -0.2)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    mfr_p = (tp * v).where(tp > tp.shift(1), 0).rolling(5).sum()
    mfr_n = (tp * v).where(tp < tp.shift(1), 0).rolling(5).sum()
    s['Range_Hedge'] = np.where(ha5 > atr20, -0.5, 0.5)
    semi_down = rets.clip(upper=0).rolling(20).std()
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['Payday'] = np.where(day.isin([1, 15, 30]), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((month == 11) & (day >= 20), 0.3, -0.1)
    s['WinDressing'] = np.where((month.isin([6,12])) & (day >= 28), 0.5, -0.1)
    s['Weekend_Gap'] = np.where((dow == 0) & ((o / c.shift(1) - 1) < -0.005), 0.4, -0.1)
    s['V_SellerExhaust'] = np.where((ibs < 0.1) & (v > v.rolling(20).mean() * 1.5) & (rsi14 < 25), 0.5, -0.1)
    s['B_Donchian'] = np.where(c > h.rolling(50).max().shift(1), 0.5, -0.1)
    s['S_ThursdayTurn'] = np.where(dow == 3, 0.2, -0.1)
    s['Stdev_Ratio'] = np.where(std10 / (std50 + 1e-10) > 1.2, 0.3, -0.1)
    s['Intra_Vol_Conc'] = np.where(v / (h-l+1e-10) > (v/(h-l+1e-10)).rolling(20).mean(), -0.2, 0.1)
    s['MACD_Dual_Confirm'] = np.where((m5 > s5) & (m12 > s12), 0.5, -0.2)
    s['Up_Down_Ratio'] = np.where(v.where(rets > 0).rolling(10).sum() / v.where(rets < 0).rolling(10).sum() > 1.5, 0.3, -0.1)
    s['macd_slow_hist_pos'] = np.where(h3 > 0, 0.3, -0.3)
    s['Week_3_Bearish'] = np.where((day >= 15) & (day <= 21), -0.3, 0.0)
    s['vol_skew_2'] = np.where(v.rolling(60).skew() > 2.0, -0.4, 0.1)
    s['Pivot_Rev'] = np.where(c < l.shift(1), -0.2, 0.1)
    s['rsi14_low'] = np.where(rsi14 < 30, 1.4, 0.0)
    s['ibs_ma20_low'] = np.where(ibs < ibs.rolling(20).mean(), 1.3, 0.0)
    s['h_accel_50'] = np.where(h12.diff(50) > 0, 1.0, 0.0)
    s['Payday_Mid'] = np.where(day.isin([1, 15]), 0.8, 0.0)
    s['AV4_VWSemiDev_10'] = np.where(rets.where(rets < 0, 0).rolling(20).std().diff() > 0, -0.4, 0.1)
    s['opt3_mag_vol'] = - (rets.abs() / (vol20 + 1e-10)).ewm(span=5).mean()
    s['c40'] = np.where(uo > 70, -1.0, 0.0)
    s['v5'] = np.where(((rets.rolling(5).std() / (vol20 + 1e-10)) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v10'] = np.where((c < c.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['FriDeRisk_Old'] = np.where(dow == 4, -0.3, 0.1)
    gap_v = (o / c.shift(1) - 1).rolling(20).std()
    s['opt1_gap_vol'] = -(gap_v / (gap_v.ewm(span=100).mean() + 1e-10))
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['v4_10'] = (c - o) / (rng + 1e-6)
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['sig41'] = -((rets * v).rolling(20).mean() / ((rets * v).rolling(20).std() + 1e-6))
    s['CrashProtector'] = np.where((month == 9) | ((month == 10) & (day <= 25)), -0.5, 0.0)
    s['OpExMon'] = np.where((day >= 15) & (day <= 21) & (dow == 0), 1.0, 0.0)
    s['Payday_1_15'] = np.where(day.isin([1, 15]), 1.0, 0.0)
    s['Third_Week_Exh'] = np.where((day >= 18) & (day <= 22), -1.0, 0.0)
    s['s100'] = np.where((o / c.shift(1) - 1 > 0.01) & (c < o), -1.0, 0.0)
    s['Gene_Recessive_Dip'] = np.where((c < sma50) & (c > sma200), 1.5, 0.0)
    s['DeathCross'] = np.where(sma50 < sma200, -0.5, 0.0)
    s['S_Iter4'] = np.where((c > c.shift(4)).astype(int).rolling(9).sum() == 9, -0.5, 0.0)
    s['MR_6'] = np.where(rsi20 < 30, 0.2, 0.0)
    s['I1_14'] = np.where((c < c.shift(1)).rolling(5).sum()==5, 0.3, 0.0)
    s['Orbital_Decay'] = np.where((sma10 < sma50) & (sma10.diff() < 0), -0.2, 0.0)
    s['Thanksgiving'] = np.where((month == 11) & (day >= 20) & (day <= 28), 0.4, -0.1)
    s['MeanDev10'] = np.where((c - sma10) / (sma10 + 1e-10) < -0.07, 1.5, 0.0)
    s['RangeExpOS'] = np.where((rng > rng.shift(1) * 1.5) & (ibs < 0.05), 1.5, 0.0)
    s['MACD_RSI_OS'] = np.where((h12 > h12.shift(1)) & (rsi2 < 10), 0.8, 0.0)
    s['MACD_Early_Turn'] = np.where((h12 > 0) & (h12.shift(1) < 0) & (h12.shift(10) < 0), 1.0, 0.0)
    s['MACD_Quick'] = np.where(m3_7 > s3_7, 0.4, -0.3)
    s['IBS_ATR_Panic'] = np.where((ibs < 0.1) & (c.rolling(14).std() > c.rolling(14).std().shift(20) * 1.5), 1.5, 0.0)
    s['IBS_Bounce'] = np.where((ibs.shift(1) < 0.1) & (ibs > 0.3), 1.1, 0.0)
    s['IBS_GapUpFade'] = np.where((ibs > 0.8) & (o > c.shift(1)), -0.4, 0.0)
    s['IBS_RSI14Rev'] = np.where((ibs < 0.2) & (rsi14 > rsi14.shift(1)), 1.1, 0.0)
    s['IBS_DoubleHigh'] = np.where((ibs > 0.8) & (ibs.shift(1) > 0.8), -0.6, 0.0)
    s['SpringFever'] = np.where(month == 4, 0.3, -0.1)

    # --- 3. Final Average ---
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_pgx(df):
    df = df.copy()
    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = close.pct_change()
    
    # Helper functions
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()
    def std(s, n): return s.rolling(n).std()
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # Core Technicals
    sma10, sma20, sma50, sma200 = sma(close, 10), sma(close, 20), sma(close, 50), sma(close, 200)
    vol20 = std(rets, 20)
    rsi14 = get_rsi(close, 14)
    rsi2 = get_rsi(close, 2)
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    
    # MACD Block
    m12 = ema(close, 12) - ema(close, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    macd_h = m12 - s9
    
    m3_7 = ema(close, 3) - ema(close, 7)
    s3_7 = m3_7.ewm(span=2, adjust=False).mean()

    # Volatility / Volume Technicals
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    range_climax = rng / (rng.rolling(50).mean() + 1e-10)
    tp = (high + low + close) / 3
    pmf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    nmf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (pmf / (nmf + 1e-6))))
    
    uo_bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([rng, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4 * (uo_bp.rolling(7).sum()/(tr.rolling(7).sum()+1e-6)) + 
                2 * (uo_bp.rolling(14).sum()/(tr.rolling(14).sum()+1e-6)) + 
                (uo_bp.rolling(28).sum()/(tr.rolling(28).sum()+1e-6))) / 7
    
    rv5, rv20 = std(rets, 5) * 15.87, vol20 * 15.87
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (vol20 + 1e-10)
    vm_p, vm_m = (high - low.shift(1)).abs(), (low - high.shift(1)).abs()
    vi_p, vi_m = vm_p.rolling(14).sum() / tr.rolling(14).sum(), vm_m.rolling(14).sum() / tr.rolling(14).sum()
    semi_down = rets.clip(upper=0).rolling(20).std()
    vwap20 = (vol * close).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)
    atr20 = tr.rolling(20).mean()
    vsma20 = vol.rolling(20).mean()
    vol_proxy = rng.rolling(14).std()

    # Dates
    day, dow, month = df.index.day, df.index.dayofweek, df.index.month

    s = pd.DataFrame(index=df.index)

    # Cleaned Signals
    s['MACD_Hist_Deccel'] = np.where(macd_h.diff() < macd_h.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross_Down'] = np.where(m12 < s9, -0.5, 0.0)
    s['Liquidity_Regime'] = np.where((vol / (atr20 + 1e-10)) < (vol / (atr20 + 1e-10)).rolling(50).mean(), -0.3, 0.1)
    s['VWAP_Dist'] = np.where(close > vwap20, 0.2, -0.4)
    s['Vortex_Trend'] = np.where(vi_p > vi_m, 0.4, -0.2)
    s['VolATR'] = np.where((vol / (vsma20 + 1e-10) > 1.5) & (rets < 0), 1.0, 0.0)
    s['IBS_Ultra_S'] = np.where(ibs > 0.99, -1.0, 0.0)
    s['High_IBS_Fade'] = np.where(ibs > 0.95, -1.0, 0.0)
    s['Golden_Vol'] = np.where((sma50 > sma200) & (vol20 * np.sqrt(252) < 0.20), 1.5, 0.0)
    s['Vol_Vol_Conviction'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['Down_Beta'] = rets.rolling(20).corr(semi_down)
    s['S_ThursdayTurn'] = np.where(dow == 3, 0.2, -0.1)
    s['ZScore_Price'] = np.where(((close - sma20) / (std(close, 20) + 1e-10)) > 1.0, 0.4, 
                                 np.where(((close - sma20) / (std(close, 20) + 1e-10)) < -1.0, -0.4, 0))
    s['Pivot_Rev'] = np.where(close < low.shift(1), -0.2, 0.1)
    s['h_accel_50'] = np.where(macd_h.diff(50) > 0, 1.0, 0.0)
    s['v5'] = np.where(((rv5 / (rv20 + 1e-10)) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['FriDeRisk_Old'] = np.where(dow == 4, -0.3, 0.1)
    s['it1_sig20'] = -(rng / (close + 1e-10)).rolling(20).std() / (vol20 + 1e-10)
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['v4_10'] = (close - open_p) / (rng + 1e-6)
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['OpExMon'] = np.where((day >= 15) & (day <= 21) & (dow == 0), 1.0, 0.0)
    s['SMA200_Slope'] = np.where(sma200.diff(5) > 0, 0.2, -0.2)
    s['Orbital_Decay'] = np.where((sma10 < sma50) & (sma10.diff() < 0), -0.2, 0.0)
    s['v10'] = np.where((close < close.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['MeanDev10'] = np.where((close - sma10) / (sma10 + 1e-10) < -0.07, 1.5, 0.0)
    s['MACD_Early_Turn'] = np.where((macd_h > 0) & (macd_h.shift(1) < 0) & (macd_h.shift(10) < 0), 1.0, 0.0)
    s['MACD_Quick'] = np.where(m3_7 > s3_7, 0.4, -0.3)
    s['IBS_ATR_Panic'] = np.where((ibs < 0.1) & (std(close, 14) > std(close, 14).shift(20) * 1.5), 1.5, 0.0)
    s['IBS_BullSupp'] = np.where((ibs < 0.3) & (close > sma20), 1.0, 0.0)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['ADX_Proxy'] = np.where((m12 > s9) & (vol_proxy > vol_proxy.rolling(20).mean()), 0.5, -0.2)
    s['Payday'] = np.where(pd.Series(day, index=df.index).isin([1, 15, 30]), 0.5, -0.1)
    s['FriDeRisk'] = np.where(dow == 4, -0.3, 0.1)
    s['Holiday_Frontrun'] = np.where((month == 11) & (day >= 20), 0.3, -0.1)
    s['SpringFever'] = np.where(month == 4, 0.3, -0.1)
    s['WinDressing'] = np.where(pd.Series(month, index=df.index).isin([6,12]) & (day >= 28), 0.5, -0.1)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_pid(df):
    df = df.copy()
    c, h, l, v, o = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = c.pct_change()
    
    # --- 1. Technical Indicator Helpers ---
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / (loss + 1e-10))))

    def get_macd(s, fast, slow, signal):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    # Indicator Pre-calculations
    sma10, sma20, sma50, sma200 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    std10, std20, std50 = c.rolling(10).std(), c.rolling(20).std(), c.rolling(50).std()
    ema20, ema100 = c.ewm(span=20).mean(), c.ewm(span=100).mean()
    rsi2, rsi14, rsi20 = get_rsi(c, 2), get_rsi(c, 14), get_rsi(c, 20)
    
    rng = (h - l).replace(0, 1e-6)
    ha5, ha20 = rng.rolling(5).mean(), rng.rolling(20).mean()
    ibs = (c - l) / rng
    atr20 = (pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)).rolling(20).mean()
    
    m12, s12, h12 = get_macd(c, 12, 26, 9)
    m5, s5, h5 = get_macd(c, 5, 13, 1)
    m3_7, s3_7, _ = get_macd(c, 3, 7, 2)
    l24_52, _, h3 = get_macd(c, 24, 52, 18)
    mv, sv, hv = get_macd(v, 5, 15, 5)

    vol20 = rets.rolling(20).std()
    vol_vol_ratio = v.pct_change().rolling(20).std() / (vol20 + 1e-10)
    v_norm = v / v.rolling(50).mean().replace(0, 1e-10)
    pk = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-10)
    
    tp = (h + l + c) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True) + 1e-10)
    
    mfi_pmf = (tp * v).where(tp > tp.shift(1), 0).rolling(14).sum()
    mfi_nmf = (tp * v).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (mfi_pmf / (mfi_nmf + 1e-6))))
    
    uo_bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4*(uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2*(uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    month, day, dow = df.index.month, df.index.day, df.index.dayofweek
    s = pd.DataFrame(index=df.index)

    # --- 2. Signal Generation (144 Unique Signals) ---
    s['s4'] = np.where(ibs < 0.2, 1.5, np.where(ibs > 0.8, -1.0, 0.5))
    s['s6'] = np.where(pk > pk.rolling(3).mean(), 1.0, -0.5)
    s['s7'] = np.where((day >= 10) & (day <= 15), 1.0, 0.1)
    s['c6'] = np.where(c > c.shift(20), 1.0, -1.0)
    s['i3_c3'] = np.where(ibs.rolling(5).mean() < 0.3, 1.2, 0.0)
    s['i5_c12'] = np.where(rsi14 < 20, 1.5, -0.5)
    s['Low_Vol_Pullback'] = np.where((rets < 0) & (v / v.rolling(20).mean() < 0.7), 0.8, 0.0)
    s['Quiet_Accum'] = np.where((v / v.rolling(20).mean() < 0.8) & (c > c.shift(1)), 0.4, 0.0)
    s['Vol_Dry_Up'] = np.where(v < v.rolling(50).min().shift(1), -0.3, 0.0)
    s['FriDeRisk'] = np.where(dow == 4, -0.3, 0.1)
    s['Gap_Regime_2'] = np.where((o / c.shift(1) - 1).abs().rolling(10).sum() > atr20, -0.5, 0.2)
    emv = (((h+l)/2) - ((h.shift(1)+l.shift(1))/2)) / (v/1e6/(rng+1e-10) + 1e-10)
    vm_p, vm_m = (h - l.shift(1)).abs(), (l - h.shift(1)).abs()
    s['Vortex_Trend'] = np.where(vm_p.rolling(14).sum() > vm_m.rolling(14).sum(), 0.4, -0.2)
    s['Resistance_Prox'] = np.where((c > h.rolling(60).max() * 0.995) & (v < v.rolling(20).mean()), -0.5, 0.0)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['Bear_Range_Expansion'] = np.where(rng > ha20 * 1.5, -0.3, 0.0)
    s['VolATR'] = np.where((v / v.rolling(20).mean() > 1.5) & (rets < 0), 1.0, 0.0)
    s['Gap_Ratio'] = np.where((o - c.shift(1)).abs() > (c - o).abs(), -0.5, 0.5)
    mfr_p = (tp * v).where(tp > tp.shift(1), 0).rolling(5).sum()
    mfr_n = (tp * v).where(tp < tp.shift(1), 0).rolling(5).sum()
    s['MidMonthRev'] = np.where((day >= 18) & (day <= 22), -0.3, 0.1)
    s['AugExit'] = np.where(month == 8, -1.0, 0.1)
    s['Range_Hedge'] = np.where(ha5 > atr20, -0.5, 0.5)
    semi_down = rets.clip(upper=0).rolling(20).std()
    s['Tail_Risk'] = rets.rolling(60).quantile(0.05) / (semi_down + 1e-10)
    s['Asym_Conv'] = semi_down.rolling(20).corr(rets.rolling(20).std())
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['WinDressing'] = np.where((month.isin([6,12])) & (day >= 28), 0.5, -0.1)
    s['MR_EMA_Top'] = np.where(c > ema20 * 1.05, -1.0, 0.0)
    s['v11'] = np.where(0.5 * v.rolling(20).mean() * (c.diff(3)**2) > (0.5 * v.rolling(20).mean() * (c.diff(3)**2)).rolling(100).quantile(0.95), -0.5, 0.2)
    s['ibs_vol_norm_revert'] = np.where((ibs < 0.2) & (v_norm > 1.5), 1.2, 0.0)
    s['Week_3_Bearish'] = np.where((day >= 15) & (day <= 21), -0.3, 0.0)
    s['SummerSolstice'] = np.where((month == 6) & (day >= 15) & (day <= 25), -0.5, 0.1)
    s['Pivot_Rev'] = np.where(c < l.shift(1), -0.2, 0.1)
    s['Inside_Range_Gap'] = np.where((o < h.shift(1)) & (o > l.shift(1)), 0.2, -0.1)
    s['rsi14_low'] = np.where(rsi14 < 30, 1.4, 0.0)
    s['m_cross_ext'] = np.where((m12 < -1) & (m12 > s12), 1.5, 0.0)
    s['h_accel_50'] = np.where(h12.diff(50) > 0, 1.0, 0.0)
    s['Payday_Mid'] = np.where(day.isin([1, 15]), 0.8, 0.0)
    s['Day15_Only'] = np.where(day == 15, 1.0, 0.0)
    s['Sept_Mid_v2'] = np.where((month == 9) & (day >= 15) & (day <= 25), -1.0, 0.0)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['it1_sig20'] = -(rng / c).rolling(20).std() / (vol20 + 1e-10)
    s['c40'] = np.where(uo > 70, -1.0, 0.0)
    s['v5'] = np.where(((rets.rolling(5).std() / (vol20 + 1e-10)) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v10'] = np.where((c < c.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['v12'] = np.where(((rng / rng.rolling(50).mean()) > 3.0) & (rets < 0), 1.5, 0.2)
    s['FriDeRisk_Old'] = np.where(dow == 4, -0.3, 0.1)
    gap_v = (o / c.shift(1) - 1).rolling(20).std()
    s['opt1_gap_vol'] = -(gap_v / (gap_v.ewm(span=100).mean() + 1e-10))
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['mr51'] = (c.ewm(span=60).mean() - c) / c
    s['v4_10'] = (c - o) / (rng + 1e-6)
    s['sig41'] = -((rets * v).rolling(20).mean() / ((rets * v).rolling(20).std() + 1e-6))
    s['CrashProtector'] = np.where((month == 9) | ((month == 10) & (day <= 25)), -0.5, 0.0)
    s['OpExMon'] = np.where((day >= 15) & (day <= 21) & (dow == 0), 1.0, 0.0)
    s['Third_Week_Exh'] = np.where((day >= 18) & (day <= 22), -1.0, 0.0)
    s['s100'] = np.where((o / c.shift(1) - 1 > 0.01) & (c < o), -1.0, 0.0)
    s['Gene_Recessive_Dip'] = np.where((c < sma50) & (c > sma200), 1.5, 0.0)
    s['HV_Break'] = np.where((c < sma20) & (v / v.rolling(20).mean() > 1.5), -0.4, 0.0)
    s['S_Iter4'] = np.where((c > c.shift(4)).astype(int).rolling(9).sum() == 9, -0.5, 0.0)
    s['Orbital_Decay'] = np.where((sma10 < sma50) & (sma10.diff() < 0), -0.2, 0.0)
    s['Thanksgiving'] = np.where((month == 11) & (day >= 20) & (day <= 28), 0.4, -0.1)
    s['RangeExpOS'] = np.where((rng > rng.shift(1) * 1.5) & (ibs < 0.05), 1.5, 0.0)
    s['MACD_RSI_OS'] = np.where((h12 > h12.shift(1)) & (rsi2 < 10), 0.8, 0.0)
    s['MACD_Quick'] = np.where(m3_7 > s3_7, 0.4, -0.3)
    s['IBS_Return_Pos'] = np.where((ibs < 0.2) & (rets > 0), 1.2, 0.0)
    s['IBS_Bounce'] = np.where((ibs.shift(1) < 0.1) & (ibs > 0.3), 1.1, 0.0)
    s['IBS_RSI2_Cross'] = np.where((ibs < 0.3) & (rsi2 > 20) & (rsi2.shift(1) <= 20), 1.1, 0.0)
    s['IBS_GapUpFade'] = np.where((ibs > 0.8) & (o > c.shift(1)), -0.4, 0.0)
    s['IBS_RSI14Rev'] = np.where((ibs < 0.2) & (rsi14 > rsi14.shift(1)), 1.1, 0.0)
    s['IBS_QuietFear'] = np.where((ibs < 0.1) & (rets.abs() < 0.005), 1.1, 0.0)
    s['IBS_BullSupp'] = np.where((ibs < 0.3) & (c > sma20), 1.0, 0.0)
    s['MACD_EMA_Filt'] = np.where((c > sma50) & (h12 > 0), 0.5, -0.2)
    s['st_trend_vol_confirm'] = np.where(v > v.shift(1), 0.2, -0.2)
    s['vol_roc_confirm'] = np.where(v.pct_change() > 0.1, 0.1, -0.1)
    s['st4_17'] = np.where(v.rolling(5).mean() > v.rolling(20).mean(), 0.5, -0.5)

    # --- 3. Final Average ---
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_qqq(df):
    df = df.copy()
    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = close.pct_change().fillna(0)
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()
    def std(s, n): return s.rolling(n).std()
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    sma10, sma20, sma50, sma200 = sma(close, 10), sma(close, 20), sma(close, 50), sma(close, 200)
    std20, rsi14, rsi2 = std(close, 20), get_rsi(close, 14), get_rsi(close, 2)
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    m12 = ema(close, 12) - ema(close, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    macd_h = m12 - s9
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    tp = (high + low + close) / 3
    sma_tp = sma(tp, 20)
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    s = pd.DataFrame(index=df.index)
    
    # --- Ensemble Signals ---
    s['IBS_Dip'] = np.where(ibs < 0.2, 1.0, 0.0)
    s['Gene_Recessive_Dip'] = np.where((close < sma50) & (close > sma200), 1.5, 0.0)
    s['DeathCross'] = np.where(sma50 < sma200, -0.5, 0.0)
    s['SMA200_Slope'] = np.where(sma200.diff(5) > 0, 0.2, -0.2)
    s['BB_Squeeze'] = np.where(((4 * std20) / (sma20 + 1e-10)) < ((4 * std20) / (sma20 + 1e-10)).rolling(100).quantile(0.1), 0.5, 0.0)
    s['HV_Break'] = np.where((close < sma20) & (vol / (sma(vol, 20) + 1e-10) > 1.5), -0.4, 0.0)
    s['S_Iter3'] = np.where((rng > sma(rng, 20)) & (close < sma20), -0.3, 0.0)
    s['S_Iter4'] = np.where((close > close.shift(4)).astype(int).rolling(9).sum() == 9, -0.5, 0.0)
    s['MR_8'] = np.where(cci < -150, 0.3, 0.0)
    s['MR_6'] = np.where(get_rsi(close, 20) < 30, 0.2, 0.0)
    s['I1_14'] = np.where((close < close.shift(1)).rolling(5).sum()==5, 0.3, 0.0)
    s['Orbital_Decay'] = np.where((sma10 < sma50) & (sma10.diff() < 0), -0.2, 0.0)
    s['Payday'] = np.where(pd.Series(df.index.day, index=df.index).isin([1, 15, 30]), 0.5, -0.1)
    s['AV_GARCH_Extreme'] = np.where(rets**2 > (rets**2).rolling(252).mean() * 10, -1.0, 0.0)
    s['v10'] = np.where((close < close.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['Vol_Conv'] = np.where(vol.pct_change().rolling(20).std() / (rets.rolling(20).std() + 1e-10) > 1.2, 1.5, 0.5)
    s['MACD_Cross'] = np.where(m12 < s9, -0.5, 0.0)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['LoVol_PB'] = np.where((m12 < s9) & (vol < sma(vol, 20)), 0.4, -0.2)
    s['Thanksgiving'] = np.where((df.index.month == 11) & (df.index.day >= 20) & (df.index.day <= 28), 0.4, -0.1)
    s['ibs_mean_revert_wide'] = np.where((ibs < 0.2) & ((close - sma20)/sma20 < -0.05), 1.5, 0.0)
    s['MeanDev10'] = np.where((close - sma10) / (sma10 + 1e-10) < -0.07, 1.5, 0.0)
    s['RangeExpOS'] = np.where((rng > rng.shift(1) * 1.5) & (ibs < 0.05), 1.5, 0.0)
    s['MACD_Zero_Cross'] = np.where(m12 > 0, 0.2, -0.2)
    s['MACD_RSI_OS'] = np.where((macd_h > macd_h.shift(1)) & (rsi2 < 10), 0.8, 0.0)
    s['MACD_Early_Turn'] = np.where((macd_h > 0) & (macd_h.shift(1) < 0) & (macd_h.shift(10) < 0), 1.0, 0.0)
    m3_7 = ema(close, 3) - ema(close, 7)
    s3_7 = m3_7.ewm(span=2, adjust=False).mean()
    s['MACD_Quick'] = np.where(m3_7 > s3_7, 0.4, -0.3)
    s['IBS_Mean_3'] = np.where(ibs.rolling(3).mean() < 0.2, 1.3, 0.0)
    s['IBS_Return_Pos'] = np.where((ibs < 0.2) & (rets > 0), 1.2, 0.0)
    s['IBS_ATR_Panic'] = np.where((ibs < 0.1) & (std(close, 14) > std(close, 14).shift(20) * 1.5), 1.5, 0.0)
    s['IBS_Bounce'] = np.where((ibs.shift(1) < 0.1) & (ibs > 0.3), 1.1, 0.0)
    s['IBS_RSI2_Cross'] = np.where((ibs < 0.3) & (rsi2 > 20) & (rsi2.shift(1) <= 20), 1.1, 0.0)
    s['IBS_GapUpFade'] = np.where((ibs > 0.8) & (open_p > close.shift(1)), -0.4, 0.0)
    s['IBS_RSI14Rev'] = np.where((ibs < 0.2) & (rsi14 > rsi14.shift(1)), 1.1, 0.0)
    s['IBS_QuietFear'] = np.where((ibs < 0.1) & (rets.abs() < 0.005), 1.1, 0.0)
    s['IBS_DoubleHigh'] = np.where((ibs > 0.8) & (ibs.shift(1) > 0.8), -0.6, 0.0)
    s['IBS_BullSupp'] = np.where((ibs < 0.3) & (close > sma20), 1.0, 0.0)
    s['MACD_EMA_Filt'] = np.where((close > sma50) & (macd_h > 0), 0.5, -0.2)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_rpv(df_input):
    df = df_input.copy()
    close, vol, high, low = df['Close'], df['Volume'], df['High'], df['Low']
    rets = close.pct_change()
    m, dy, dw = df.index.month, df.index.day, df.index.dayofweek

    def calculate_macd(series, fast, slow, signal_span):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        return macd_line, signal_line
    
    # Pre-computations
    ha20, ha5 = (high - low).rolling(20).mean(), (high - low).rolling(5).mean()
    m12, s12 = calculate_macd(close, 12, 26, 9)
    h12 = m12 - s12
    m50, s50 = calculate_macd(close, 50, 200, 20)
    e20, e50 = close.ewm(span=20).mean(), close.ewm(span=50).mean()
    vol_avg20 = vol.rolling(20).mean()
    vol_std20 = rets.rolling(20).std()
    
    s = pd.DataFrame(index=df.index)
    
    # Volatility and Momentum
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (vol_std20 + 1e-10)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    vol_proxy = (high - low).rolling(14).std()
    s['ADX_Proxy'] = np.where((m12 > s12) & (vol_proxy > vol_proxy.rolling(20).mean()), 0.5, -0.2)
    s['LoVol_PB'] = np.where((m12 < s12) & (vol < vol_avg20), 0.4, -0.2)

    # Seasonality
    s['Payday'] = np.where(pd.Series(dy, index=df.index).isin([1, 15, 30]), 0.5, -0.1)
    s['FriDeRisk'] = np.where(dw == 4, -0.3, 0.1)
    s['JanEff'] = np.where((m == 1) & (dy <= 15), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((m == 11) & (dy >= 20), 0.3, -0.1)
    s['SpringFever'] = np.where(m == 4, 0.3, -0.1)
    s['SummerSolstice'] = np.where((m == 6) & (dy >= 15) & (dy <= 25), -0.5, 0.1)
    s['WinDressing'] = np.where((pd.Series(m, index=df.index).isin([6,12])) & (dy >= 28), 0.5, -0.1)
    s['Thanksgiving'] = np.where((m == 11) & (dy >= 20) & (dy <= 28), 0.4, -0.1)

    # Aggregate and Shift(1) to apply signal from T at T+1
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_rwj(df_input):
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    rets = close.pct_change()
    m, dy, dw = df.index.month, df.index.day, df.index.dayofweek
    dy_s, dw_s, m_s = pd.Series(dy, index=df.index), pd.Series(dw, index=df.index), pd.Series(m, index=df.index)
    
    # Gap calculation
    gap = (df['Open'] / df['Close'].shift(1)) - 1
    
    # Pre-computations
    def ema(series, n): return series.ewm(span=n, adjust=False).mean()
    m12 = ema(close, 12) - ema(close, 26)
    s12 = m12.ewm(span=9, adjust=False).mean()
    h12 = m12 - s12
    vol_std20 = rets.rolling(20).std()
    ma20_p = close.rolling(20).mean()
    std20_p = close.rolling(20).std()
    ha20 = (high - low).rolling(20).mean()
    ha5 = (high - low).rolling(5).mean()
    vol_avg20 = vol.rolling(20).mean()
    vol_proxy = (high - low).rolling(14).std()
    
    s = pd.DataFrame(index=df.index)
    
    # --- Base Signals ---
    vol_vol_ratio = vol.pct_change().rolling(20).std() / (vol_std20 + 1e-10)
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['ADX_Proxy'] = np.where((m12 > s12) & (vol_proxy > vol_proxy.rolling(20).mean()), 0.5, -0.2)
    s['LoVol_PB'] = np.where((m12 < s12) & (vol < vol_avg20), 0.4, -0.2)
    s['Payday'] = np.where(dy_s.isin([1, 15, 30]), 0.5, -0.1)
    s['FriDeRisk'] = np.where(dw_s == 4, -0.3, 0.1)
    s['JanEff'] = np.where((m_s == 1) & (dy_s <= 15), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((m_s == 11) & (dy_s >= 20), 0.3, -0.1)
    s['SummerSolstice'] = np.where((m_s == 6) & (dy_s >= 15) & (dy_s <= 25), -0.5, 0.1)
    s['WinDressing'] = np.where((m_s.isin([6,12])) & (dy_s >= 28), 0.5, -0.1)
    s['Thanksgiving'] = np.where((m_s == 11) & (dy_s >= 20) & (dy_s <= 28), 0.4, -0.1)
    s['Overnight'] = np.where((df['Open'] - close.shift(1)) / (close.shift(1) + 1e-10) > 0.005, -0.3, 0.1)
    s['Pivot_Rev'] = np.where(close < low.shift(1), -0.2, 0.1)
    s['BB_MR'] = np.where(close < ma20_p - 2*std20_p, 0.5, -0.1)
    s['IBS'] = np.where((close - low) / (high - low + 1e-10) < 0.2, 0.3, -0.1)
    s['PR_Ratio'] = np.where((close - low) / (high - low + 1e-10) < 0.1, 0.4, -0.1)
    s['Vol_ROC_20'] = np.where(vol.pct_change(20) > 0.5, -0.2, 0.1)
    s['Weekend_Gap'] = np.where((dw_s == 0) & (gap < -0.005), 0.4, -0.1)
    s['Inside_Range_Gap'] = np.where((df['Open'] < high.shift(1)) & (df['Open'] > low.shift(1)), 0.2, -0.1)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_rwo(df):
    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = df['Close'].pct_change().fillna(0)
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def std(s, n): return s.rolling(n).std()
    def sma(s, n): return s.rolling(n).mean()

    # Pre-computations
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    vol20 = rets.rolling(20).std()
    vol60 = rets.rolling(60).std()
    vol252 = rets.rolling(252).std()
    m12 = ema(close, 12) - ema(close, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    range_climax = rng / rng.rolling(50).mean().replace(0, np.nan)
    tp = (high + low + close) / 3
    mfi_raw = tp * vol
    mfi = 100 - (100 / (1 + (mfi_raw.where(tp > tp.shift(1), 0).rolling(14).sum() / 
                            mfi_raw.where(tp < tp.shift(1), 0).rolling(14).sum().replace(0, 1e-6))))
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    dom, dow, month = df.index.day, df.index.dayofweek, df.index.month

    v_norm = vol / sma(vol, 50)
    std20 = std(rets, 20)
    std100 = std(rets, 100)
    ha20 = rng.rolling(20).mean()
    ha5 = rng.rolling(5).mean()
    
    delta = close.diff()
    up = delta.where(delta > 0, 0)
    down = delta.where(delta < 0, 0).abs()
    rsi14 = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean().replace(0, 1e-6)))
    cmf = (((close - low) - (high - close)) / rng * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, 1e-6)
    
    uo_bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4 * (uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2 * (uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    l24_52 = ema(close, 24) - ema(close, 52)
    h3 = l24_52 - ema(l24_52, 18)
    
    rv5, rv20 = std(rets, 5) * 15.87, std(rets, 20) * 15.87
    gap = open_p / close.shift(1) - 1

    s = pd.DataFrame(index=df.index)

    # Base Signals
    s['macd_slow_hist_pos'] = np.where(h3 > 0, 0.3, -0.3)
    s['ibs_vol_norm_revert'] = np.where((ibs < 0.2) & (vol/sma(vol, 50) > 1.5), 1.2, 0.0)
    s['Week_3_Bearish'] = np.where((dom >= 15) & (dom <= 21), -0.3, 0.0)
    s['vol_skew_2'] = np.where(vol.rolling(60).skew() > 2.0, -0.4, 0.1)
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['Payday'] = np.where(dom.isin([1, 15, 30]), 0.5, -0.1)
    s['FriDeRisk'] = np.where(dow == 4, -0.3, 0.1)
    s['SummerSolstice'] = np.where((month == 6) & (dom >= 15) & (dom <= 25), -0.5, 0.1)
    s['Pivot_Rev'] = np.where(close < low.shift(1), -0.2, 0.1)
    s['IBS'] = np.where(ibs < 0.2, 0.3, -0.1)
    s['Inside_Range_Gap'] = np.where((open_p < high.shift(1)) & (open_p > low.shift(1)), 0.2, -0.1)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['rsi14_low'] = np.where(rsi14 < 30, 1.4, 0.0)
    s['ibs_ma20_low'] = np.where(ibs < ibs.rolling(20).mean(), 1.3, 0.0)
    s['m_cross_ext'] = np.where((m12 < -1) & (m12 > s9), 1.5, 0.0)
    s['h_accel_50'] = np.where((m12-s9).diff(50) > 0, 1.0, 0.0)
    s['Payday_v2'] = np.where(dom.isin([1, 15]), 0.8, 0.0)
    s['Day15_v2'] = np.where(dom == 15, 1.0, 0.0)
    s['Sept_Mid_v2'] = np.where((month == 9) & (dom >= 15) & (dom <= 25), -1.0, 0.0)
    s['B_Climax'] = np.where(((vol / sma(vol, 50)) * (rng / sma(rng, 50))) > 3.0, -1.0, 0.0)
    s['AV_GARCH_Extreme'] = np.where(rets**2 > (rets**2).rolling(252).mean() * 10, -1.0, 0.0)
    s['AV4_VWSemiDev_10'] = np.where(std(rets.where(rets * vol / sma(vol, 20) < 0, 0), 20).diff() > 0, -0.4, 0.1)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['opt3_mag_vol'] = - ema(rets.abs() / (vol20 + 1e-10), 5)
    s['it1_sig20'] = -(rng / close).rolling(20).std() / (vol20 + 1e-10)
    s['c40'] = np.where(uo > 70, -1.0, 0.0)
    s['CrashProtector'] = np.where((month == 9) | ((month == 10) & (dom <= 25)), -0.5, 0.0)
    s['Third_Week_Exh'] = np.where((dom >= 18) & (dom <= 22), -1.0, 0.0)

    # Added signals from 5 iterations
    neg_rets = rets.copy(); neg_rets[neg_rets > 0] = 0
    s['downside_vol_ratio'] = np.where(std(neg_rets, 20) / (vol20 + 1e-6) > 0.6, -0.5, 0.1)
    atr = (high - low).rolling(14).mean()
    s['vol_of_atr'] = np.where(std(atr, 20) / (atr + 1e-6) > 0.2, -0.4, 0.1)

    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_vaw(df):
    """
    Final optimized systematic strategy for VAW.
    Exposure: -1.0 to 1.5. 
    Timing: Close-to-Close.
    """
    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = close.pct_change().fillna(0)
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def std(s, n): return s.rolling(n).std()
    def sma(s, n): return s.rolling(n).mean()

    # --- Pre-computations ---
    vol20 = rets.rolling(20).std()
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    rv5, rv20 = rets.rolling(5).std() * 15.87, vol20 * 15.87
    tp = (high + low + close) / 3
    pmf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    nmf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (pmf / (nmf + 1e-6))))
    range_climax = rng / rng.rolling(50).mean().replace(0, np.nan)
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    
    uo_bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4 * (uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2 * (uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    delta = close.diff()
    up = delta.where(delta > 0, 0); down = delta.where(delta < 0, 0).abs()
    rsi14 = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean().replace(0, 1e-6)))

    s = pd.DataFrame(index=df.index)
    
    # Base Signals
    s['v5'] = np.where(((rv5 / rv20) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['v10'] = np.where((close < close.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['v12'] = np.where((range_climax > 3.0) & (rets < 0), 1.5, 0.2)
    s['FriDeRisk_Old'] = np.where(df.index.dayofweek == 4, -0.3, 0.1)
    s['Vol_Conv'] = np.where(vol.pct_change().rolling(20).std() / (vol20 + 1e-10) > 1.2, 1.5, 0.5)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['opt3_mag_vol'] = - ema(rets.abs() / (vol20 + 1e-10), 5)
    s['it1_sig20'] = -(rng / close).rolling(20).std() / (vol20 + 1e-10)
    s['skew_mom_rev'] = -rets.rolling(100).skew().diff()
    s['mr51'] = (ema(close, 60) - close) / close
    s['v4_10'] = (close - open_p) / (rng + 1e-6)
    s['c40'] = np.where(uo > 70, -1.0, 0.0)
    s['c67'] = np.where(rsi14 > 80, -1.0, 0.0)
    s['sig41'] = -((rets * vol).rolling(20).mean() / ((rets * vol).rolling(20).std() + 1e-6))
    
    s['FriDeRisk'] = np.where(df.index.dayofweek == 4, -1.0, 0.0)
    s['CrashProtector'] = np.where((df.index.month == 9) | ((df.index.month == 10) & (df.index.day <= 25)), -0.5, 0.0)
    s['OpExMon'] = np.where((df.index.day >= 15) & (df.index.day <= 21) & (df.index.dayofweek == 0), 1.0, 0.0)
    s['Payday_1_15'] = np.where(df.index.day.isin([1, 15]), 1.0, 0.0)
    s['Third_Week_Exh'] = np.where((df.index.day >= 18) & (df.index.day <= 22), -1.0, 0.0)
    
    s['s100'] = np.where((open_p / close.shift(1) - 1 > 0.01) & (close < open_p), -1.0, 0.0)

    # Final Exposure
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_vdc(df):
    """
    Consumer Staples Adaptive Reversion (CSAR) - Final Optimized Portfolio.
    Ensemble of 35 signals covering Reversion, Seasonality, MACD, and Asymmetric Volatility.
    Constraints: -1.0 to 1.5. No lookahead bias.
    """
    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = df['Close'].pct_change().fillna(0)
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def std(s, n): return s.rolling(n).std()
    def sma(s, n): return s.rolling(n).mean()

    # Pre-computations
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    ret60 = close.pct_change(60)
    rv5, rv20 = rets.rolling(5).std() * 15.87, rets.rolling(20).std() * 15.87
    range_climax = rng / rng.rolling(50).mean().replace(0, np.nan)
    m12 = ema(close, 12) - ema(close, 26)
    s9 = m12.ewm(span=9, adjust=False).mean()
    delta = close.diff(); up = delta.where(delta > 0, 0); down = delta.where(delta < 0, 0).abs()
    rsi14 = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean().replace(0, 1e-6)))
    cmf = (((close - low) - (high - close)) / rng * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, 1e-6)
    s2_pivot = ((high + low + close) / 3) - (high - low)
    dom, dow, month = df.index.day, df.index.dayofweek, df.index.month

    # Asymmetric Vol Pre-computations
    gap = open_p / close.shift(1) - 1
    dn_rets = rets.where(rets < 0, 0)
    up_std = std(rets.where(rets > 0, 0), 20)
    v20 = std(rets, 20)
    vw_rets = rets * vol / sma(vol, 20)
    dn_vw = vw_rets.where(vw_rets < 0, 0)

    s = pd.DataFrame(index=df.index)
    
    # --- CORE REVERSION & MOMENTUM ---
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['v12'] = np.where((range_climax > 3.0) & (rets < 0), 1.5, 0.2)
    s['rsi14_low'] = np.where(rsi14 < 30, 1.4, 0.0)
    s['cmf_rev'] = np.where(cmf < -0.2, 1.4, 0.0)
    s['ibs_ma20_low'] = np.where(ibs < ibs.rolling(20).mean(), 1.3, 0.0)
    s['m_cross_ext'] = np.where((m12 < -1) & (m12 > s9), 1.5, 0.0)
    s['m_peak_revert'] = np.where(m12 > m12.rolling(10).max().shift(1), -0.5, 0.0)
    s['h_accel_50'] = np.where((m12-s9).diff(50) > 0, 1.0, 0.0)
    
    # --- SEASONALITY ---
    s['Payday'] = np.where(dom.isin([1, 15]), 0.8, 0.0)
    s['Day15_v2'] = np.where(dom == 15, 1.0, 0.0)
    s['OpEx_Mon_v2'] = np.where((dow == 0) & (dom >= 11) & (dom <= 17), 0.8, 0.0)
    s['Sept_Mid_v2'] = np.where((month == 9) & (dom >= 15) & (dom <= 25), -1.0, 0.0)
    
    # --- BREADTH & SKEW ---
    s['B_Climax'] = np.where(((vol / vol.rolling(50).mean()) * (rng / rng.rolling(50).mean())) > 3.0, -1.0, 0.0)
    s['AV3_11'] = np.where((rng / (rets.rolling(20).skew().abs() + 1e-6)) > 0.02, -0.5, 0.0)
    s['AV4_5'] = np.where(rng.rolling(100).skew() > 1.0, -0.5, 0.0)
    s['AV5_5'] = np.where(((high - open_p) / open_p).rolling(100).skew() > 1.5, -0.5, 0.0)
    s['AV2_10'] = np.where(rng.rolling(100).skew() > 1.5, -0.5, 0.0)
    s['AV_GARCH_Extreme'] = np.where(rets**2 > (rets**2).rolling(252).mean() * 10, -1.0, 0.0)
    s['AV_Range_Expansion_Panic'] = np.where(rng / close > std(rets, 20) * 3, -1.0, 0.0)
    s['AV4_VWSemiDev_10'] = np.where(std(dn_vw, 20).diff() > 0, -0.4, 0.1)
    s['AV5_NegClust_12'] = np.where((rets < 0).rolling(20).sum() >= 14, 0.8, 0.0)

    # 4. Fibonacci Cycle Additions
    s['fib_cycle_21'] = np.where(df.index.dayofyear % 21 == 0, 0.5, 0.0)
    s['fib_cycle_55'] = np.where(df.index.dayofyear % 55 == 0, 0.5, 0.0)

    # Final Exposure Logic (Shifted by 1)
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_vde(df):
    df = df.copy()
    c, h, l, v, o = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = c.pct_change()
    
    # --- 1. Technical Indicator Helpers ---
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / (loss + 1e-10))))

    def get_macd(s, fast, slow, signal):
        f_ema = s.ewm(span=fast, adjust=False).mean()
        s_ema = s.ewm(span=slow, adjust=False).mean()
        macd = f_ema - s_ema
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig)

    # Indicator Pre-calculations
    sma10, sma20, sma50, sma200 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    std10, std20, std50 = c.rolling(10).std(), c.rolling(20).std(), c.rolling(50).std()
    ema20, ema100 = c.ewm(span=20).mean(), c.ewm(span=100).mean()
    rsi2, rsi14, rsi20 = get_rsi(c, 2), get_rsi(c, 14), get_rsi(c, 20)
    
    rng = (h - l).replace(0, 1e-6)
    ha5, ha20 = rng.rolling(5).mean(), rng.rolling(20).mean()
    ibs = (c - l) / rng
    atr20 = (pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)).rolling(20).mean()
    
    m12, s12, h12 = get_macd(c, 12, 26, 9)
    m5, s5, h5 = get_macd(c, 5, 13, 1)
    m3_7, s3_7, _ = get_macd(c, 3, 7, 2)
    l24_52, _, h3 = get_macd(c, 24, 52, 18)
    mv, sv, hv = get_macd(v, 5, 15, 5)

    vol20 = rets.rolling(20).std()
    vol_vol_ratio = v.pct_change().rolling(20).std() / (vol20 + 1e-10)
    v_norm = v / v.rolling(50).mean().replace(0, 1e-10)
    pk = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-10)
    
    tp = (h + l + c) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True) + 1e-10)
    
    mfi_pmf = (tp * v).where(tp > tp.shift(1), 0).rolling(14).sum()
    mfi_nmf = (tp * v).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (mfi_pmf / (mfi_nmf + 1e-6))))
    
    uo_bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    uo_tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    uo = 100 * (4*(uo_bp.rolling(7).sum()/uo_tr.rolling(7).sum().replace(0, 1e-6)) + 2*(uo_bp.rolling(14).sum()/uo_tr.rolling(14).sum().replace(0, 1e-6)) + (uo_bp.rolling(28).sum()/uo_tr.rolling(28).sum().replace(0, 1e-6))) / 7
    
    month, day, dow = df.index.month, df.index.day, df.index.dayofweek
    s = pd.DataFrame(index=df.index)

    # --- 2. Signal Generation (144 Unique Signals) ---
    s['s4'] = np.where(ibs < 0.2, 1.5, np.where(ibs > 0.8, -1.0, 0.5))
    s['s6'] = np.where(pk > pk.rolling(3).mean(), 1.0, -0.5)
    s['s7'] = np.where((day >= 10) & (day <= 15), 1.0, 0.1)
    s['i5_c12'] = np.where(rsi14 < 20, 1.5, -0.5)
    s['L4_C12'] = np.where(v.pct_change() > 0.5, -1.0, 0.5)
    s['IBS_Short_Vol_Conviction'] = np.where((ibs > 0.8) & (v > v.rolling(20).mean()), -0.4, 0.0)
    s['Low_Vol_Pullback'] = np.where((rets < 0) & (v / v.rolling(20).mean() < 0.7), 0.8, 0.0)
    s['Vol_Dry_Up'] = np.where(v < v.rolling(50).min().shift(1), -0.3, 0.0)
    s['JanEff'] = np.where((month == 1) & (day <= 15), 0.5, -0.1)
    s['IBS_Extreme_S'] = np.where(ibs > 0.98, -1.0, 0.0)
    s['Gap_Regime_2'] = np.where((o / c.shift(1) - 1).abs().rolling(10).sum() > atr20, -0.5, 0.2)
    s['Liquidity_Regime'] = np.where((v / (atr20 + 1e-10)) < (v / (atr20 + 1e-10)).rolling(50).mean(), -0.3, 0.1)
    s['VWAP_Dist'] = np.where(c > ((v * c).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)), 0.2, -0.4)
    emv = (((h+l)/2) - ((h.shift(1)+l.shift(1))/2)) / (v/1e6/(rng+1e-10) + 1e-10)
    s['EMV_Z'] = np.where(((emv - emv.rolling(20).mean()) / (emv.rolling(20).std() + 1e-10)) > 1.0, 0.3, -0.3)
    s['MACD_Vol_Mom'] = np.where(hv > 0, -0.3, 0.1)
    s['Kelt_Pos'] = np.where(c > (sma20 + 2*atr20), -0.4, np.where(c < (sma20 - 2*atr20), 0.4, 0.1))
    vm_p, vm_m = (h - l.shift(1)).abs(), (l - h.shift(1)).abs()
    s['Vol_Conv'] = np.where(vol_vol_ratio > 1.2, 1.5, 0.5)
    s['VolHedge'] = np.where(vol20*15.87 > rets.rolling(60).std()*15.87*1.3, -1.0, 0.0)
    s['Bear_Range_Expansion'] = np.where(rng > ha20 * 1.5, -0.3, 0.0)
    s['VolATR'] = np.where((v / v.rolling(20).mean() > 1.5) & (rets < 0), 1.0, 0.0)
    s['IBS_Ultra_S'] = np.where(ibs > 0.99, -1.0, 0.0)
    s['High_IBS_Fade'] = np.where(ibs > 0.95, -1.0, 0.0)
    s['Golden_Vol'] = np.where((sma50 > sma200) & (vol20 < 0.20), 1.5, 0.0)
    s['Gap_Ratio'] = np.where((o - c.shift(1)).abs() > (c - o).abs(), -0.5, 0.5)
    s['Gap_Fade'] = np.where((o / c.shift(1) - 1) > 0.01, -0.5, 0.2)
    mfr_p = (tp * v).where(tp > tp.shift(1), 0).rolling(5).sum()
    mfr_n = (tp * v).where(tp < tp.shift(1), 0).rolling(5).sum()
    s['MFI5'] = np.where(100 - (100 / (1 + mfr_p/(mfr_n+1e-10))) < 10, 1.0, 0.0)
    s['DonchMid'] = np.where((c < (h.rolling(20).max() + l.rolling(20).min())/2) & (c > (h.rolling(20).max() + l.rolling(20).min())/2 * 0.99), 0.5, 0.0)
    s['MidMonthRev'] = np.where((day >= 18) & (day <= 22), -0.3, 0.1)
    s['AugExit'] = np.where(month == 8, -1.0, 0.1)
    s['Range_Hedge'] = np.where(ha5 > atr20, -0.5, 0.5)
    semi_down = rets.clip(upper=0).rolling(20).std()
    s['Down_Beta'] = rets.rolling(20).corr(semi_down)
    s['Tail_Risk'] = rets.rolling(60).quantile(0.05) / (semi_down + 1e-10)
    s['Asym_Conv'] = semi_down.rolling(20).corr(rets.rolling(20).std())
    s['MACD_Decel'] = np.where(h12.diff() < h12.diff().shift(1), -0.5, 0.0)
    s['MACD_Cross'] = np.where(m12 < s12, -0.5, 0.0)
    s['Payday'] = np.where(day.isin([1, 15, 30]), 0.5, -0.1)
    s['Holiday_Frontrun'] = np.where((month == 11) & (day >= 20), 0.3, -0.1)
    s['WinDressing'] = np.where((month.isin([6,12])) & (day >= 28), 0.5, -0.1)
    s['Weekend_Gap'] = np.where((dow == 0) & ((o / c.shift(1) - 1) < -0.005), 0.4, -0.1)
    s['MR_EMA_Top'] = np.where(c > ema20 * 1.05, -1.0, 0.0)
    s['V_SellerExhaust'] = np.where((ibs < 0.1) & (v > v.rolling(20).mean() * 1.5) & (rsi14 < 25), 0.5, -0.1)
    s['S_ThursdayTurn'] = np.where(dow == 3, 0.2, -0.1)
    s['Stdev_Ratio'] = np.where(std10 / (std50 + 1e-10) > 1.2, 0.3, -0.1)
    s['VIX_Proxy'] = np.where((c / (c.rolling(20).max() + 1e-10) - 1) < -0.05, 0.5, -0.1)
    s['ZScore_Price'] = np.where(((c - sma20)/std20).abs() > 1.0, 0.4, 0.0)
    s['MACD_Fractal_D'] = np.where(m12.diff(10).abs() / m12.diff().abs().rolling(10).sum() > 0.6, 0.3, -0.1)
    s['v2'] = np.where((c.pct_change(60) > 0) & (ibs < 0.2), 1.5, 0.2)
    s['v11'] = np.where(0.5 * v.rolling(20).mean() * (c.diff(3)**2) > (0.5 * v.rolling(20).mean() * (c.diff(3)**2)).rolling(100).quantile(0.95), -0.5, 0.2)
    s['macd_slow_hist_pos'] = np.where(h3 > 0, 0.3, -0.3)
    s['ibs_vol_norm_revert'] = np.where((ibs < 0.2) & (v_norm > 1.5), 1.2, 0.0)
    s['Week_3_Bearish'] = np.where((day >= 15) & (day <= 21), -0.3, 0.0)
    s['vol_skew_2'] = np.where(v.rolling(60).skew() > 2.0, -0.4, 0.1)
    s['SummerSolstice'] = np.where((month == 6) & (day >= 15) & (day <= 25), -0.5, 0.1)
    s['Pivot_Rev'] = np.where(c < l.shift(1), -0.2, 0.1)
    s['rsi14_low'] = np.where(rsi14 < 30, 1.4, 0.0)
    s['ibs_ma20_low'] = np.where(ibs < ibs.rolling(20).mean(), 1.3, 0.0)
    s['m_cross_ext'] = np.where((m12 < -1) & (m12 > s12), 1.5, 0.0)
    s['h_accel_50'] = np.where(h12.diff(50) > 0, 1.0, 0.0)
    s['Day15_Only'] = np.where(day == 15, 1.0, 0.0)
    s['Sept_Mid_v2'] = np.where((month == 9) & (day >= 15) & (day <= 25), -1.0, 0.0)
    s['AV_GARCH_Extreme'] = np.where(rets**2 > (rets**2).rolling(252).mean() * 10, -1.0, 0.0)
    s['AV4_VWSemiDev_10'] = np.where(rets.where(rets < 0, 0).rolling(20).std().diff() > 0, -0.4, 0.1)
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['opt3_mag_vol'] = - (rets.abs() / (vol20 + 1e-10)).ewm(span=5).mean()
    s['it1_sig20'] = -(rng / c).rolling(20).std() / (vol20 + 1e-10)
    s['c40'] = np.where(uo > 70, -1.0, 0.0)
    s['v5'] = np.where(((rets.rolling(5).std() / (vol20 + 1e-10)) > 1.5) & (ibs < 0.2), 1.5, 0.2)
    s['v10'] = np.where((c < c.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['FriDeRisk_Old'] = np.where(dow == 4, -0.3, 0.1)
    gap_v = (o / c.shift(1) - 1).rolling(20).std()
    s['opt1_gap_vol'] = -(gap_v / (gap_v.ewm(span=100).mean() + 1e-10))
    s['mr51'] = (c.ewm(span=60).mean() - c) / c
    s['v4_10'] = (c - o) / (rng + 1e-6)
    s['sig41'] = -((rets * v).rolling(20).mean() / ((rets * v).rolling(20).std() + 1e-6))
    s['CrashProtector'] = np.where((month == 9) | ((month == 10) & (day <= 25)), -0.5, 0.0)
    s['OpExMon'] = np.where((day >= 15) & (day <= 21) & (dow == 0), 1.0, 0.0)
    s['Payday_1_15'] = np.where(day.isin([1, 15]), 1.0, 0.0)
    s['s100'] = np.where((o / c.shift(1) - 1 > 0.01) & (c < o), -1.0, 0.0)
    s['DeathCross'] = np.where(sma50 < sma200, -0.5, 0.0)
    s['BB_Squeeze'] = np.where(((4 * std20)/sma20) < ((4 * std20)/sma20).rolling(100).quantile(0.1), 0.5, 0.0)
    s['MR_6'] = np.where(rsi20 < 30, 0.2, 0.0)
    s['I1_14'] = np.where((c < c.shift(1)).rolling(5).sum()==5, 0.3, 0.0)
    s['Orbital_Decay'] = np.where((sma10 < sma50) & (sma10.diff() < 0), -0.2, 0.0)
    s['LoVol_PB'] = np.where((m12 < s12) & (v < v.rolling(20).mean()), 0.4, -0.2)
    s['Thanksgiving'] = np.where((month == 11) & (day >= 20) & (day <= 28), 0.4, -0.1)
    s['ibs_mean_revert_wide'] = np.where((ibs < 0.2) & ((c - sma20)/sma20 < -0.05), 1.5, 0.0)
    s['MeanDev10'] = np.where((c - sma10) / (sma10 + 1e-10) < -0.07, 1.5, 0.0)
    s['MACD_Zero_Cross'] = np.where(m12 > 0, 0.2, -0.2)
    s['MACD_Early_Turn'] = np.where((h12 > 0) & (h12.shift(1) < 0) & (h12.shift(10) < 0), 1.0, 0.0)
    s['MACD_Quick'] = np.where(m3_7 > s3_7, 0.4, -0.3)
    s['IBS_Mean_3'] = np.where(ibs.rolling(3).mean() < 0.2, 1.3, 0.0)
    s['IBS_Return_Pos'] = np.where((ibs < 0.2) & (rets > 0), 1.2, 0.0)
    s['IBS_ATR_Panic'] = np.where((ibs < 0.1) & (c.rolling(14).std() > c.rolling(14).std().shift(20) * 1.5), 1.5, 0.0)
    s['IBS_Bounce'] = np.where((ibs.shift(1) < 0.1) & (ibs > 0.3), 1.1, 0.0)
    s['IBS_RSI2_Cross'] = np.where((ibs < 0.3) & (rsi2 > 20) & (rsi2.shift(1) <= 20), 1.1, 0.0)
    s['IBS_BullSupp'] = np.where((ibs < 0.3) & (c > sma20), 1.0, 0.0)
    s['MACD_EMA_Filt'] = np.where((c > sma50) & (h12 > 0), 0.5, -0.2)
    s['ADX_Proxy'] = np.where((m12 > s12) & (rng.rolling(14).std() > rng.rolling(14).std().rolling(20).mean()), 0.5, -0.2)
    s['SpringFever'] = np.where(month == 4, 0.3, -0.1)
    s['vol_roc_confirm'] = np.where(v.pct_change() > 0.1, 0.1, -0.1)
    s['st4_17'] = np.where(v.rolling(5).mean() > v.rolling(20).mean(), 0.5, -0.5)

    # --- 3. Final Average ---
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

def strat_xop(df):
    """
    Consolidated Strategy for XOP after core, breadth, asymmetric vol, and MACD iterations.
    Optimized to handle Energy sector drawdowns (GFC, 2015 Oil Glut, 2018 Q4 selloff).
    Constraints: Exposure [-1.0, 1.5]. No lookahead bias.
    """
    close, high, low, vol, open_p = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
    rets = df['Close'].pct_change().fillna(0)
    
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def std(s, n): return s.rolling(n).std()
    def sma(s, n): return s.rolling(n).mean()

    # Pre-computations
    rng = (high - low).replace(0, 1e-6)
    ibs = (close - low) / rng
    vol20 = rets.rolling(20).std()
    m12 = ema(close, 12) - ema(close, 26)
    s12 = m12.ewm(span=9, adjust=False).mean()
    h12 = m12 - s12
    ret60 = close.pct_change(60)
    tp = (high + low + close) / 3
    mfi = 100 - (100 / (1 + ((tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum() / 
                            (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum().replace(0, 1e-6))))
    gap_v = (open_p / close.shift(1) - 1).rolling(20).std()
    ha20 = rng.rolling(20).mean()
    ha5 = rng.rolling(5).mean()
    dom = df.index.day
    dw_s = df.index.dayofweek
    m_s = df.index.month
    
    delta = close.diff()
    up = delta.where(delta > 0, 0)
    down = delta.where(delta < 0, 0).abs()
    rsi14 = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean().replace(0, 1e-6)))
    cmf = (((close - low) - (high - close)) / rng * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, 1e-6)
    dn_vw = (rets * vol / sma(vol, 20)).where(rets < 0, 0)
    up_std = std(rets.where(rets > 0, 0), 20)
    dn_std = std(rets.where(rets < 0, 0), 20)
    
    s = pd.DataFrame(index=df.index)
    
    # 1. Base Systematic Ensemble
    s['v8'] = np.where(mfi < 20, 1.4, 0.4)
    s['v10'] = np.where((close < close.shift(4)).astype(int).rolling(9).sum() >= 8, 1.4, 0.3)
    s['Vol_Conv'] = np.where(vol.pct_change().rolling(20).std() / (vol20 + 1e-10) > 1.2, 1.5, 0.5)
    s['opt1_gap_vol'] = -(gap_v / (ema(gap_v, 100) + 1e-10))
    s['opt3_mag_vol'] = - ema(rets.abs() / (vol20 + 1e-10), 5)
    s['it1_sig20'] = -(rng / close).rolling(20).std() / (vol20 + 1e-10)
    s['mr51'] = (ema(close, 60) - close) / close
    s['Intra_V_Decel'] = np.where(ha5 < ha20 * 0.8, 0.5, -0.2)
    s['FriDeRisk'] = np.where(dw_s == 4, -0.3, 0.1)
    s['JanEff'] = np.where((m_s == 1) & (dom <= 15), 0.5, -0.1)
    s['WinDressing'] = np.where((m_s == 6) | (m_s == 12) & (dom >= 28), 0.5, -0.1)
    s['Weekend_Gap'] = np.where((dw_s == 0) & ((open_p / close.shift(1) - 1) < -0.005), 0.4, -0.1)
    s['v6'] = np.where(close.pct_change(5) < -0.07, 1.5, 0.2)
    s['cmf_rev'] = np.where(cmf < -0.2, 1.4, 0.0)
    s['ibs_ma20_low'] = np.where(ibs < ibs.rolling(20).mean(), 1.3, 0.0)
    s['m_cross_ext'] = np.where((m12 < -1) & (m12 > s12), 1.5, 0.0)
    s['AV4_5'] = np.where(rng.rolling(100).skew() > 1.0, -0.5, 0.0)
    s['AV4_VWSemiDev_10'] = np.where(std(dn_vw, 20).diff() > 0, -0.4, 0.1)
    s['macd_bull_flag'] = np.where((m12 > 0) & (h12 < 0) & (h12 > h12.shift(1)), 1.4, 0.0)
    s['macd_vw_slope'] = np.where((m12 * vol).diff() > 0, 0.5, -0.5)
    s['br2_exhaust_dn'] = np.where((close < close.shift(3)) & (rng/close.rolling(50).mean() > 2.0), 0.8, -0.1)
    s['br3_wide_stop_run'] = np.where((low < low.shift(1)) & (close > high.shift(1)), 1.2, -0.2)
    s['br4_neg_drift'] = np.where(close.diff(5).rolling(20).mean() < 0, -0.3, 0.1)
    s['br5_trap_bounce'] = np.where((low < low.rolling(5).min().shift(1)) & (close > open_p), 0.8, -0.2)
    s['av3_16'] = np.where((up_std > dn_std) & (close > sma(close, 200)), 0.6, -0.1)
    s['macd_h1_slope_v'] = np.where(h12.diff() * vol > 0, 0.4, -0.4)
    s['macd_peak_rev'] = np.where(m12 > m12.rolling(50).max().shift(1), -0.8, 0.1)
    
    l24_52 = ema(close, 24) - ema(close, 52)
    h3 = l24_52 - ema(l24_52, 18)
    s['macd_slow_hist_pos'] = np.where(h3 > 0, 0.3, -0.3)
    s['macd_trough_rev'] = np.where(m12 < m12.rolling(50).min().shift(1), 1.0, -0.2)
    m1 = ema(close, 10) - ema(close, 20); sl1 = ema(m1, 7); h1 = m1 - sl1
    s['it_macd_3'] = np.where(h1 > h1.rolling(10).mean(), 0.5, -0.5)

    # Combined Exposure chosen at close T, held for T+1.
    exposure = s.mean(axis=1).shift(1).fillna(0).clip(-1.0, 1.5)
    return exposure

# --- Execution ---
files_map = {
    'BWX': 'BWX.csv',
    'DBC': 'DBC.csv',
    'EWJ': 'EWJ.csv',
    'EWW': 'EWW.csv',
    'FXI': 'FXI.csv',
    'FXU': 'FXU.csv',
    'GLD': 'GLD.csv',
    'GSY': 'GSY.csv',
    'HYG': 'HYG.csv',
    'ITA': 'ITA.csv',
    'IYH': 'IYH.csv',
    'NLR': 'NLR.csv',
    'PGX': 'PGX.csv',
    'PID': 'PID.csv',
    'QQQ': 'QQQ.csv',
    'RPV': 'RPV.csv',
    'RWJ': 'RWJ.csv',
    'RWO': 'RWO.csv',
    'VAW': 'VAW.csv',
    'VDC': 'VDC.csv',
    'VDE': 'VDE.csv',
    'XOP': 'XOP.csv',
}

strategy_returns_list = []

for name, file in files_map.items():
    df = pd.read_csv(file, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date Time'], format='mixed')
    df = df.set_index('Date').sort_index()
    
    # Calculate individual strategy exposure
    if name == 'BWX': exp = strat_bwx(df)
    if name == 'DBC': exp = strat_dbc(df)
    elif name == 'EWJ': exp = strat_ewj(df)
    elif name == 'EWW': exp = strat_eww(df)
    elif name == 'FXI': exp = strat_fxi(df)
    elif name == 'FXU': exp = strat_fxu(df)
    elif name == 'GLD': exp = strat_gld(df)
    elif name == 'GSY': exp = strat_gsy(df)
    elif name == 'HYG': exp = strat_hyg(df)
    elif name == 'ITA': exp = strat_ita(df)
    elif name == 'IYH': exp = strat_iyh(df)
    elif name == 'NLR': exp = strat_nlr(df)
    elif name == 'PGX': exp = strat_pgx(df)
    elif name == 'PID': exp = strat_pid(df)
    elif name == 'QQQ': exp = strat_qqq(df)
    elif name == 'RPV': exp = strat_rpv(df)
    elif name == 'RWJ': exp = strat_rwj(df)
    elif name == 'RWO': exp = strat_rwo(df)
    elif name == 'VAW': exp = strat_vaw(df)
    elif name == 'VDC': exp = strat_vdc(df)
    elif name == 'VDE': exp = strat_vde(df)
    elif name == 'XOP': exp = strat_xop(df)
    
    # Calculate individual returns
    # Note: We keep NaNs for periods where the asset didn't exist yet
    asset_rets = (exp * df['Close'].pct_change()).rename(name)
    strategy_returns_list.append(asset_rets)

# Use an Outer Join to keep all dates from all assets
strategy_returns = pd.concat(strategy_returns_list, axis=1)

# --- Equal Weight Calculation ---
# Identify all assets (columns) in the dataframe
asset_names = strategy_returns.columns
num_assets = len(asset_names)

# Assign 1/N weight to every asset
derived_w = pd.Series(1/num_assets, index=asset_names)

print("--- Equal Weights Assigned ---")
print(derived_w)

# Correct Portfolio Return Calculation (Handling unbalanced start dates)
def calculate_portfolio(returns_df, weights_series):
    # Create a mask of which assets are available on each day (not NaN)
    available_mask = returns_df[weights_series.index].notna()
    
    # Get weight values
    w = weights_series.values
    
    # Calculate daily returns: (Daily Returns * Weights)
    # Fill NaNs with 0 so they don't zero out the whole row sum
    daily_weighted_rets = returns_df[weights_series.index].fillna(0).dot(w)
    
    # Re-normalize: If an asset is missing, divide the daily return by the sum 
    # of weights of currently active assets to maintain 100% exposure.
    active_weights_sum = available_mask.dot(w)
    
    return daily_weighted_rets / active_weights_sum

# Apply the Equal Weight portfolio calculation
strategy_returns['EW_Portfolio'] = calculate_portfolio(strategy_returns, derived_w)

# Performance Summary
def get_metrics(df_rets):
    res = []
    for col in df_rets.columns:
        ar = df_rets[col].mean() * 252
        av = df_rets[col].std() * np.sqrt(252)
        res.append({'Strategy': col, 'Return': ar, 'Vol': av, 'Sharpe': ar/(av+1e-10)})
    return pd.DataFrame(res).set_index('Strategy')

p_map = {
    "TRAIN PERIOD (PRE-2020)": strategy_returns.loc[:'2019-12-31'],
    "VALIDATION PERIOD (2020-2021)": strategy_returns.loc['2020-01-01':'2021-12-31'],
    "BLIND HOLDOUT PERIOD (2022-PRESENT)": strategy_returns.loc['2022-01-01':],
    "FULL PERIOD": strategy_returns
}

for label, df_slice in p_map.items():
    print(f"\n--- {label} ---")
    print(get_metrics(df_slice))