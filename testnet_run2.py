
#!/usr/bin/env python3
"""
Live REST‑polling trading bot for 5m bars (Binance Futures testnet):
- Fetches ~120 days of 5m bars via paged REST
- On each new closed 5m bar (polled every 10s), appends to history (never drops)
- Uses ml_way3.py strategy logic (preprocess, find_entries, run_backtest)
- Executes market entry + multi‑stage exits via REST
- Logs all trades to CSV
"""
import os
import time
import math
import logging
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
import importlib.util

# ── CONFIG ─────────────────────────────────────────────────────
SYMBOL        = "BTCUSDT"
INTERVAL      = "5m"
DAYS_HISTORY  = 120
SLIPPAGE      = 0.0005
LEVERAGE      = 10
CONTRACT_SZ   = 0.001
POLL_SECONDS  = 10       # REST polling interval
TRADES_CSV    = "live_trades.csv"
LOG_FILE      = "live_bot.log"

# ── LOGGING ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ── LOAD STRATEGY MODULE ───────────────────────────────────────
spec = importlib.util.spec_from_file_location("ml_way3", "ml_way3.py")
ml   = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

preprocess_1h = ml.preprocess_1h
preprocess_5m = ml.preprocess_5m
find_entries  = ml.find_entries
LOOKBACK      = ml.LOOKBACK

# ── LOAD HYPERPARAMETERS ──────────────────────────────────────
combo = pd.read_csv("use_this_combi.csv").iloc[0]
params = combo.drop(["sharpe","num_entries"]).to_dict()

# ── INIT OUTPUT CSV ────────────────────────────────────────────
os.makedirs(os.path.dirname(TRADES_CSV) or '.', exist_ok=True)
pd.DataFrame(columns=["timestamp","side","qty","price","atr","exit_stage"]) \
  .to_csv(TRADES_CSV, index=False)

# ── LOAD API KEYS ──────────────────────────────────────────────
load_dotenv()
API_KEY    = os.getenv("BINANCE_FUTURES_API_KEY")
API_SECRET = os.getenv("BINANCE_FUTURES_SECRET_KEY")
client = Client(API_KEY, API_SECRET, testnet=True)
client.API_URL, client.FUTURES_URL = (
    'https://testnet.binancefuture.com',
    'https://testnet.binancefuture.com/fapi'
)
client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)

# ── FETCH ~120 DAYS HISTORY ────────────────────────────────────
target_start = datetime.utcnow() - timedelta(days=DAYS_HISTORY)
all_bars = []
end_time = None
while True:
    params = {
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'limit': 1500
    }
    if end_time:
        params['endTime'] = int(end_time.timestamp()*1000)
    klines = client.futures_klines(**params)
    if not klines:
        break
    bars = []
    for k in klines:
        t = datetime.utcfromtimestamp(k[0]/1000)
        bars.append({
            'datetime': t,
            'open': float(k[1]), 'high': float(k[2]),
            'low': float(k[3]),  'close': float(k[4]),
            'volume': float(k[5])
        })
    all_bars = bars + all_bars
    first = all_bars[0]['datetime']
    if first <= target_start or len(klines) < 1500:
        break
    end_time = bars[0]['datetime']

history_df = pd.DataFrame(all_bars)
history_df.sort_values('datetime', inplace=True)
history_df.reset_index(drop=True, inplace=True)
logging.info("Loaded %d bars (~%.1f days), from %s to %s",
             len(history_df), len(history_df)*5/60/24,
             history_df['datetime'].iloc[0], history_df['datetime'].iloc[-1])

# ── LIVE STATE ─────────────────────────────────────────────────
in_trade = False
entry_rec = {}
exit_stage= 0
peak_price= None
entry_time= None

# ── LIVE LOOP ─────────────────────────────────────────────────
while True:
    try:
        time.sleep(POLL_SECONDS)
        kl = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=2)
        closed = kl[-2]
        ts = datetime.utcfromtimestamp(closed[0]/1000)
        if ts <= history_df['datetime'].iloc[-1]:
            continue
        bar = {
            'datetime': ts,
            'open': float(closed[1]), 'high': float(closed[2]),
            'low': float(closed[3]),  'close': float(closed[4]),
            'volume': float(closed[5])
        }
        history_df = history_df.append(bar, ignore_index=True)
        logging.info("New bar: %s", ts)

        # 1H resample
        df1h = (
            history_df.set_index('datetime')
                      .resample('1H')
                      .agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
                      .dropna()
                      .reset_index()
        )
        # slice for signals
        df5 = history_df.iloc[-(LOOKBACK+1):].copy()

        hf   = preprocess_1h(df1h, params)
        b5   = preprocess_5m(history_df.copy(), params, hf)
        sigs = find_entries(b5, params)

        # ENTRY
        if not in_trade and len(sigs):
            sig = sigs.iloc[-1]
            side= 'BUY' if sig.signal=='CALL' else 'SELL'
            price, atr = sig.open, sig.atr
            risk = history_df['close'].iloc[-1] * params['RISK_PCT'] * LEVERAGE
            qty  = math.floor(risk/(params['SL_ATR_MULT']*atr*CONTRACT_SZ))*CONTRACT_SZ
            if qty>=CONTRACT_SZ:
                client.futures_create_order(symbol=SYMBOL, side=side, type='MARKET', quantity=qty)
                in_trade, exit_stage, peak_price, entry_time = True, 0, price, ts
                entry_rec = {'side':side,'qty':qty,'entry_price':price,'atr':atr}
                logging.info("ENTER %s qty=%.4f@%.2f", side, qty, price)
                pd.DataFrame([{**entry_rec,'timestamp':ts,'exit_stage':'ENTRY'}])\
                  .to_csv(TRADES_CSV,mode='a',header=False,index=False)

        # EXIT logic (same multi‑stage as backtest)
        if in_trade:
            highs, lows = df5['high'].max(), df5['low'].min()
            if exit_stage==0 and ((entry_rec['side']=='BUY' and highs>=entry_rec['entry_price']+entry_rec['atr']) or
                                  (entry_rec['side']=='SELL' and lows<=entry_rec['entry_price']-entry_rec['atr'])):
                exit_stage=1
                client.futures_cancel_all_open_orders(symbol=SYMBOL)
                be = entry_rec['entry_price'] + (0.2*entry_rec['atr'] if entry_rec['side']=='BUY' else -0.2*entry_rec['atr'])
                client.futures_create_order(symbol=SYMBOL,
                    side=('SELL' if entry_rec['side']=='BUY' else 'BUY'),
                    type='STOP_MARKET',stopPrice=round(be,2),closePosition=True)
                logging.info("BREAKEVEN set @%.2f",be)
                pd.DataFrame([{'timestamp':ts,'exit_stage':'BREAKEVEN'}])\
                    .to_csv(TRADES_CSV,mode='a',header=False,index=False)
            if exit_stage<=1 and ((entry_rec['side']=='BUY' and highs>=entry_rec['entry_price']+2*entry_rec['atr']) or
                                   (entry_rec['side']=='SELL' and lows<=entry_rec['entry_price']-2*entry_rec['atr'])):
                half = entry_rec['qty']/2
                client.futures_create_order(symbol=SYMBOL,
                    side=('SELL' if entry_rec['side']=='BUY' else 'BUY'),
                    type='MARKET',quantity=half)
                exit_stage=2
                logging.info("SCALE half@market qty=%.4f",half)
                pd.DataFrame([{'timestamp':ts,'exit_stage':'SCALE'}])\
                    .to_csv(TRADES_CSV,mode='a',header=False,index=False)
            peak_price = max(peak_price,highs) if entry_rec['side']=='BUY' else min(peak_price,lows)
            if exit_stage<=2:
                client.futures_cancel_all_open_orders(symbol=SYMBOL)
                trail = peak_price - (entry_rec['atr'] if entry_rec['side']=='BUY' else -entry_rec['atr'])
                client.futures_create_order(symbol=SYMBOL,
                    side=('SELL' if entry_rec['side']=='BUY' else 'BUY'),
                    type='STOP_MARKET',stopPrice=round(trail,2),closePosition=True)
                exit_stage=3
                logging.info("TRAIL set @%.2f",trail)
                pd.DataFrame([{'timestamp':ts,'exit_stage':'TRAIL'}])\
                    .to_csv(TRADES_CSV,mode='a',header=False,index=False)
            if ts>= entry_time + timedelta(minutes=LOOKBACK*5):
                client.futures_create_order(symbol=SYMBOL,
                    side=('SELL' if entry_rec['side']=='BUY' else 'BUY'),type='MARKET',quantity=entry_rec['qty'])
                exit_stage=4
                logging.info("TIME-STOP exit@market")
                pd.DataFrame([{'timestamp':ts,'exit_stage':'TIME'}])\
                    .to_csv(TRADES_CSV,mode='a',header=False,index=False)
            pos = client.futures_position_information(symbol=SYMBOL)
            amt = float(next(p for p in pos if p['symbol']==SYMBOL)['positionAmt'])
            if amt==0 and in_trade:
                in_trade=False
                logging.info("Trade closed")
                pd.DataFrame([{'timestamp':ts,'exit_stage':'CLOSED'}])\
                    .to_csv(TRADES_CSV,mode='a',header=False,index=False)

    except Exception as e:
        logging.exception("Error in main loop")
        time.sleep(POLL_SECONDS)

