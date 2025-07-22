# ── Cell: Single‑Position Live Loop with OCO Exits & Metrics Logging ───────────────────────
import os, sys, time, math, requests
from datetime import datetime
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import importlib.util

# Ensure output directory exists
os.makedirs("output", exist_ok=True)
network_log = "output/network.logs"

# 1) Load your API keys
load_dotenv(".env")
API_KEY = os.getenv("BINANCE_FUTURES_API_KEY")
API_SECRET = os.getenv("BINANCE_FUTURES_SECRET_KEY")
if not API_KEY or not API_SECRET:
    raise RuntimeError("Set API keys in .env")

# 2) Initialize & point client
client = Client(API_KEY, API_SECRET, {"timeout": 30})
client.API_URL     = "https://fapi.binance.com"
client.FUTURES_URL = "https://fapi.binance.com/fapi"


# 3) Import ML functions
sys.path.append(".")
spec = importlib.util.spec_from_file_location("ml_way2", "./ml_way2.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)
preprocess_1h, preprocess_5m, find_entries, run_backtest = (
    ml.preprocess_1h, ml.preprocess_5m, ml.find_entries, ml.run_backtest
)

# 4) Load combo & settings
combo = pd.read_csv("use_this_combi.csv").iloc[0]
params = combo.drop(["sharpe", "num_entries"]).to_dict()
SYMBOL, LEVERAGE, CONTRACT_SZ, USDT_CAPITAL = "BTCUSDT", 10, 0.001, 58.0

trade_file       = "output/testnet_trades.csv"
metrics_file     = "output/testnet_metrics.csv"
metrics_cols     = [
    "cycle_time", "total_return_pct","annualized_return_pct","sharpe_ratio",
    "max_drawdown_pct","win_rate_pct","num_trades","profit_factor","avg_win_usd",
    "avg_loss_usd","max_win_usd","max_loss_usd","avg_duration_min","final_balance_usdt",
    "ATR_WINDOW","EMA_LONG","SL_ATR_MULT","TP_ATR_MULT","VOL_MULT",
    "RISK_PCT","REGIME_MULT"
]

# Ensure CSVs exist
if not os.path.exists(trade_file):
    pd.DataFrame(columns=["timestamp","side","qty","price","exit_reason"]).to_csv(trade_file, index=False)
if not os.path.exists(metrics_file):
    pd.DataFrame(columns=metrics_cols).to_csv(metrics_file, index=False)

# 5) Set leverage once
client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)

# Flag to allow only one open position
position_open = False

def open_position_with_oco(side, qty, sl_price, tp_price):
    # Market entry
    client.futures_create_order(
        symbol=SYMBOL, side=side, type="MARKET", quantity=round(qty, 3)
    )
    # Stop‐loss
    client.futures_create_order(
        symbol=SYMBOL,
        side="SELL" if side=="BUY" else "BUY",
        type="STOP_MARKET",
        stopPrice=sl_price,
        closePosition=True
    )
    # Take‐profit
    client.futures_create_order(
        symbol=SYMBOL,
        side="SELL" if side=="BUY" else "BUY",
        type="TAKE_PROFIT_MARKET",
        stopPrice=tp_price,
        closePosition=True
    )

# Fetch helper unchanged...
def fetch_latest_data(retries=3, backoff_s=5):
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    for attempt in range(1, retries+1):
        try:
            kl5 = client.futures_klines(symbol=SYMBOL, interval="5m", limit=ml.LOOKBACK+50)
            df5 = pd.DataFrame(kl5, columns=cols)
            df5[["open","high","low","close","volume"]] = df5[["open","high","low","close","volume"]].astype(float)
            df5["datetime"] = pd.to_datetime(df5["open_time"], unit="ms")

            kl1 = client.futures_klines(symbol=SYMBOL, interval="1h", limit=200)
            df1h = pd.DataFrame(kl1, columns=cols)
            df1h[["open","high","low","close","volume"]] = df1h[["open","high","low","close","volume"]].astype(float)
            df1h["datetime"] = pd.to_datetime(df1h["open_time"], unit="ms")

            return df5, df1h
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            msg = f"{datetime.utcnow().isoformat()}  Fetch attempt {attempt} failed: {e}\n"
            with open(network_log, "a") as nf:
                nf.write(msg)
            print(msg, end="")
            if attempt < retries:
                time.sleep(backoff_s)
            else:
                raise RuntimeError("Failed to fetch market data after retries") from e

# 7) Continuous loop w/ metrics logging
try:
    while True:
        cycle_time = datetime.utcnow().isoformat()
        df5_live, df1h_live = fetch_latest_data()

        # Compute signals
        hf = preprocess_1h(df1h_live, params)
        b5 = preprocess_5m(df5_live.copy(), params, hf)
        sigs = find_entries(b5, params)

        # Entry logic
        if not position_open and not sigs.empty:
            sig = sigs.iloc[-1]
            side = "BUY" if sig.signal=="CALL" else "SELL"
            entry_price = sig.open
            sl_price = entry_price - params["SL_ATR_MULT"]*sig.atr if side=="BUY" else entry_price + params["SL_ATR_MULT"]*sig.atr
            tp_price = entry_price + params["TP_ATR_MULT"]*sig.atr if side=="BUY" else entry_price - params["TP_ATR_MULT"]*sig.atr

            risk_usd = USDT_CAPITAL * params["RISK_PCT"] * LEVERAGE
            qty = math.floor(risk_usd / (abs(entry_price - sl_price) * CONTRACT_SZ)) * CONTRACT_SZ

            open_position_with_oco(side, qty, sl_price, tp_price)
            with open(trade_file, "a") as f:
                f.write(f"{cycle_time},{side},{qty},{entry_price},ENTRY\n")
            position_open = True

        # Exit detection
        if position_open:
            positions = client.futures_position_information(symbol=SYMBOL)
            amt = float(next(p for p in positions if p['symbol']==SYMBOL)['positionAmt'])
            if amt == 0:
                with open(trade_file, "a") as f:
                    f.write(f"{cycle_time},N/A,N/A,N/A,EXIT\n")
                position_open = False

        # Metrics logging (run backtest on live 5m slice)
        hist_trades, hist_metrics = run_backtest(b5, params)
        # assemble metrics row
        row = {**hist_metrics, **params}
        row["cycle_time"] = cycle_time
        dfm = pd.DataFrame([row])[metrics_cols]
        dfm.to_csv(metrics_file, mode='a', header=False, index=False)

        print(f"{cycle_time} → logged metrics and trades.")
        time.sleep(5 * 60)

except KeyboardInterrupt:
    print("Live loop stopped by user.")
