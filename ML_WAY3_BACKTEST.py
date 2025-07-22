#!/usr/bin/env python3
"""
Offline simulation of your live 5m WebSocket trading logic on historical CSV data.
Reads a 5m CSV, resamples to 1h, applies your entry/exit rules exactly as live,
logs simulated trades and backtest metrics each bar.
"""
import os
import math
import pandas as pd
from datetime import datetime
import importlib.util

# ── User Configuration ────────────────────────────────────────────
DATA_5M_CSV   = r"D:\algo_crypto\algo_crypto\mymodules\data\BTCUSDT_5m_7day.csv"
OUTPUT_DIR    = "output_hist_sim"
TRADES_FILE   = os.path.join(OUTPUT_DIR, "trades.csv")
METRICS_FILE  = os.path.join(OUTPUT_DIR, "metrics.csv")

# ── Load your backtest/ML module (ml_way3.py) ─────────────────────
spec = importlib.util.spec_from_file_location("ml_way3", "ml_way3.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

preprocess_1h   = ml.preprocess_1h
preprocess_5m   = ml.preprocess_5m
find_entries    = ml.find_entries
run_backtest    = ml.run_backtest
LOOKBACK        = ml.LOOKBACK
INITIAL_USDT    = ml.INITIAL_USDT
LEVERAGE        = ml.LEVERAGE
CONTRACT_SZ     = ml.CONTRACT_SZ

# ── Load your chosen hyperparameters ────────────────────────────
combo = pd.read_csv("top_optuna_combo_filtered.csv").iloc[0]
params = combo.drop(["sharpe", "num_entries"]).to_dict()

# ── Prepare output files ─────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trades CSV: one row per closed trade
pd.DataFrame(columns=[
    "timestamp","side","contracts","entry_px","exit_px",
    "pnl_usd","equity","banked"
]).to_csv(TRADES_FILE, index=False)

# Metrics CSV: one row per bar
metric_keys = [
    "total_return_pct","annualized_return_pct","sharpe_ratio",
    "max_drawdown_pct","win_rate_pct","num_trades","profit_factor",
    "avg_win_usd","avg_loss_usd","max_win_usd","max_loss_usd",
    "avg_duration_min","final_balance_usdt"
]
metric_cols = ["cycle_time"] + metric_keys + list(params.keys())
pd.DataFrame(columns=metric_cols).to_csv(METRICS_FILE, index=False)

# ── Load 5m CSV ──────────────────────────────────────────────────
df5_all = pd.read_csv(DATA_5M_CSV)

# Normalize timestamp → datetime
if "timestamp" in df5_all.columns:
    df5_all["datetime"] = pd.to_datetime(df5_all["timestamp"], unit="ms")
elif "open_time" in df5_all.columns:
    df5_all["datetime"] = pd.to_datetime(df5_all["open_time"], unit="ms")
else:
    df5_all["datetime"] = pd.to_datetime(df5_all["datetime"])

df5_all = df5_all.sort_values("datetime").reset_index(drop=True)

# ── Simulation State ─────────────────────────────────────────────
prev_trade_count = 0

# ── Main Loop: emulate one new 5m bar per iteration ─────────────
for i in range(LOOKBACK, len(df5_all)):
    # 1) build the rolling 5m slice
    df5 = df5_all.iloc[: i+1].copy()

    # 2) derive 1h bars by resampling those 5m bars
    df1h = (
        df5.set_index("datetime")
           .resample("1H")
           .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
           .dropna()
           .reset_index()
    )

    # 3) compute signals and backtest on that slice
    hf   = preprocess_1h(df1h, params)
    b5   = preprocess_5m(df5.copy(), params, hf)
    trades_df, metrics = run_backtest(b5, params)

    # 4) log any new trades
    if len(trades_df) > prev_trade_count:
        new_trades = trades_df.iloc[prev_trade_count:]
        for _, t in new_trades.iterrows():
            pd.DataFrame([{
                "timestamp": t["exit_dt"].isoformat(),
                "side":      t["signal"],
                "contracts": t["contracts"],
                "entry_px":  t["entry_px"],
                "exit_px":   t["exit_px"],
                "pnl_usd":   t["pnl_usd"],
                "equity":    t["equity"],
                "banked":    t["banked"]
            }]).to_csv(TRADES_FILE, mode="a", header=False, index=False)
        prev_trade_count = len(trades_df)

    # 5) log metrics for this bar
    row = {"cycle_time": df5["datetime"].iloc[-1].isoformat()}
    row.update(metrics)
    row.update(params)
    pd.DataFrame([row])[metric_cols].to_csv(METRICS_FILE, mode="a", header=False, index=False)

print(f"Simulation complete → trades: {TRADES_FILE}, metrics: {METRICS_FILE}")


