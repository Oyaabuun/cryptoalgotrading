Hello Everyone In this post i will be sharing my progress and need suggestions

https://github.com/Oyaabuun/cryptoalgotrading

using histgbm , xgboost and did hyperparameters tuning 2. used bayesian search optuna based tuning

either you can use grid search from random numbers combinations to brute force and find params or else you can go for any method which suits you initially i was using grid search then GPT-o4 ,it suggested bayesian based optuna, but i used xgboost and tried to find parameters with that first then i went on to use method 2 which was suggested by chatgpt.

I had collected 5 yrs data and then used oldest 1460(4 years) as train /test. in xgboost 70/30 spilt was done. kept last 1 year data as unseen in Bayesian method a walk forward method was used after splitting the data to train /test, it on multiple timeframes within the same 1460 days .

Whatever money was earned 30 percent was banked and 70 percent was reused as equity again for further positions. Got top 5 params out of these .used these params to test on last one month of data mid of june to mid of july 2025 ( this is complete blind data) .

I am not into price prediction model rather finding best magic numbers params tunning them.

observations

FUTURES BTC/USDT PERP on last 4 weeks test a 58USDT balance account with 10X leverage and 0.05 slippage assumption and 0.010 taker fee XGBOOST PARAMS PERFORMANCE

Total net PnL (USDT)          39.256281 Final combined equity+banked (USDT)  75.865100 Win rate (%)              75.000000Average win (USDT)           13.249496Average loss (USDT)          -0.492207Profit factor             80.755553

BAYESIAN OPTUNA PERFORMANCE total_return_pct,annualized_return_pct,sharpe_ratio,win_rate_pct,profit_factor,max_drawdown_pct,final_balance_usdt46.66271675378246,1299,,100.0,-inf,0.0,85.06437571719383

Now few things to consider real deployment will have slippage dynamic, partial order fills , rate limits but still i am currently observing the performance of model .some of you might think why not try testing on testnet futures of binance.

Its not really practical as there is price difference at any give point of time if you observer the charts and volume also is not similar to live markets. so rather a live like csv loading simulator is better with dynamic slippage functions i feel .If anyone knows about platforms which give real simulation of volumes and prices in futures BTC/USDT PERP or ETH/USDT data please suggest in comments . checkout the output (for bayesian models params results) and output_hist_sim(xgboost) params performance for xgboost run ml_way3.py to generate params , ML_WAY3_BACKTEST.py to generate metrcies and trade and use the last cell of the ml_way.ipynb to see the performance for bayesian use ml_way4_optuna.py use this to generate top_optuna_combos_filtered.csv and use the params to backtest testnet_run4_optuna_backtest_py #algotrading#crypto

You can reach out to me on my telegram : aroy@52
