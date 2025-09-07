# Trading Bot v4 - Fully Advanced (Railway-friendly)

This is the advanced release with:
- Multi-timeframe confirmations (15m, 1h, 4h)
- EMA(50/200), RSI, MACD checks
- ATR-based SL and multi-TPs (TP1..TP5)
- Advice block attached to each signal and each TP notification
- Screenshot generator showing Entry/SL/TPs and side
- Webhook mode for Railway (set WEBHOOK_URL), or polling for local

## Deploy on Railway
1. Push this repo to GitHub.
2. In Railway project -> Variables, set:
   - BOT_TOKEN = <your token from BotFather>
   - WEBHOOK_URL = https://<your-railway-domain>.up.railway.app (recommended)
3. Deploy and check logs. Bot will start and set webhook if WEBHOOK_URL provided.

## Commands
- `/analyze SYMBOL` - run analysis and get entry/SL/TPs + image + advice
- `/monitor_here` - monitor the last analyzed symbol; bot will notify this chat on TP/SL hits
- `/list_trades` - show open monitored trades
- `/stoptrade <index_or_symbol>` - stop monitoring a trade
- `/backtest SYMBOL DAYS` - simple backtest

Notes: This bot is informational only. Test thoroughly before using real funds.