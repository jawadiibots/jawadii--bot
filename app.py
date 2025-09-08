#!/usr/bin/env python3
"""
app.py - Telegram Crypto Signal Bot (Advanced + Futuristic features)
Features:
 - Multi-timeframe analysis (EMA/MACD/RSI/ATR/Volume)
 - Clear Entry / SL / multi TPs + chart
 - Per-user notifications + channel posting
 - Daily RSS crypto-news posting
 - Background monitor for TP/SL with advices
 - Conversation-like /risk flow
"""

import os, io, asyncio, logging, traceback, math, json, time
from datetime import datetime, timedelta
import feedparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dateutil.parser

import ccxt
from tradingview_ta import TA_Handler, Interval

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# local utils (ensure utils.py from earlier is in same repo)
from utils import save_json, load_json, nice, compute_position_size

# ---------- Configuration ----------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is required")

ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID")) if os.getenv("ADMIN_USER_ID") else None
CHANNEL_CHAT_ID = os.getenv("CHANNEL_CHAT_ID")  # e.g. @yourchannel or -1001234567890
CHANNEL_INVITE_LINK = os.getenv("CHANNEL_INVITE_LINK", "https://t.me/+T2lFw-AjK21kYWM0")
CHANNEL_FOOTER = f"\n\nJoin channel: {CHANNEL_INVITE_LINK}"

CCXT_EXCHANGE_ID = os.getenv("CCXT_EXCHANGE", "binance")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "25"))
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "500"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
TRADES_FILE = os.getenv("TRADES_FILE", "trades_v4.json")
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL")) if os.getenv("INITIAL_CAPITAL") else None
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # if provided, sets webhook mode

DAILY_NEWS_HOUR = int(os.getenv("DAILY_NEWS_HOUR", "10"))  # UTC hour to post news
NEWS_FEEDS = os.getenv("NEWS_FEEDS", 
    "https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss,https://cryptonews.com/news/feed") \
    .split(",")

CLOSE_ON_FIRST_TP = os.getenv("CLOSE_ON_FIRST_TP", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crypto-bot")

# Load saved trades / monitoring list
_trades = load_json(TRADES_FILE, default=[])

# ---------- Utilities / indicators ----------
def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_atr(df, period=14):
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

# ---------- Market data ----------
def _get_ccxt_exchange():
    exchange_cls = getattr(ccxt, CCXT_EXCHANGE_ID)
    ex = exchange_cls({"enableRateLimit": True})
    try:
        ex.load_markets()
    except Exception:
        pass
    return ex

def fetch_ohlcv_ccxt(symbol_ccxt: str, timeframe='1h', limit=OHLCV_LIMIT):
    ex = _get_ccxt_exchange()
    try:
        ohlcv = ex.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, limit=limit)
    except Exception as e:
        logger.warning("ccxt fetch failed for %s %s: %s", symbol_ccxt, timeframe, e)
        return None
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

# ---------- Analysis ----------
def analyze_symbol(tv_symbol: str, timeframes=['15m','1h','4h']):
    if ':' in tv_symbol:
        exchange, symbol = tv_symbol.split(':', 1)
    else:
        exchange = CCXT_EXCHANGE_ID.upper()
        symbol = tv_symbol
    symbol_upper = symbol.upper()
    ccxt_symbol = None
    for suf in ['USDT','USD','BTC','ETH','EUR']:
        if symbol_upper.endswith(suf):
            base = symbol_upper[:-len(suf)]
            ccxt_symbol = f"{base}/{suf}"
            break
    if ccxt_symbol is None:
        ccxt_symbol = f"{symbol_upper}/USDT"

    results = {}
    last_price = None
    atr_val = None

    for tf in timeframes:
        df = None
        try:
            df = fetch_ohlcv_ccxt(ccxt_symbol, timeframe=tf, limit=OHLCV_LIMIT)
        except Exception:
            df = None
        if df is None or df.empty:
            try:
                handler = TA_Handler(symbol=symbol, screener="crypto", exchange=exchange.upper(), interval=Interval.INTERVAL_1_HOUR)
                ta = handler.get_analysis()
                indicators = ta.indicators
                price = indicators.get("close") or indicators.get("last") or indicators.get("open")
                last_price = float(price) if price else last_price
            except Exception:
                pass
            continue

        close = df['close']
        ema50 = compute_ema(close, 50)
        ema200 = compute_ema(close, 200)
        rsi = compute_rsi(close, 14)
        macd, macd_signal, macd_hist = compute_macd(close)
        atr = compute_atr(df, ATR_PERIOD)
        atr_val = float(atr.iloc[-1]) if atr is not None else atr_val
        last_price = float(close.iloc[-1])

        tf_signal = {}
        tf_signal['ema_trend'] = 'bull' if ema50.iloc[-1] > ema200.iloc[-1] else 'bear'
        tf_signal['rsi'] = float(rsi.iloc[-1])
        tf_signal['rsi_signal'] = 'oversold' if rsi.iloc[-1] < 30 else ('overbought' if rsi.iloc[-1] > 70 else 'neutral')
        tf_signal['macd_hist'] = float(macd_hist.iloc[-1])
        tf_signal['macd_signal'] = 'bull' if macd_hist.iloc[-1] > 0 else 'bear'
        tf_signal['volume'] = float(df['vol'].iloc[-1]) if 'vol' in df.columns else None
        tf_signal['price'] = float(close.iloc[-1])
        results[tf] = tf_signal

    # scoring
    bulls = 0; bears = 0
    for tf, r in results.items():
        score = 0
        if r['ema_trend']=='bull': score += 1
        if r['macd_signal']=='bull': score += 1
        if r['rsi'] < 60: score += 1
        if score >= 2: bulls += 1
        else: bears += 1

    side = 'wait'
    if bulls > bears: side = 'long'
    elif bears > bulls: side = 'short'

    price = last_price
    atr_effective = atr_val if atr_val else (price * 0.01 if price else 0.0)
    atr_mult = 1.5 * atr_effective if atr_effective else max(0.01*price if price else 0.0, 0.0001)

    if price is None:
        return {"error":"Could not fetch price", "symbol": tv_symbol}

    if side == 'long':
        entry = price
        sl = price - atr_mult
        risk = entry - sl
        tps = [entry + risk * r for r in [1, 1.5, 2, 3, 5]]
    elif side == 'short':
        entry = price
        sl = price + atr_mult
        risk = sl - entry
        tps = [entry - risk * r for r in [1, 1.5, 2, 3, 5]]
    else:
        entry = price; sl = price - atr_effective
        tps = [price + price * pct for pct in [0.005, 0.01, 0.02]]

    def fmt(x):
        try:
            if x is None: return None
            x = float(x)
            if x >= 10: return round(x,3)
            if x >= 1: return round(x,4)
            return round(x,6)
        except:
            return x

    analysis = {
        "symbol": tv_symbol,
        "ccxt_symbol": ccxt_symbol,
        "side": side,
        "price": fmt(price),
        "atr": fmt(atr_val),
        "entry": fmt(entry),
        "sl": fmt(sl),
        "tps": [fmt(x) for x in tps][:5],
        "mtf": results,
    }
    analysis["advice"] = generate_advice(analysis)
    return analysis

def generate_advice(analysis):
    side = analysis.get("side", "wait")
    adv = []
    if side == "long":
        adv.append("Ride the momentum but keep an eye on higher-timeframe divergences.")
        adv.append("Consider scaling out at TP1 and TP2 to lock profits.")
    elif side == "short":
        adv.append("Look for rejection at resistance and manage risk closely.")
    else:
        adv.append("Market unclear — avoid new positions until confirmation.")
    try:
        mtf = analysis.get("mtf", {})
        if mtf:
            highest = sorted(mtf.keys(), key=lambda x: int(x.replace('m','').replace('h','')))
            if highest:
                r = mtf[highest[-1]].get("rsi")
                if r and r < 35:
                    adv.append("Higher timeframe RSI low — watch for oversold bounce.")
                if r and r > 65:
                    adv.append("Higher timeframe RSI high — be cautious of pullback.")
    except:
        pass
    return "\n".join(f"- {a}" for a in adv)

# ---------- Chart plotting ----------
def plot_levels_image(analysis_result, width=1400, height=800):
    df = None
    if analysis_result.get("ohlcv"):
        try:
            df = pd.DataFrame(analysis_result["ohlcv"])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        except Exception:
            df = None
    if df is None:
        try:
            df = fetch_ohlcv_ccxt(analysis_result['ccxt_symbol'], timeframe='1h', limit=200)
        except Exception:
            df = None

    entry = analysis_result.get('entry'); sl = analysis_result.get('sl'); tps = analysis_result.get('tps', [])
    side = analysis_result.get('side','WAIT').upper(); symbol = analysis_result.get('symbol')

    plt.rcParams.update({'font.size':12})
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

    if df is not None and len(df) > 1:
        ax.plot(df.index, df['close'], linewidth=1.2, label='Close')
        ax.fill_between(df.index, df['close'], alpha=0.02)
        ax.set_xlim(df.index[-min(len(df),200)], df.index[-1])
    else:
        now = pd.Timestamp.utcnow()
        ax.plot([now], [entry], marker='o')

    if entry is not None:
        ax.axhline(entry, linestyle='--', linewidth=1.5, label=f'Entry @ {entry}')
    if sl is not None:
        box_top = max(entry, sl) if side == 'SHORT' else entry
        box_bottom = min(entry, sl) if side == 'SHORT' else sl
        if box_top == box_bottom:
            box_top = entry + 0.002 * entry
            box_bottom = entry - 0.002 * entry
        ax.add_patch(plt.Rectangle((0.01, box_bottom), 0.98, box_top - box_bottom,
                                   transform=ax.get_xaxis_transform(), alpha=0.18, color='red', label='Stop Loss Area'))

    for i, tp in enumerate(tps):
        ax.axhline(tp, linestyle='-', linewidth=1.2, alpha=0.95, label=f'TP{i+1} @ {tp}')
        ax.add_patch(plt.Rectangle((0.85, tp - 0.0006*tp), 0.15, 0.0012*tp,
                                   transform=ax.get_xaxis_transform(), alpha=0.22, color='green'))
        ax.text(0.99, tp, f"TP{i+1}", transform=ax.get_xaxis_transform(), ha='right', va='center', color='green', fontsize=10)

    ax.text(0.01, 0.98, f"Side: {side}", transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='black', alpha=0.6, pad=6), color='white')

    ax.set_title(f"{symbol} — Entry:{entry} SL:{sl}")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle=':', linewidth=0.4)
    ax.legend(loc='upper left', fontsize='small', ncol=1)

    advice = analysis_result.get('advice','')
    if advice:
        ax.text(0.01, -0.12, advice, transform=ax.transAxes, fontsize=10, va='top')

    buf = io.BytesIO()
    plt.tight_layout(rect=[0,0.02,1,0.95])
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------- Persist & monitoring ----------
def add_trade_to_watch(analysis, requested_by_chat_id=None):
    global _trades
    item = {
        "id": f"{analysis.get('symbol')}_{int(time.time())}",
        "symbol": analysis.get("symbol"),
        "ccxt_symbol": analysis.get("ccxt_symbol"),
        "side": analysis.get("side"),
        "entry": analysis.get("entry"),
        "sl": analysis.get("sl"),
        "tps": analysis.get("tps", []),
        "resolved": False,
        "notified_tps": [],
        "created_at": str(datetime.utcnow()),
        "requested_by": requested_by_chat_id
    }
    _trades.append(item)
    save_json(TRADES_FILE, _trades)
    return item

def mark_trade_resolved(trade_id):
    global _trades
    for t in _trades:
        if t.get("id") == trade_id:
            t["resolved"] = True
    save_json(TRADES_FILE, _trades)

# ---------- News fetcher ----------
async def fetch_and_post_daily_news(app):
    """
    Runs daily at DAILY_NEWS_HOUR (UTC) and posts a brief digest to channel/admin.
    Uses RSS feeds from NEWS_FEEDS (comma-separated).
    """
    logger.info("Daily news poster started (hour %s UTC)", DAILY_NEWS_HOUR)
    while True:
        now = datetime.utcnow()
        # compute seconds until the next run at DAILY_NEWS_HOUR
        run_time = datetime(now.year, now.month, now.day, DAILY_NEWS_HOUR)
        if run_time <= now:
            run_time = run_time + timedelta(days=1)
        wait_seconds = (run_time - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        try:
            items = []
            for feed in NEWS_FEEDS:
                try:
                    d = feedparser.parse(feed)
                    for e in d.entries[:4]:  # top 4 from each feed
                        title = getattr(e, "title", "")
                        link = getattr(e, "link", "")
                        published = getattr(e, "published", "") or getattr(e, "updated", "")
                        items.append((published, title, link))
                except Exception as fe:
                    logger.exception("feed parse error: %s", fe)
            # sort by published date if possible
            def parse_date(x):
                try:
                    return dateutil.parser.parse(x[0])
                except:
                    return datetime.utcnow()
            items = sorted(items, key=parse_date, reverse=True)[:8]
            if not items:
                continue
            text = "🗞️ Daily Crypto News Digest\n\n"
            for pub, title, link in items:
                text += f"- {title}\n  {link}\n"
            text += CHANNEL_FOOTER
            # send to channel (if set) and admin
            if CHANNEL_CHAT_ID:
                try:
                    await app.bot.send_message(CHANNEL_CHAT_ID, text)
                except Exception:
                    logger.exception("Failed to post news to channel")
            if ADMIN_USER_ID:
                try:
                    await app.bot.send_message(ADMIN_USER_ID, text)
                except Exception:
                    logger.exception("Failed to post news to admin")
        except Exception as e:
            logger.exception("Daily news exception: %s", e)
            await asyncio.sleep(60)

# background monitor - polls prices and notifies on TP/SL hits
async def background_monitor(app):
    logger.info("Background monitor started, polling every %s seconds", POLL_INTERVAL)
    while True:
        try:
            trades = load_json(TRADES_FILE, default=_trades)
            if not trades:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            ex = _get_ccxt_exchange()
            for t in trades:
                if t.get("resolved"):
                    continue
                ccxt_sym = t.get("ccxt_symbol")
                requester = t.get("requested_by")
                try:
                    ticker = ex.fetch_ticker(ccxt_sym)
                    last = float(ticker.get("last"))
                except Exception:
                    continue
                side = t.get("side","wait")
                sl = t.get("sl")
                tps = t.get("tps",[])
                # SL checks
                if sl is not None:
                    if side.lower() == "long" and last <= sl:
                        text = f"⚠️ SL hit for {t['symbol']} at {nice(last)}\nProfessional note: Review your risk and journaling this trade.\n{CHANNEL_FOOTER}"
                        # notify requester
                        try:
                            if requester:
                                await app.bot.send_message(requester, text)
                        except Exception:
                            pass
                        # channel & admin
                        if CHANNEL_CHAT_ID:
                            try:
                                await app.bot.send_message(CHANNEL_CHAT_ID, text)
                            except Exception:
                                pass
                        if ADMIN_USER_ID:
                            try:
                                await app.bot.send_message(ADMIN_USER_ID, text)
                            except Exception:
                                pass
                        t['resolved'] = True
                        save_json(TRADES_FILE, trades)
                        continue
                    if side.lower() == "short" and last >= sl:
                        text = f"⚠️ SL hit for {t['symbol']} at {nice(last)}\nProfessional note: Re-evaluate entry & position sizing.\n{CHANNEL_FOOTER}"
                        try:
                            if requester:
                                await app.bot.send_message(requester, text)
                        except Exception:
                            pass
                        if CHANNEL_CHAT_ID:
                            try:
                                await app.bot.send_message(CHANNEL_CHAT_ID, text)
                            except Exception:
                                pass
                        if ADMIN_USER_ID:
                            try:
                                await app.bot.send_message(ADMIN_USER_ID, text)
                            except Exception:
                                pass
                        t['resolved'] = True
                        save_json(TRADES_FILE, trades)
                        continue
                # TP checks
                for idx, tp in enumerate(tps):
                    if tp in t.get("notified_tps", []):
                        continue
                    if side.lower() == "long" and last >= tp:
                        msg = f"🎉 TP{idx+1} hit for {t['symbol']} at {nice(last)}\nCongrats! Tip: consider trailing SL to secure profits.\n{CHANNEL_FOOTER}"
                        try:
                            if requester:
                                await app.bot.send_message(requester, msg)
                        except Exception:
                            pass
                        if CHANNEL_CHAT_ID:
                            try:
                                await app.bot.send_message(CHANNEL_CHAT_ID, msg)
                            except Exception:
                                pass
                        if ADMIN_USER_ID:
                            try:
                                await app.bot.send_message(ADMIN_USER_ID, msg)
                            except Exception:
                                pass
                        t.setdefault("notified_tps", []).append(tp)
                        if CLOSE_ON_FIRST_TP:
                            t['resolved'] = True
                        save_json(TRADES_FILE, trades)
                    if side.lower() == "short" and last <= tp:
                        msg = f"🎉 TP{idx+1} hit for {t['symbol']} at {nice(last)}\nCongrats! Consider partial profits.\n{CHANNEL_FOOTER}"
                        try:
                            if requester:
                                await app.bot.send_message(requester, msg)
                        except Exception:
                            pass
                        if CHANNEL_CHAT_ID:
                            try:
                                await app.bot.send_message(CHANNEL_CHAT_ID, msg)
                            except Exception:
                                pass
                        if ADMIN_USER_ID:
                            try:
                                await app.bot.send_message(ADMIN_USER_ID, msg)
                            except Exception:
                                pass
                        t.setdefault("notified_tps", []).append(tp)
                        if CLOSE_ON_FIRST_TP:
                            t['resolved'] = True
                        save_json(TRADES_FILE, trades)
            await asyncio.sleep(POLL_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("Background monitor error: %s", e)
            await asyncio.sleep(POLL_INTERVAL)

# ---------- Telegram handlers ----------
# For /risk we will use a simple two-step Conversation: ENTRY -> SL -> compute
RISK_ENTRY, RISK_SL = range(2)

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "Salam! Use /analyze SYMBOL or /news to see latest digest."
    text += CHANNEL_FOOTER
    await update.message.reply_text(text)

async def analyze_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /analyze SYMBOL (eg. /analyze BTCUSDT)")
        return
    symbol = context.args[0].strip().upper()
    await update.message.chat.send_action("typing")
    try:
        analysis = analyze_symbol(symbol, timeframes=['15m','1h','4h'])
    except Exception as e:
        logger.exception("analyze error: %s", e)
        await update.message.reply_text("Error analyzing symbol." + CHANNEL_FOOTER)
        return

    if analysis.get("error"):
        await update.message.reply_text(f"Error: {analysis['error']}" + CHANNEL_FOOTER)
        return

    # build message
    lines = []
    lines.append(f"🔎 Analysis: {symbol}")
    lines.append(f"Side: *{analysis['side'].upper()}*")
    lines.append(f"Price: `{analysis['price']}`")
    lines.append(f"Entry: `{analysis['entry']}`  SL: `{analysis['sl']}`")
    for i, tp in enumerate(analysis.get('tps',[])):
        lines.append(f"TP{i+1}: `{tp}`")
    lines.append("\nAdvice:")
    lines.append(analysis.get("advice",""))
    msg = "\n".join(lines) + CHANNEL_FOOTER

    # send chart
    try:
        buf = plot_levels_image(analysis)
        buf.seek(0)
        await update.message.reply_photo(photo=buf, caption=msg, parse_mode="Markdown")
    except Exception as e:
        logger.exception("plot/send error: %s", e)
        await update.message.reply_text(msg, parse_mode="Markdown")

    # add to watchlist with requester's chat id
    watch = add_trade_to_watch(analysis, requested_by_chat_id=chat_id)
    await update.message.reply_text(f"Added to watchlist (id: {watch['id']}). I'll notify you and the channel on TP/SL hits." + CHANNEL_FOOTER)

async def news_now_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # allow manual trigger of news digest
    items = []
    for feed in NEWS_FEEDS:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:4]:
                title = getattr(e, "title", "")
                link = getattr(e, "link", "")
                items.append((title, link))
        except Exception:
            pass
    if not items:
        await update.message.reply_text("No news found." + CHANNEL_FOOTER)
        return
    text = "🗞️ Crypto News (manual)\n\n"
    for title, link in items[:8]:
        text += f"- {title}\n  {link}\n"
    text += CHANNEL_FOOTER
    await update.message.reply_text(text)

# /risk conversation
async def risk_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me ENTRY price (single number).")
    return RISK_ENTRY

async def risk_entry_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        entry = float(update.message.text.strip())
        context.user_data['entry'] = entry
        await update.message.reply_text("Now send SL price (single number).")
        return RISK_SL
    except:
        await update.message.reply_text("Invalid number. Send ENTRY price like: 28000")
        return RISK_ENTRY

async def risk_sl_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sl = float(update.message.text.strip())
        entry = context.user_data.get('entry')
        if entry is None:
            await update.message.reply_text("Missing entry. Restart /risk.")
            return ConversationHandler.END
        # compute size using INITIAL_CAPITAL if available
        if INITIAL_CAPITAL:
            qty = compute_position_size(INITIAL_CAPITAL, 1.0, entry, sl)  # default 1% risk
            await update.message.reply_text(f"Estimated qty at 1% risk (balance {INITIAL_CAPITAL}): {nice(qty)}" + CHANNEL_FOOTER)
        else:
            await update.message.reply_text(f"Entry: {entry} SL: {sl}. Provide balance like /riskbal BALANCE to compute." + CHANNEL_FOOTER)
    except:
        await update.message.reply_text("Invalid SL. Send SL price like: 27800")
    return ConversationHandler.END

async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caller = update.effective_user.id
    if ADMIN_USER_ID and caller != ADMIN_USER_ID:
        await update.message.reply_text("Only admin can view full history." + CHANNEL_FOOTER)
        return
    trades = load_json(TRADES_FILE, default=[])
    if not trades:
        await update.message.reply_text("No trades watched yet." + CHANNEL_FOOTER)
        return
    text = "Watchlist:\n"
    for t in trades[-40:]:
        text += f"- {t['id']} {t['symbol']} side:{t['side']} resolved:{t.get('resolved')} notified:{len(t.get('notified_tps',[]))}\n"
    text += CHANNEL_FOOTER
    await update.message.reply_text(text)

# ---------- main ----------
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("analyze", analyze_handler))
    app.add_handler(CommandHandler("news", news_now_handler))
    app.add_handler(CommandHandler("history", history_handler))

    risk_conv = ConversationHandler(
        entry_points=[CommandHandler('risk', risk_start)],
        states={
            RISK_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, risk_entry_received)],
            RISK_SL: [MessageHandler(filters.TEXT & ~filters.COMMAND, risk_sl_received)],
        },
        fallbacks=[]
    )
    app.add_handler(risk_conv)

    # start background monitor and news poster
    app.create_task(background_monitor(app))
    app.create_task(fetch_and_post_daily_news(app))

    # webhook vs polling
    if WEBHOOK_URL:
        webhook_path = f"/webhook/{BOT_TOKEN}"
        port = int(os.getenv("PORT", "8080"))
        await app.start()
        await app.updater.start_webhook(listen="0.0.0.0", port=port, webhook_url=WEBHOOK_URL+webhook_path)
        logger.info("Webhook started at %s", WEBHOOK_URL+webhook_path)
        await asyncio.Event().wait()
    else:
        await app.start()
        logger.info("Bot started (polling).")
        await app.updater.start_polling()
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
