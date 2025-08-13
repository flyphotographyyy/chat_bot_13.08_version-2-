# -*- coding: utf-8 -*-
# Stock Signals PRO – Pro+ Patch4
# DATE: 2025-08-12
#
# Новото в Patch4 (без промени по дизайна/лейаута):
# 1) Портфейлен walk-forward backtest (train 18m / test 6m, rolling) – equal-weight top-K, 
#    седмично ребалансиране, транзакционни разходи и slippage, OOS метрики (CAGR/MaxDD/Sharpe/Trades/Turnover).
# 2) Auto-EV calibration веднъж седмично върху SP100 + твоя watchlist (минимум n≥100 на бин; fallback към 60–79).
# 3) Фундаментите: реален TTM dividend от yfinance.dividends (ако има), outlier guard (игнориране извън [0.1;12]%).
#    Sentiment: по-слабо влияние на интрадей (30m) чрез news.intraday_weight.
#
# Забележка: Дизайнът остава същият. Добавих само 1 caption ред за портфейлния OOS резултат (както и преди за SPY).

import os, re, json, time, math, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st

import yfinance as yf
try:
    from pandas_datareader import data as pdr  # optional Stooq fallback
except Exception:
    pdr = None
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None
try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
try:
    import yaml  # optional (за config.yaml)
except Exception:
    yaml = None

APP_TITLE = "📈 Stock Signals PRO – Enhanced Multi-Source Analysis"
HOME = Path.home()
WATCHLIST_FILE = HOME / "stock_signals_watchlist.json"
SETTINGS_FILE  = HOME / "stock_signals_settings.json"
CALIB_FILE     = HOME / "signals_ev_calibration.json"
PORTFOLIO_FILE = HOME / "portfolio_state.json"
for p in [WATCHLIST_FILE, SETTINGS_FILE, CALIB_FILE, PORTFOLIO_FILE]:
    p.parent.mkdir(parents=True, exist_ok=True)
# копие на калибрацията от репото към HOME (ако липсва)
REPO_CALIB = Path.cwd() / "signals_ev_calibration.json"
if REPO_CALIB.exists() and not CALIB_FILE.exists():
    CALIB_FILE.write_text(REPO_CALIB.read_text(encoding="utf-8"), encoding="utf-8")

# -------------------- Config (optional YAML/JSON override) --------------------
DEFAULT_CFG = {
    "risk_profile": "balanced",
    "lookback_days": 120,
    "news_days": 7,
    "show_charts": True,
    "auto_refresh": True,
    # EV / calibration
    "ev": {"horizon_days": 10, "min_n": 100, "fallback_bin": "60-79", "precalib_universe": "SP100", "rebuild_days": 7},
    # Walk-forward (months)
    "wf": {"train_months": 18, "test_months": 6, "top_k": 8, "rebalance": "W-MON", "cost_bps": 10, "slip_bps": 10},
    # Regime/VIX
    "regime": {"vix_elevated": 20.0, "vix_high": 25.0},
    # Volume threshold bounds
    "volume": {"min_base": 1.1, "max_base": 1.5},
    # News caps (+ по-ниска тежест на интрадей)
    "news": {"sentiment_cap_points": 7, "opinion_stoplist": ["opinion","rumor","speculative","blog","downgrade","upgrade","price target","short seller"], "intraday_weight": 0.25},
    # Risk manager
    "risk": {"max_positions": 8, "sector_cap_pct": 0.30, "daily_risk_budget": 0.15},
    # Network
    "net": {"sec_rps": 5, "timeout": 12, "retries": 3, "backoff": 0.75, "ua": "SignalsPro/1.0 (contact: youremail@example.com)"}
}
CONFIG_PATHS = [Path.cwd()/"config.yaml", Path.cwd()/"config.json"]

def load_config() -> Dict:
    cfg = DEFAULT_CFG.copy()
    try:
        for p in CONFIG_PATHS:
            if p.exists():
                if p.suffix == ".yaml" and yaml is not None:
                    user = yaml.safe_load(p.read_text(encoding="utf-8"))
                elif p.suffix == ".json":
                    user = json.loads(p.read_text(encoding="utf-8"))
                else:
                    user = {}
                if isinstance(user, dict):
                    def deep_merge(a,b):
                        out=a.copy()
                        for k,v in b.items():
                            if isinstance(v,dict) and isinstance(out.get(k),dict): out[k]=deep_merge(out[k],v)
                            else: out[k]=v
                        return out
                    cfg = deep_merge(cfg, user)
                break
    except Exception:
        pass
    return cfg

CFG = load_config()

# -------------------- Market profiles & SP100 --------------------
MARKETS = {
    "US – NYSE/Nasdaq (09:30–16:00 ET)": {"tz": "America/New_York", "open": (9,30),  "close": (16,0),  "cal": "XNYS"},
    "Germany – XETRA (09:00–17:30 DE)":  {"tz": "Europe/Berlin",    "open": (9,0),   "close": (17,30), "cal": "XETR"},
    "UK – LSE (08:00–16:30 UK)":         {"tz": "Europe/London",    "open": (8,0),   "close": (16,30), "cal": "XLON"},
    "France – Euronext Paris (09:00–17:30 FR)": {"tz": "Europe/Paris", "open": (9,0), "close": (17,30), "cal": "XPAR"}
}

SP100 = [
    "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","META","BRK-B","UNH","XOM","JNJ","JPM","V","PG","CVX","HD","LLY","MA","ABBV","PFE","BAC","KO","PEP","COST","AVGO","WMT","DIS","CSCO","MCD","ACN","TMO","ABT","WFC","ADBE","NFLX","CRM","CMCSA","DHR","NKE","TXN","NEE","LIN","ORCL","PM","AMD","HON","INTC","LOW","AMGN","UPS","SBUX","RTX","QCOM","IBM","MS","MDT","INTU","CVS","CAT","GS","SPGI","PLD","GE","BLK","BA","BKNG","SCHW","LMT","DE","AMAT","AXP","ADI","C","ELV","UNP","T","NOW","MU","USB","SYK","ISRG","GILD","MDLZ","TJX","PYPL","BDX","SO","ZTS","REGN","MMC","CB","CI","ADP","VRTX","PGR","TGT","PNC","MO","DUK","EQIX","APD","CL","ICE","SHW"
]

# -------------------- Central HTTP (retries + rate-limit) --------------------
_last_call_times: Dict[str, List[float]] = {"sec": []}

def _rate_limit(domain_key: str, rps: int):
    if rps <= 0: return
    now = time.time(); window = 1.0
    arr = _last_call_times.setdefault(domain_key, [])
    _last_call_times[domain_key] = [t for t in arr if now - t < window]
    while len(_last_call_times[domain_key]) >= rps:
        time.sleep(0.02)
        now = time.time()
        _last_call_times[domain_key] = [t for t in _last_call_times[domain_key] if now - t < window]
    _last_call_times[domain_key].append(now)

def http_get(url: str, timeout: Optional[int]=None, retries: Optional[int]=None, backoff: Optional[float]=None, headers: Optional[Dict]=None) -> Optional[requests.Response]:
    timeout = timeout if timeout is not None else CFG['net']['timeout']
    retries = retries if retries is not None else CFG['net']['retries']
    backoff = backoff if backoff is not None else CFG['net']['backoff']
    headers = headers or {}
    headers.setdefault('User-Agent', CFG['net']['ua'])
    domain_key = 'sec' if 'sec.gov' in url else 'default'
    if domain_key == 'sec': _rate_limit('sec', CFG['net']['sec_rps'])
    for i in range(retries+1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200: return r
        except Exception:
            pass
        time.sleep(backoff * (2 ** i))
    return None

# -------------------- Helpers --------------------
def finite(x) -> bool: return x is not None and np.isfinite(x)

def now_tz(tz_name: str) -> dt.datetime: return dt.datetime.now(pytz.timezone(tz_name))

# Dividend sanitization: TTM from history, guard range
@st.cache_data(ttl=24*3600, show_spinner=False)
def div_ttm_percent(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        price = float(t.fast_info.get("last_price") or np.nan)
        dv = t.dividends
        if dv is not None and len(dv)>0 and finite(price) and price>0:
            yttm = float(dv.tail(4).sum() / price * 100)
            yttm = max(0.0, min(yttm, 20.0))
            if 0.1 <= yttm <= 12.0:
                return round(yttm, 2)
    except Exception:
        pass
    return None

def _normalize_dividend(div) -> Optional[float]:
    try:
        if div is None: return None
        val = float(div)
        if val < 0: return None
        val = val * 100 if val <= 1 else val
        return round(min(val, 20.0), 2)
    except Exception:
        return None

# -------------------- Data fetch (Yahoo + optional Stooq) --------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(ticker: str, days: int, interval: str = "1d") -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        if interval == "30m": df = stock.history(period="60d", interval="30m")
        else: df = stock.history(period=f"{days}d", interval="1d")
        if df is not None and not df.empty: return df
    except Exception: pass
    if interval == "1d" and pdr is not None:
        try:
            start = dt.date.today() - dt.timedelta(days=days + 30)
            end   = dt.date.today()
            d = pdr.DataReader(ticker, "stooq", start=start, end=end)
            if d is not None and not d.empty:
                d = d.sort_index()
                for c in ["Open","High","Low","Close","Volume"]:
                    if c not in d.columns: d[c] = np.nan
                return d[["Open","High","Low","Close","Volume"]]
        except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_fast_info(ticker: str) -> Dict:
    try:
        fi = yf.Ticker(ticker).fast_info or {}
        return {"last_price": fi.get("last_price"), "market_cap": fi.get("market_cap"), "beta": fi.get("beta")}
    except Exception:
        return {}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fundamentals(ticker: str) -> Dict:
    # SEC приоритетно
    sec = fetch_sec_facts(ticker)
    if sec and (sec.get('ttm_eps') is not None):
        pe = (sec['price']/sec['ttm_eps']) if finite(sec['price']) and finite(sec['ttm_eps']) and sec['ttm_eps']!=0 else None
        dv_ttm = div_ttm_percent(ticker)
        dv = dv_ttm if dv_ttm is not None else _normalize_dividend(sec.get('dividend_yield'))
        if isinstance(dv,(int,float)) and (dv<0.1 or dv>12.0): dv=None
        return {"trailing_pe": pe, "dividend_yield": dv, "sector": sec.get('sector'), "industry": sec.get('industry')}
    # Yahoo fallback
    info = {}
    try:
        t = yf.Ticker(ticker)
        try: info_dict = t.get_info()
        except Exception: info_dict = getattr(t, "info", {}) or {}
        if info_dict:
            pe = info_dict.get("trailingPE") or info_dict.get("trailing_pe") or info_dict.get("peRatio")
            dv_ttm = div_ttm_percent(ticker)
            if dv_ttm is None:
                div_raw = info_dict.get("dividendYield") or info_dict.get("trailingAnnualDividendYield") or info_dict.get("yield")
                dv = _normalize_dividend(div_raw)
            else:
                dv = dv_ttm
            if isinstance(dv,(int,float)) and (dv<0.1 or dv>12.0): dv=None
            info = {"trailing_pe": pe, "dividend_yield": dv, "sector": info_dict.get("sector"), "industry": info_dict.get("industry")}
    except Exception: pass
    return info

# -------------------- SEC Company Facts/Submissions (без токени) --------------------
@st.cache_data(ttl=72*3600, show_spinner=False)
def fetch_sec_facts(ticker: str) -> Optional[Dict]:
    try:
        r = http_get(f"https://data.sec.gov/submissions/CIK{ticker}.json")
        if r is None or r.status_code != 200:
            return None
        data = r.json()
        cik = data.get('cik') or data.get('cik_str') or data.get('CIK')
        if not cik: return None
        cik = str(cik).zfill(10)
        facts_r = http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
        facts = facts_r.json() if facts_r is not None and facts_r.status_code==200 else {}
        ttm_eps = None
        try:
            eps_obj = facts['facts']['us-gaap'].get('EarningsPerShareDiluted') or facts['facts']['us-gaap'].get('EarningsPerShareBasic')
            if eps_obj and 'units' in eps_obj:
                for unit, arr in eps_obj['units'].items():
                    vals = sorted(arr, key=lambda x: x.get('end',''))[-4:]
                    s = [v.get('val') for v in vals if isinstance(v.get('val'), (int,float))]
                    if len(s)>=4:
                        ttm_eps = float(np.sum(s[-4:]))
                        break
        except Exception:
            pass
        sector = data.get('sicDescription') or None
        price = float(yf.Ticker(ticker).fast_info.get('last_price') or np.nan)
        return {"ttm_eps": ttm_eps, "sector": sector, "industry": None, "price": price, "dividend_yield": None}
    except Exception:
        return None

# -------------------- Indicators --------------------
def ema(series: pd.Series, span: int) -> pd.Series: return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean(); roll_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-9); return 100 - (100/(1+rs))

def macd(series: pd.Series, fast=12, slow=26, sig=9):
    m = ema(series, fast) - ema(series, slow); s = ema(m, sig); h = m - s; return m, s, h

def _true_range(h, l, c): pc = c.shift(1); return pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def atr(h, l, c, period=14): return _true_range(h,l,c).ewm(alpha=1/period, adjust=False).mean()

def bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = series.rolling(period).mean(); sd = series.rolling(period).std(); return mid + std_dev*sd, mid, mid - std_dev*sd

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    close = df['Close'].astype(float); high=df.get('High',close); low=df.get('Low',close); vol=df.get('Volume', pd.Series(1_000_000,index=df.index))
    for p in [20,50,200]: df[f'SMA{p}'] = close.rolling(p).mean()
    df['RSI14'] = rsi(close,14)
    macd_line, macd_sig, macd_hist = macd(close)
    df['MACD']=macd_line; df['MACD_SIG']=macd_sig; df['MACD_HIST']=macd_hist
    up, mid, lo = bollinger_bands(close)
    df['BB_Upper']=up; df['BB_Middle']=mid; df['BB_Lower']=lo
    width = (up - lo).replace([0,np.inf,-np.inf], np.nan)
    df['BB_Position'] = np.clip(((close - lo) / width) * 100, 0, 100)
    df['Volume_SMA20'] = vol.rolling(20).mean(); df['Volume_Ratio'] = (vol / df['Volume_SMA20']).replace([np.inf,-np.inf], np.nan)
    df['ATR'] = atr(high, low, close, 14)
    df['Volatility'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    for p in [5,20]: df[f'Return_{p}d'] = close.pct_change(p) * 100
    window = min(len(df), 252)
    df['HI52'] = close.rolling(window).max(); df['LO52'] = close.rolling(window).min()
    return df

# -------------------- Candle handling & market status --------------------
@st.cache_data(ttl=60, show_spinner=False)
def is_market_open_raw(profile_key: str) -> bool:
    prof = MARKETS.get(profile_key); 
    if not prof: return False
    tz = pytz.timezone(prof['tz']); now = dt.datetime.now(tz)
    if mcal and prof.get('cal'):
        try:
            cal = mcal.get_calendar(prof['cal']); sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty: return False
            o = sched.iloc[0]['market_open'].tz_convert(tz); c = sched.iloc[0]['market_close'].tz_convert(tz)
            return o <= now < c
        except Exception: pass
    if now.weekday()>4: return False
    (oh,om),(ch,cm) = prof['open'], prof['close']
    o = now.replace(hour=oh,minute=om,second=0,microsecond=0); c = now.replace(hour=ch,minute=cm,second=0,microsecond=0)
    return o <= now < c

@st.cache_data(ttl=900, show_spinner=False)
def trim_to_closed(df: pd.DataFrame, interval: str, market_key: str) -> pd.DataFrame:
    if df is None or df.empty: return df
    try: open_now = is_market_open_raw(market_key)
    except Exception: open_now = False
    if not isinstance(df.index, pd.DatetimeIndex): return df
    if interval == '30m' and open_now and len(df)>1: return df.iloc[:-1]
    if interval == '1d':
        prof = MARKETS.get(market_key)
        if prof:
            tz = pytz.timezone(prof['tz']); now_local = dt.datetime.now(tz)
            last = df.index[-1].tz_localize(tz) if df.index.tz is None else df.index[-1].tz_convert(tz)
            (oh,om),(ch,cm) = prof['open'], prof['close']
            close_today = now_local.replace(hour=ch,minute=cm,second=0,microsecond=0)
            if last.date()==now_local.date() and now_local<close_today and len(df)>1:
                return df.iloc[:-1]
    return df

# -------------------- Market regime (SPY + ^VIX) --------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_spy_vix(days: int = 400) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spy = fetch_price_history("SPY", days, "1d"); vix = fetch_price_history("^VIX", days, "1d"); return spy, vix

@st.cache_data(ttl=1800, show_spinner=False)
def current_regime() -> Dict:
    spy, vix = fetch_spy_vix(400)
    regime = {"state":"neutral","vix":"normal","thr_buy_adj":0,"thr_sell_adj":0,"atr_mult":2.0}
    try:
        spy = compute_indicators(spy)
        if not spy.empty: regime['state'] = 'bull' if spy['SMA50'].iloc[-1] > spy['SMA200'].iloc[-1] else 'bear'
    except Exception: pass
    try:
        vclose = float(vix['Close'].iloc[-1]) if not vix.empty else 18.0
        if vclose >= CFG['regime']['vix_high']:
            regime['vix'] = 'high'; regime['thr_buy_adj'] += 8; regime['atr_mult'] = 1.6
        elif vclose >= CFG['regime']['vix_elevated']:
            regime['vix'] = 'elevated'; regime['thr_buy_adj'] += 5; regime['atr_mult'] = 1.8
    except Exception: pass
    if regime['state']=='bear': regime['thr_buy_adj'] += 5; regime['thr_sell_adj'] -= 2
    return regime

# -------------------- News & Sentiment --------------------
def _clean_title(t: str) -> str: return re.sub(r"[\W_]+", " ", (t or "").lower()).strip()

NEWS_RSS = [
    "https://news.google.com/rss/search?q={q}+stock+when:{days}d&hl=en-US&gl=US&ceid=US:en",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={q}&region=US&lang=en-US",
    "https://www.prnewswire.com/rss/news-releases-list.rss?keyword={q}",
    "https://www.globenewswire.com/RssFeed/org-classic/{q}"
]

@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_items(ticker: str, days: int = 7) -> List[dict]:
    items=[]; seen=set()
    for tpl in NEWS_RSS:
        try: url = tpl.format(q=ticker, days=days)
        except Exception: url = tpl
        try:
            r = http_get(url)
            feed = feedparser.parse(r.text if r is not None else '')
            for e in feed.entries[:30]:
                title = e.title if hasattr(e,'title') else ''
                norm = _clean_title(title)
                if not norm or norm in seen: continue
                seen.add(norm)
                pub = dt.datetime(*e.published_parsed[:6]) if getattr(e,'published_parsed',None) else dt.datetime.utcnow()
                src = e.get('source',{}).get('title') or e.get('publisher') or 'RSS'
                items.append({"title": title, "source": src, "published": pub, "link": e.get('link','')})
        except Exception:
            continue
    return items

@st.cache_data(ttl=600, show_spinner=False)
def analyze_sentiment(items: List[dict]) -> Dict[str,float]:
    if not items: return {"compound":0.0,"n":0,"confidence":0.0}
    stop = set(CFG['news']['opinion_stoplist'])
    vader = SentimentIntensityAnalyzer(); now = dt.datetime.utcnow()
    by_source: Dict[str, List[float]] = {}
    for it in items:
        title = it.get('title',''); norm=_clean_title(title)
        if not norm: continue
        if any(w in norm for w in stop): continue
        age = max(0.2, (now - it.get('published', now)).total_seconds()/86400.0)
        w = float(np.exp(-age/3.0))
        s = vader.polarity_scores(title)['compound']
        if TextBlob:
            try: s = 0.7*s + 0.3*TextBlob(title).sentiment.polarity
            except Exception: pass
        by_source.setdefault(it.get('source','RSS'), []).append(s*w)
    if not by_source: return {"compound":0.0,"n":0,"confidence":0.0}
    src_scores = [np.mean(v) for v in by_source.values() if v]
    wmean = float(np.mean(src_scores)) if src_scores else 0.0
    std = float(np.std(src_scores)) if len(src_scores)>1 else 0.0
    conf = max(0.0, 1.0-std)
    return {"compound":wmean, "n": sum(len(v) for v in by_source.values()), "confidence": conf, "sources": len(by_source)}

# -------------------- EV Calibration --------------------
SCORE_BINS = [(50,59),(60,69),(70,79),(80,150)]

def _score_simple(row: pd.Series) -> int:
    s=0
    p=row.get('Close',np.nan); s20=row.get('SMA20',np.nan); s50=row.get('SMA50',np.nan); s200=row.get('SMA200',np.nan)
    rsi=row.get('RSI14',np.nan); mac=row.get('MACD',np.nan); sig=row.get('MACD_SIG',np.nan)
    bb=row.get('BB_Position',np.nan); vr=row.get('Volume_Ratio',np.nan)
    r5=row.get('Return_5d',np.nan); r20=row.get('Return_20d',np.nan)
    if all(finite(x) for x in [p,s20,s50,s200]):
        if p>s20>s50>s200: s+=20
        elif p<s20<s50<s200: s-=20
        elif p>s50: s+=8
    if finite(rsi):
        if rsi<30: s+=12
        if rsi>70: s-=12
    if all(finite(x) for x in [mac,sig]):
        if mac>sig: s+=6
        else: s-=2
    if finite(bb):
        if bb<10: s+=6
        if bb>90: s-=6
    if all(finite(x) for x in [r5,r20]):
        if r5>5 and r20>10: s+=8
        if r5<-5 and r20<-10: s-=8
    if finite(vr) and vr>1.5: s+=4
    s = int(np.interp(s, [-40,40], [0,100])); return max(0,min(100,s))

def _bin_for_score(score: int) -> str:
    for lo,hi in SCORE_BINS:
        if lo <= score <= hi: return f"{lo}-{hi}"
    return "<50"

@st.cache_data(ttl=86400, show_spinner=False)
def build_spy_for_regime(days:int=1200) -> pd.DataFrame:
    spy = fetch_price_history('SPY', days, '1d'); spy = compute_indicators(spy); return spy

@st.cache_data(ttl=86400, show_spinner=False)
def precalc_ev_if_needed(extra_universe: Optional[List[str]] = None) -> None:
    try:
        need_build = True
        if CALIB_FILE.exists():
            age_days = (dt.datetime.utcnow() - dt.datetime.utcfromtimestamp(CALIB_FILE.stat().st_mtime)).days
            need_build = age_days >= CFG['ev']['rebuild_days']
        if not need_build: return
        universe = list(SP100)
        if extra_universe:
            for t in extra_universe:
                if isinstance(t,str) and t.upper() not in universe:
                    universe.append(t.upper())
        spy = build_spy_for_regime(1200)
        calib: Dict[str, Dict[str,float]] = {}
        for t in universe:
            df = fetch_price_history(t, 1200, '1d')
            if df is None or df.empty: continue
            df = compute_indicators(df)
            close = df['Close']
            scores = df.apply(_score_simple, axis=1)
            fwd = close.pct_change(CFG['ev']['horizon_days']).shift(-CFG['ev']['horizon_days']) * 100
            data = pd.DataFrame({'score':scores, 'fwd':fwd}).dropna()
            if len(data) < 60: continue
            for idx,row in data.iterrows():
                bin_key = _bin_for_score(int(row['score']))
                try:
                    dt_i = idx
                    if dt_i not in spy.index:
                        dt_i = spy.index[spy.index.get_loc(dt_i, method='pad')]
                    reg = 'bull' if spy.loc[dt_i,'SMA50']>spy.loc[dt_i,'SMA200'] else 'bear'
                except Exception:
                    reg = 'unknown'
                key = f"{reg}|{bin_key}"
                calib.setdefault(key, {"sum":0.0, "n":0}); calib[key]["sum"] += float(row['fwd']); calib[key]["n"] += 1
        out = {k: {"mean": round(v['sum']/v['n'],4), "n": int(v['n'])} for k,v in calib.items() if v['n']>0}
        CALIB_FILE.write_text(json.dumps(out, indent=2), encoding='utf-8')
    except Exception:
        pass


def lookup_ev(regime_state: str, score: int) -> Optional[Tuple[float,int]]:
    try:
        if not CALIB_FILE.exists(): return None
        calib = json.loads(CALIB_FILE.read_text(encoding='utf-8'))
        bin_key = _bin_for_score(score); key = f"{regime_state}|{bin_key}"
        rec = calib.get(key)
        if rec and rec.get('n',0) >= CFG['ev']['min_n']:
            return float(rec['mean']), int(rec['n'])
        fb = CFG['ev']['fallback_bin']
        if fb and bin_key != fb:
            rec2 = calib.get(f"{regime_state}|{fb}")
            if rec2 and rec2.get('n',0) >= CFG['ev']['min_n']:
                return float(rec2['mean']), int(rec2['n'])
    except Exception:
        pass
    return None

# -------------------- Confirmations --------------------

def two_bar_confirmation(df: pd.DataFrame) -> Dict[str,bool]:
    if df is None or len(df)<3: return {"rsi":False,"macd":False,"price":False}
    last2 = df.iloc[-2:]
    rsi_c = all(finite(x) and x>50 for x in last2.get('RSI14', pd.Series([np.nan,np.nan])).tolist())
    macd_c = all((last2.get('MACD', pd.Series([np.nan,np.nan])).values > last2.get('MACD_SIG', pd.Series([np.nan,np.nan])).values))
    price_c = all(last2['Close'].values > last2.get('SMA20', pd.Series([np.nan,np.nan])).values)
    return {"rsi":bool(rsi_c),"macd":bool(macd_c),"price":bool(price_c)}

# -------------------- Classification --------------------

def classify_one(ticker: str, df: pd.DataFrame, risk_profile: str, market_key: str, use_news: bool = True, interval: str = "1d") -> Dict:
    if df.empty: return {"error":"No data"}
    last_ts = df.index[-1]
    if isinstance(last_ts, pd.Timestamp):
        if (dt.datetime.utcnow().date() - last_ts.date()).days > 5:
            return {"ticker":ticker,"signal":"HOLD","score":50,"confidence":50,
                    "price": float(df['Close'].iloc[-1]),
                    "reasons":["Stale price data (>5 days)"]}

    cur = df.iloc[-1]; prev = df.iloc[-2] if len(df)>=2 else cur
    fi = fetch_fast_info(ticker)
    fnd= fetch_fundamentals(ticker)
    pe = fnd.get('trailing_pe', np.nan)

    news = analyze_sentiment(fetch_news_items(ticker, CFG['news_days'])) if use_news else {}

    # dynamic volume threshold via ATR%
    atr_pc = float(cur['ATR']/cur['Close']) if all(finite(x) for x in [cur.get('ATR',np.nan), cur.get('Close',np.nan)]) else 0.02
    entry_vmin = CFG['volume']['min_base'] + (CFG['volume']['max_base']-CFG['volume']['min_base'])*min(1.0, atr_pc/0.02)

    regime = current_regime()

    signals = {"trend":0,"momentum":0,"volume":0,"sentiment":0,"fundamental":0}
    reasons=[]; confs=[]
    price=float(cur['Close']); sma20=cur.get('SMA20',np.nan); sma50=cur.get('SMA50',np.nan); sma200=cur.get('SMA200',np.nan)
    rsi14=float(cur.get('RSI14',np.nan)); mac=float(cur.get('MACD',np.nan)); sig=float(cur.get('MACD_SIG',np.nan))
    bbpos=float(cur.get('BB_Position',np.nan)); vr=float(cur.get('Volume_Ratio',np.nan))
    r5=float(cur.get('Return_5d',np.nan)); r20=float(cur.get('Return_20d',np.nan))

    # Trend & momentum
    if all(finite(x) for x in [price,sma20,sma50,sma200]):
        if price>sma20>sma50>sma200:
            signals['trend']+=20; reasons.append('Strong uptrend – price > SMA20>SMA50>SMA200'); confs.append(0.9)
        elif price<sma20<sma50<sma200:
            signals['trend']-=20; reasons.append('Strong downtrend – price < SMA20<SMA50<SMA200'); confs.append(0.9)
        elif price>sma50:
            signals['trend']+=8; reasons.append('Above medium-term trend'); confs.append(0.6)

    if finite(rsi14):
        if rsi14<30: signals['momentum']+=12; reasons.append(f'RSI oversold ({rsi14:.1f})'); confs.append(0.8)
        if rsi14>70: signals['momentum']-=12; reasons.append(f'RSI overbought ({rsi14:.1f})'); confs.append(0.8)
        if finite(prev.get('RSI14',np.nan)):
            if prev['RSI14']<50<=rsi14: signals['momentum']+=6; reasons.append('RSI crossed above 50'); confs.append(0.6)
            if prev['RSI14']>50>=rsi14: signals['momentum']-=6; reasons.append('RSI crossed below 50'); confs.append(0.6)

    if all(finite(x) for x in [mac,sig, prev.get('MACD',np.nan), prev.get('MACD_SIG',np.nan)]):
        if prev['MACD']<prev['MACD_SIG'] and mac>sig:
            signals['momentum']+=10; reasons.append('MACD bullish crossover'); confs.append(0.7)
        if prev['MACD']>prev['MACD_SIG'] and mac<sig:
            signals['momentum']-=10; reasons.append('MACD bearish crossover'); confs.append(0.7)

    if finite(bbpos):
        if bbpos<10: signals['trend']+=8; reasons.append('Near Bollinger lower band'); confs.append(0.6)
        if bbpos>90: signals['trend']-=8; reasons.append('Near Bollinger upper band'); confs.append(0.6)

    if finite(vr):
        if vr>1.5: signals['volume']+=6; reasons.append(f'High volume ({vr:.1f}× avg)'); confs.append(0.5)
        if vr<0.5: signals['volume']-=4; reasons.append('Low volume'); confs.append(0.3)

    if all(finite(x) for x in [r5,r20]):
        if r5>5 and r20>10: signals['momentum']+=12; reasons.append('Strong positive momentum (5d & 20d)'); confs.append(0.7)
        if r5<-5 and r20<-10: signals['momentum']-=12; reasons.append('Strong negative momentum (5d & 20d)'); confs.append(0.7)

    # Fundamentals
    if finite(pe):
        if pe < 15: signals['fundamental'] += 6; reasons.append(f'Low P/E ({pe:.1f})')
        if pe > 30: signals['fundamental'] -= 4; reasons.append(f'High P/E ({pe:.1f})')
        if pe > 80: signals['fundamental'] -= 3; reasons.append('Very high P/E penalty')
    else:
        signals['fundamental'] -= 2; reasons.append('Unknown P/E')

    divy = fnd.get('dividend_yield')
    if isinstance(divy,(int,float)) and 0.1 <= divy <= 12.0 and price < sma200:
        signals['fundamental'] -= 3; reasons.append('High dividend in downtrend (possible trap)')

    # News sentiment with cap (+ lower weight on intraday)
    if news and news.get('n',0)>0:
        s = float(news.get('compound',0)); c=float(news.get('confidence',0.5))
        pts = int(10*c) if abs(s)>0.3 else (int(5*c) if abs(s)>0.1 else 0)
        pts = int(np.sign(s) * min(abs(pts), CFG['news']['sentiment_cap_points']))
        if interval != '1d':
            pts = int(pts * CFG['news'].get('intraday_weight', 0.25))
        if pts!=0: signals['sentiment'] += pts; reasons.append(f"News sentiment {s:+.2f} (cap {CFG['news']['sentiment_cap_points']}{' intraday×'+str(CFG['news'].get('intraday_weight',0.25)) if interval!='1d' else ''})")

    vol_ann = float(cur.get('Volatility',0) or 0)
    raw = sum(signals.values()) - min(10, vol_ann/5.0)
    score = int(np.interp(raw, [-40,40], [0,100])); score=max(0,min(100,score))

    base_thr_buy = {"conservative":65,"balanced":60,"aggressive":55}[risk_profile]
    base_thr_sell= {"conservative":35,"balanced":40,"aggressive":45}[risk_profile]
    regime = current_regime()
    thr_buy = base_thr_buy + regime.get('thr_buy_adj',0)
    thr_sell= base_thr_sell + regime.get('thr_sell_adj',0)

    band = 2
    if score >= thr_buy + band: signal='BUY'
    elif score <= thr_sell - band: signal='SELL'
    else: signal='HOLD'

    trend_ok = (price>sma50>sma200) if all(finite(x) for x in [price,sma50,sma200]) else False
    momentum_ok = (mac>sig and r5>0) if all(finite(x) for x in [mac,sig,r5]) else False
    volume_ok = (finite(vr) and vr>=entry_vmin)
    consensus = sum([trend_ok, momentum_ok, volume_ok])
    if signal=='BUY' and consensus < 2:
        signal='HOLD'; reasons.append('Consensus 2/3 not met')

    er=None
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if ed is not None and len(ed)>0:
            idx = ed.index.tz_localize(None) if hasattr(ed.index,'tz') and ed.index.tz is not None else ed.index
            nowu = dt.datetime.utcnow()
            future = [(d.to_pydatetime()-nowu).days for d in idx if (d.to_pydatetime()-nowu).days>=0]
            if future: er = min(future)
    except Exception: pass
    if er is not None and er<=7 and signal=='BUY':
        signal='HOLD'; reasons.append('Earnings lockout (≤7d)')

    if signal=='BUY':
        local_vmin = entry_vmin + (0.2 if (finite(bbpos) and bbpos>90) else 0.0)
        if not volume_ok or (finite(vr) and vr < local_vmin):
            signal='HOLD'; reasons.append(f'Volume < {local_vmin:.1f}× avg')
        if finite(rsi14) and rsi14>70:
            signal='HOLD'; reasons.append('RSI>70')
        if not trend_ok:
            signal='HOLD'; reasons.append('Trend not aligned (need Close>SMA50>SMA200)')
        confs2 = two_bar_confirmation(df)
        if not (confs2['rsi'] or confs2['macd'] or confs2['price']):
            signal='HOLD'; reasons.append('Need 2-bar confirmation')
        prev_score = _score_simple(prev)
        if prev_score < thr_buy:
            signal='HOLD'; reasons.append('Signal stability: prev bar below threshold')

    ev_info = lookup_ev(regime.get('state','unknown'), score)
    if ev_info is not None:
        ev, n = ev_info
        reasons.append(f'EV (10d) ≈ {ev:+.2f}% [n={n}]')
        if ev < 0 and signal=='BUY':
            signal='HOLD'; reasons.append('EV<0 → skip entry')

    avg_conf = int(min(100, max(0, (np.mean(confs) if confs else 0.5)*100)))

    pos_size = None
    if vol_ann and vol_ann>0:
        target_vol = 10.0
        pos_size = float(np.clip(target_vol / vol_ann, 0.2, 1.0))
        reasons.append(f'Pos size≈{pos_size:.2f}× (vol targeting)')

    out = {
        "ticker": ticker,
        "signal": signal,
        "score": int(score),
        "confidence": int(avg_conf),
        "price": float(price),
        "signals_breakdown": signals,
        "reasons": reasons[:12],
        "risk_profile": risk_profile,
        "fundamental_data": {
            "pe_ratio": float(pe) if finite(pe) else None,
            "beta": float(fi.get('beta')) if finite(fi.get('beta')) else None,
            "market_cap": int(fi.get('market_cap')) if finite(fi.get('market_cap')) else None,
            "dividend_yield": float(divy) if isinstance(divy,(int,float)) else None,
            "sector": fnd.get('sector'), "industry": fnd.get('industry')},
            "earnings_in_days": er,
            "regime": regime,
            "position_size": pos_size,
            "ev": ev_info[0] if ev_info else None,
            "ev_n": ev_info[1] if ev_info else None
    }
    return out

# -------------------- Backtests --------------------

def backtest_with_atr(df: pd.DataFrame, risk_profile: str, regime: Dict, confirm_2bars: bool=True, cost_bps: float=10, slippage_bps: float=10) -> Dict:
    if df is None or df.empty: return {"trades":0}
    df = compute_indicators(df.copy())
    thr_buy = {"conservative":65,"balanced":60,"aggressive":55}[risk_profile] + regime.get('thr_buy_adj',0)
    thr_sell= {"conservative":35,"balanced":40,"aggressive":45}[risk_profile] + regime.get('thr_sell_adj',0)
    atr_mult = regime.get('atr_mult', 2.0)

    cash=1.0; pos=0.0; entry_px=0.0; wins=0; losses=0; trades=0
    peak=1.0; maxdd=0.0

    scores = df.apply(_score_simple, axis=1)

    def confirmed(i):
        if not confirm_2bars or i<2: return True if i>=2 else False
        rsi_ok = all(df.iloc[i-k]['RSI14']>50 for k in [0,1] if finite(df.iloc[i-k]['RSI14']))
        mac_ok = all(df.iloc[i-k]['MACD']>df.iloc[i-k]['MACD_SIG'] for k in [0,1] if all(finite(x) for x in [df.iloc[i-k]['MACD'],df.iloc[i-k]['MACD_SIG']]))
        pr_ok  = all(df.iloc[i-k]['Close']>df.iloc[i-k]['SMA20'] for k in [0,1] if all(finite(x) for x in [df.iloc[i-k]['Close'],df.iloc[i-k]['SMA20']]))
        return rsi_ok or mac_ok or pr_ok

    last = len(df)-1
    for i in range(2,last):
        px = float(df.iloc[i]['Close'])
        if pos>0:
            stop = entry_px - atr_mult*float(df.iloc[i]['ATR']) if finite(df.iloc[i]['ATR']) else entry_px*0.95
            if scores.iloc[i] <= thr_sell or px < stop:
                sell_px = px * (1 - (cost_bps+slippage_bps)/1e4)
                cash *= (sell_px/entry_px)
                pos=0.0; trades+=1
                peak=max(peak,cash); maxdd=max(maxdd,1-cash/peak)
                continue
        if pos==0 and scores.iloc[i] >= thr_buy and confirmed(i):
            buy_px = px * (1 + (cost_bps+slippage_bps)/1e4)
            entry_px = buy_px; pos=1.0
        peak=max(peak,cash); maxdd=max(maxdd,1-cash/peak)

    cagr = (cash ** (252/max(1,len(df)))) - 1 if len(df)>252 else cash-1
    return {"trades":trades,"final_equity":cash,"CAGR":cagr,"maxDD":maxdd}


# >>>>>>>>>>>>>>>>>>>>>>>>  ЗАМЕСТЕНА ФУНКЦИЯ  <<<<<<<<<<<<<<<<<<<<<<<<<<
def portfolio_walkforward_backtest(
    tickers: List[str],
    risk_profile: str,
    train_m: int = 18,
    test_m: int = 6,
    top_k: int = 5,
    rebalance: str = "M",          # 'M' (месечно) или 'W-MON' (седмично)
    cost_bps: int = 5,
    slip_bps: int = 5,
    min_hold_days: int = 7,        # минимум дни за държане
) -> Dict:
    """
    Rolling walk-forward портфейлен OOS бектест.
    PATCH:
      - Hysteresis: пази текущите позиции ако ранк ≤ top_k+2 и не са SELL
      - SELL правило: score < buy_thr-7 или EV<=0
      - EV>0 и score ≥ buy_thr+5 за НОВИ включвания
      - Turnover връща ГОДИШНА стойност
    """
    if not tickers:
        return {"oos_trades": 0, "oos_equity": 1.0, "oos_CAGR": 0.0, "oos_maxDD": 0.0, "oos_turnover": 0.0, "oos_sharpe": 0.0}

    # 1) Данни
    data: Dict[str, pd.DataFrame] = {}
    hist_days = int((train_m + test_m) * 22 * 3)  # буфер
    for t in tickers:
        df = fetch_price_history(t, hist_days, "1d")
        if df is None or df.empty:
            continue
        df = compute_indicators(df)
                # --- ново: моментум и волатилност ---
        ret1 = df['Close'].pct_change()
        df['vol20'] = ret1.rolling(20).std() * np.sqrt(252)   # годишна волатилност
        df['mom126'] = df['Close'].pct_change(126)            # ~6 месеца

        data[t] = df
    if not data:
        return {"oos_trades": 0, "oos_equity": 1.0, "oos_CAGR": 0.0, "oos_maxDD": 0.0, "oos_turnover": 0.0, "oos_sharpe": 0.0}

    all_idx = sorted(set().union(*[df.index for df in data.values()]))
    all_idx = pd.DatetimeIndex(all_idx)

    # --- режим от SPY (за EV филтъра) ---
    spy = build_spy_for_regime(1200)
        # --- ново: 6м моментум и за SPY ---
    try:
        spy['mom126'] = spy['Close'].pct_change(126)
    except Exception:
        pass

    # Прагова логика (фиксирана за OOS)
    base_buy  = {"conservative": 65, "balanced": 60, "aggressive": 55}[risk_profile]
    buy_thr   = base_buy + 2
    sell_thr  = buy_thr - 7  # по-лесен sell от buy (hysteresis)
    slack     = 2            # позволяваме да държим до top_k+2 по ранг

    # 2) Rolling сегменти
    step = int(test_m * 22)
    start_i = int(train_m * 22)
    if len(all_idx) <= start_i + 5:
        return {"oos_trades": 0, "oos_equity": 1.0, "oos_CAGR": 0.0, "oos_maxDD": 0.0, "oos_turnover": 0.0, "oos_sharpe": 0.0}

    equity = 1.0
    daily_rets: List[float] = []
    turnover_sum = 0.0  # кумулативен, ще го годишним по-долу
    trades = 0
    cost_rate = (cost_bps + slip_bps) / 1e4

    held: Dict[str, float] = {}   # тегла
    age: Dict[str, int] = {}      # дни в позиция
    last_px: Dict[str, float] = {}

    for seg_start in range(start_i, len(all_idx) - 1, step):
        seg_end = min(seg_start + step, len(all_idx) - 1)
        test_idx = all_idx[seg_start:seg_end]
        if len(test_idx) < 5:
            break

        rebal_dates = pd.date_range(test_idx[0], test_idx[-1], freq=rebalance)
        rebal_dates = set(pd.DatetimeIndex(rebal_dates))

        for d in test_idx:
            # --- Ребаланс ---
            if d in rebal_dates:
                # Кандидати (нов вход): score ≥ buy_thr+5 И EV>0 по режим
                # Кандидати (нов вход): score ≥ buy_thr+5, EV>0, тренд и моментум срещу SPY
                ranked: List[Tuple[str, int]] = []

                # режим + mom на SPY за тази дата
                reg = 'unknown'
                spy_mom = 0.0
                try:
                    idx = d
                    if not spy.empty:
                        if idx not in spy.index:
                            idx = spy.index[spy.index.get_loc(idx, method='pad')]
                        reg = 'bull' if float(spy.loc[idx, 'SMA50']) > float(spy.loc[idx, 'SMA200']) else 'bear'
                        spy_mom = float(spy.loc[idx, 'mom126']) if 'mom126' in spy.columns else 0.0
                except Exception:
                    reg = 'unknown'

                for t, df in data.items():
                    if d not in df.index:
                        continue
                    row = df.loc[d]
                    sc = _score_simple(row)

                    # тренд > Close > SMA50 > SMA200 (fallback: Close > SMA200)
                    trend_ok = False
                    try:
                        trend_ok = float(row['Close']) > float(row['SMA50']) > float(row['SMA200']
                    except Exception:
                        try:
                            trend_ok = float(row['Close']) > float(row['SMA200']
                        except Exception:
                            trend_ok = False 
                    # моментум срещу SPY
                    mom_t = float(row['mom126']) if 'mom126' in row else -1.0
                    mom_ok = mom_t > spy_mom
                    # волатилност под таван (рязко шумни активи ги режем)
                    vol_t = float(row['vol20']) if 'vol20' in row else 0.5
                    vol_ok = vol_t <= 0.60  # 60% годишна като горна граница

                    ev_ok = False
                    if reg != 'unknown':
                        ev_info = lookup_ev(reg, sc)
                        if ev_info is not None and ev_info[0] > 0:
                            ev_ok = True

                    if sc >= (buy_thr + 5) and ev_ok and trend_ok and mom_ok and vol_ok:
                        ranked.append((t, sc))

                ranked.sort(key=lambda x: x[1], reverse=True)
                rank_pos = {t: i for i, (t, _) in enumerate(ranked)}  # 0 е най-добър


                # Кои от текущите да задържим (hysteresis + sell правило)
                keep: List[str] = []
                for t in list(held.keys()):
                    # ако не е навършил min_hold_days → задържаме
                    if age.get(t, 0) < min_hold_days:
                        keep.append(t)
                        continue
                    # ако няма данни за днес — по-добре задържай
                    if t not in data or d not in data[t].index:
                        keep.append(t)
                        continue
                    row = data[t].loc[d]
                    sc_now = _score_simple(row)
                    # EV по режим за текущата позиция
                    ev_ok = False
                    if reg != 'unknown':
                        ev_info = lookup_ev(reg, sc_now)
                        if ev_info is not None and ev_info[0] > 0:
                            ev_ok = True
                    rnk = rank_pos.get(t, 9999)
                    # Държим ако НЕ е SELL и не е изпаднал далеч от топа
                    if (sc_now >= sell_thr) and ev_ok and (rnk <= top_k + slack):
                        keep.append(t)

                # Добавяме нови от ranked до top_k
                selected = list(dict.fromkeys(keep))  # unique & order
                for t, _ in ranked:
                    if len(selected) >= top_k:
                        break
                    if t not in selected:
                        selected.append(t)

                # Пресмятане turnover и разходи 
                # Тегла : инверзна волатилност с клемове(15% ..60%) 
                if selected:
                    vols = []
                    for t in selected:
                        try:
                            v = float(data[t].loc[d, "vol20"])
                        except Exeption:
                            v = 0.35 # дефолт ако липсва
                        v = min(max(v, 0.15), 0.60)  # clamp
                        vols.append(v)
                    inv = np.array([1.0 / v for v in vols], dtype=float)
                    w = inv / inv.sum()
                    desired = {t: float(wi) for t, wi in zip(selected, w)}
                else:
                    desired = {}
                
                t_over = 0.0
                changed = 0
                for t in set(list(held.keys()) + list(desired.keys())):
                    w_old = held.get(t, 0.0)
                    w_new = desired.get(t, 0.0)
                    if abs(w_new - w_old) > 1e-9:
                        t_over += abs(w_new - w_old)
                        changed += 1
                if t_over > 0:
                    equity *= max(0.0, 1.0 - t_over * cost_rate)
                    turnover_sum += t_over
                    trades += changed

                # Обнови портфейла
                new_age: Dict[str, int] = {}
                for t in desired:
                    new_age[t] = age.get(t, -1) + 1
                    if t not in held and t in data and d in data[t].index:
                        last_px[t] = float(data[t].loc[d, "Close"])
                for t in list(last_px.keys()):
                    if t not in desired:
                        last_px.pop(t, None)

                held = desired
                age = new_age

            # --- Дневна доходност ---
            day_ret = 0.0
            for t, w in held.items():
                df = data.get(t)
                if df is None or d not in df.index:
                    continue
                px = float(df.loc[d, "Close"])
                if t not in last_px:
                    last_px[t] = px
                    continue
                if last_px[t] > 0:
                    r = (px - last_px[t]) / last_px[t]
                    day_ret += w * r
                last_px[t] = px

            # възраст
            for t in list(age.keys()):
                age[t] = age.get(t, 0) + 1

            equity *= (1.0 + day_ret)
            daily_rets.append(day_ret)

    # 3) Метрики
    rets = np.array(daily_rets, dtype=float)
    if len(rets) == 0:
        return {"oos_trades": trades, "oos_equity": equity, "oos_CAGR": 0.0, "oos_maxDD": 0.0, "oos_turnover": 0.0, "oos_sharpe": 0.0}

    equity_curve = (1.0 + rets).cumprod()
    years = max(1e-9, len(rets) / 252.0)
    cagr = equity_curve[-1] ** (1.0 / years) - 1.0
    roll_max = np.maximum.accumulate(equity_curve)
    maxdd = float(np.max((roll_max - equity_curve) / np.maximum(roll_max, 1e-9)))
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252.0)) if len(rets) > 1 else 0.0

    # ГОДИШЕН turnover (не кумулативен)
    ann_turnover = float(turnover_sum / years)

    return {
        "oos_trades": trades,
        "oos_equity": float(equity_curve[-1]),
        "oos_CAGR": float(cagr),
        "oos_maxDD": float(maxdd),
        "oos_turnover": float(ann_turnover),
        "oos_sharpe": float(sharpe),
    }

# <<<<<<<<<<<<<<<<<<<<<<  КРАЙ НА ЗАМЯНАТА  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# -------------------- Scan & Risk Manager --------------------

def load_watchlist() -> List[str]:
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                out = []
                for t in data:
                    if isinstance(t, str) and 1 <= len(t.strip()) <= 10:
                        out.append(t.strip().upper())
                return sorted(set(out))
    except Exception:
        pass
    default_watchlist = ["AAPL","MSFT","GOOGL","NVDA","AMZN","META","TSLA","SPY"]
    save_watchlist(default_watchlist); return default_watchlist

def save_watchlist(tickers: List[str]) -> bool:
    try:
        clean = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
        uniq  = sorted(set([t for t in clean if 1 <= len(t) <= 10]))
        WATCHLIST_FILE.write_text(json.dumps(uniq, indent=2, ensure_ascii=False), encoding="utf-8")
        st.sidebar.success(f"✅ Watchlist saved ({len(uniq)})"); return True
    except Exception as e:
        st.sidebar.error(f"❌ Error saving watchlist: {e}"); return False


def process_one(ticker: str, cfg: Dict):
    days = cfg.get('lookback_days',120); interval=cfg.get('interval','1d'); market_key = cfg.get('market_key', list(MARKETS.keys())[0])
    use_news = cfg.get('use_news', True); risk = cfg.get('risk_profile','balanced')
    df_raw = fetch_price_history(ticker, days, interval)
    if df_raw.empty: return None, None
    df = trim_to_closed(df_raw, interval, market_key)
    if df.empty or len(df)<3: return None, None
    df = compute_indicators(df)
    analysis = classify_one(ticker, df, risk_profile=risk, market_key=market_key, use_news=use_news, interval=interval)
    cur = df.iloc[-1]
    row = {
        "Ticker": ticker,
        "Signal": analysis.get('signal','N/A'),
        "Score": analysis.get('score',0),
        "Confidence": f"{analysis.get('confidence',0)}%",
        "Price": f"${float(cur['Close']):.2f}",
        "RSI": f"{float(cur.get('RSI14',np.nan)):.1f}" if finite(cur.get('RSI14',np.nan)) else "N/A",
        "Volume Ratio": f"{float(cur.get('Volume_Ratio',np.nan)):.1f}×" if finite(cur.get('Volume_Ratio',np.nan)) else "N/A",
        "5D Return": f"{float(cur.get('Return_5d',np.nan)):+.1f}%" if finite(cur.get('Return_5d',np.nan)) else "N/A",
        "P/E Ratio": f"{float(analysis.get('fundamental_data',{}).get('pe_ratio',np.nan)):.1f}" if finite(analysis.get('fundamental_data',{}).get('pe_ratio',np.nan)) else "N/A",
    }
    return analysis, row


def apply_risk_caps(results: List[Dict]) -> List[Dict]:
    if not results: return results
    max_pos = CFG['risk']['max_positions']; sector_cap = CFG['risk']['sector_cap_pct']
    by_sector: Dict[str,int] = {}; kept = 0
    for r in results:
        if r.get('signal') != 'BUY': continue
        sector = (r.get('fundamental_data',{}).get('sector') or 'Unknown')
        limit = max(1, int(max_pos * sector_cap))
        if kept >= max_pos or by_sector.get(sector,0) >= limit:
            r['reasons'] = (r.get('reasons') or []) + ["Risk cap reached (max positions/sector cap)"]
            r['signal'] = 'HOLD'
        else:
            by_sector[sector] = by_sector.get(sector,0) + 1; kept += 1
    return results


def scan_tickers(tickers: List[str], cfg: Dict, progress=None) -> Tuple[List[Dict], List[Dict]]:
    precalc_ev_if_needed(extra_universe=tickers)
    results, rows = [], []
    if not tickers: return results, rows
    max_workers = min(6, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_one, t, cfg): t for t in tickers}
        for i, fut in enumerate(as_completed(futs)):
            t = futs[fut]
            try:
                res, row = fut.result()
                if res: results.append(res)
                if row: rows.append(row)
            except Exception as e:
                st.warning(f"{t}: {e}")
            if progress: progress((i+1)/len(tickers))
            time.sleep(0.02)
    results.sort(key=lambda r: r.get('score',0), reverse=True)
    results = apply_risk_caps(results)
    return results, rows

# -------------------- UI (unchanged layout) --------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.title(APP_TITLE); st.caption("Advanced multi-source analysis with regime, EV, SEC facts & risk caps. Not financial advice.")
    if st_autorefresh: st_autorefresh(interval=15*60*1000, key="auto_refresh_15min")

    settings = {"risk_profile": CFG.get('risk_profile','balanced'), "lookback_days": CFG.get('lookback_days',120), "news_days": CFG.get('news_days',7), "show_charts": CFG.get('show_charts',True)}

    st.sidebar.header("⚙️ Enhanced Configuration")
    market_key = st.sidebar.selectbox("Market Profile:", list(MARKETS.keys()), index=0)
    is_open = is_market_open_raw(market_key); mkt = MARKETS[market_key]
    st.sidebar.markdown(f"**Market Status:** {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
    st.sidebar.markdown(f"**Local Time:** {now_tz(mkt['tz']).strftime('%H:%M:%S %Z')}")

    st.sidebar.subheader("📊 Analysis")
    risk_profile = st.sidebar.selectbox("Risk Profile:", ["conservative","balanced","aggressive"], index=["conservative","balanced","aggressive"].index(settings.get("risk_profile","balanced")))
    lookback_days = st.sidebar.slider("Historical Data (days):", 30, 365, settings.get("lookback_days",120))
    interval = st.sidebar.selectbox("Data Interval:", ["1d","30m"], index=0)

    st.sidebar.subheader("🧪 Extras")
    show_charts = st.sidebar.checkbox("Interactive Charts", value=settings.get("show_charts",True))

    st.sidebar.subheader("📋 Persistent Watchlist")
    wl = load_watchlist()
    if wl: st.sidebar.markdown(f"**Saved ({len(wl)}):** `{', '.join(wl[:8])}{' ...' if len(wl)>8 else ''}`")
    colA,colB = st.sidebar.columns(2)
    with colA:
        new_t = st.text_input("Add Stock:", placeholder="AAPL").strip().upper()
        if st.button("➕ Add") and new_t:
            if 1<=len(new_t)<=10 and new_t not in wl:
                wl.append(new_t); 
                if save_watchlist(wl): st.rerun()
            else: st.sidebar.warning("Invalid or duplicate ticker")
    with colB:
        if wl:
            rem = st.selectbox("Remove:", ["Select..."]+wl)
            if st.button("➖ Remove") and rem!="Select...":
                wl.remove(rem); 
                if save_watchlist(wl): st.rerun()

    if not wl:
        st.warning("🚨 No stocks in watchlist. Add tickers in the sidebar."); return

    cfg = {"lookback_days":lookback_days, "interval":interval, "market_key":market_key, "use_news":True, "risk_profile":risk_profile}

    if st.button("🚀 Run Enhanced Analysis", type="primary"):
        prog = st.progress(0); info = st.empty()
        def upd(p): prog.progress(p); info.text(f"Analyzing {len(wl)} stocks… {int(p*100)}%")
        with st.spinner("Running analysis (EV/regime/risk caps)…"):
            results, rows = scan_tickers(wl, cfg, upd)
        prog.empty(); info.empty()
        if not results:
            st.error("❌ No analysis results."); return

        st.header("📊 Analysis Dashboard")
        strong_buy = len([r for r in results if r['signal']=='BUY' and r['score']>=80])
        buy_cnt    = len([r for r in results if r['signal']=='BUY'])
        sell_cnt   = len([r for r in results if r['signal']=='SELL'])
        avg_conf   = np.mean([r.get('confidence',0) for r in results])
        total      = len(results)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Strong Buy", strong_buy, delta=f"{strong_buy}/{total}")
        c2.metric("Buy Signals", buy_cnt, delta=f"{buy_cnt}/{total}")
        c3.metric("Sell Signals", sell_cnt, delta=f"{sell_cnt}/{total}")
        c4.metric("Avg Confidence", f"{avg_conf:.0f}%")
        c5.metric("Stocks Analyzed", total)

        st.subheader("📈 Detailed Results")
        if rows:
            df_res = pd.DataFrame(rows); st.dataframe(df_res, use_container_width=True)
            st.download_button("📥 Download CSV", df_res.to_csv(index=False).encode('utf-8'), file_name=f"stock_analysis_{dt.date.today():%Y%m%d}.csv", mime="text/csv")

        st.subheader("🎯 Individual Stock Analysis")
        for r in results:
            t=r['ticker']; sig=r['signal']; sc=r['score']; conf=r.get('confidence',0)
            badge = "🟢" if sig=="BUY" else ("🔴" if sig=="SELL" else "⚪")
            with st.expander(f"{badge} {t} – {sig} (Score {sc}, Confidence {conf}%)"):
                col1,col2 = st.columns([2,1])
                with col1:
                    st.markdown("**Key Signals:**")
                    for i,reason in enumerate(r.get('reasons',[])[:10],1):
                        st.markdown(f"{i}. {reason}")
                    br = r.get('signals_breakdown',{})
                    if br:
                        st.markdown("**Signal Components:**")
                        for k,v in br.items():
                            if v!=0: st.markdown(f"{'➕' if v>0 else '➖'} {k.title()}: {v:+d}")
                with col2:
                    st.markdown("**Current Data:**")
                    st.markdown(f"Price: **${r['price']:.2f}**")
                    fd = r.get('fundamental_data',{})
                    if fd.get('pe_ratio') is not None: st.markdown(f"P/E: **{fd['pe_ratio']:.1f}**")
                    if fd.get('dividend_yield') is not None: st.markdown(f"Dividend: **{fd['dividend_yield']:.2f}%**")
                    if r.get('earnings_in_days') is not None: st.markdown(f"🗓️ Earnings in **{r['earnings_in_days']}** days")
                    reg = r.get('regime',{})
                    st.markdown(f"Regime: **{reg.get('state','?')}**, VIX: **{reg.get('vix','?')}**")
                    if r.get('ev') is not None: st.markdown(f"EV (10d): **{r['ev']:+.2f}%** [n={r.get('ev_n')}] ")
                if CFG.get('show_charts', True):
                    try:
                        look = max(120, CFG.get('lookback_days',120))
                        dfx = fetch_price_history(t, look, '1d')
                        dfx = trim_to_closed(dfx, '1d', market_key)
                        dfx = compute_indicators(dfx)
                        fig = make_subplots(rows=3, cols=1, shared_xaxis=True, vertical_spacing=0.08, row_heights=[0.6,0.2,0.2], subplot_titles=[f'{t} Price','RSI','MACD'])
                        fig.add_trace(go.Candlestick(x=dfx.index, open=dfx['Open'], high=dfx['High'], low=dfx['Low'], close=dfx['Close'], name=t), row=1,col=1)
                        for p in [20,50]:
                            c=f'SMA{p}';
                            if c in dfx.columns: fig.add_trace(go.Scatter(x=dfx.index, y=dfx[c], mode='lines', name=c, opacity=0.7), row=1,col=1)
                        fig.add_trace(go.Scatter(x=dfx.index, y=dfx['RSI14'], mode='lines', name='RSI(14)'), row=2,col=1)
                        fig.add_hline(y=70, line_dash='dash', row=2,col=1); fig.add_hline(y=30, line_dash='dash', row=2,col=1)
                        fig.add_trace(go.Scatter(x=dfx.index, y=dfx['MACD'], mode='lines', name='MACD'), row=3,col=1)
                        fig.add_trace(go.Scatter(x=dfx.index, y=dfx['MACD_SIG'], mode='lines', name='Signal'), row=3,col=1)
                        fig.update_layout(title=f"{t} – Enhanced Technicals", xaxis_rangeslider_visible=False, height=780, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

        # Walk-forward OOS caption – ПОРТФЕЙЛЕН
      # Walk-forward OOS caption – ПОРТФЕЙЛЕН
# Walk-forward OOS caption – ПОРТФЕЙ
                # Walk-forward OOS caption – ПОРТФЕЙЛЕН
        try:
            # Универз за портфейла: SP100 + текущите ти тикери (без дубликати)
            univ = list({*SP100, *[r['ticker'] for r in results]})

            res_pf = portfolio_walkforward_backtest(
                univ,
                risk_profile,
                CFG['wf']['train_months'],
                CFG['wf']['test_months'],
                CFG['wf']['top_k'],
                CFG['wf']['rebalance'],
                CFG['wf']['cost_bps'],
                CFG['wf']['slip_bps'],
                min_hold_days=CFG['wf'].get('min_hold_days', 7)
            )
            if res_pf.get('oos_trades', 0) >= 0:
                st.caption(
                    f"📦 Portfolio OOS: CAGR={res_pf.get('oos_CAGR', 0):.2%} · "
                    f"maxDD={res_pf.get('oos_maxDD', 0):.2%} · "
                    f"Sharpe~{res_pf.get('oos_sharpe', 0):.2f} · "
                    f"turnover={res_pf.get('oos_turnover', 0):.2f}"
                )
        except Exception:
            pass

        # ТОВА ОСТАВА ВЪТРЕ В if-блока, за да вижда results
        st.success(f"✅ Analysis complete! Processed {len(results)} stocks.")
        

    
if __name__ == "__main__":
    main()
