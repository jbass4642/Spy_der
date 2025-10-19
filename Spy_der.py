from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import time
import schwabdev
import os
from dotenv import load_dotenv
import pytz
import requests
import json
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import plotly.io as pio
from PIL import Image, ImageDraw, ImageFont
import io
from scipy.stats import norm
from scipy.optimize import brentq

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scalpnet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Initialize Schwab client
try:
    client = schwabdev.Client(
        os.getenv('SCHWAB_APP_KEY'),
        os.getenv('SCHWAB_APP_SECRET'),
        os.getenv('SCHWAB_CALLBACK_URL')
    )
    logger.info("Schwab client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Schwab client: {e}")
    client = None

# Discord webhook URLs
DISCORD_WEBHOOK_URLS = [os.getenv("DISCORD_WEBHOOK")]

# Timezone
CDT = pytz.timezone('America/Chicago')

# Global variables
last_alert_time = 0
previous_put_wall = None
historical_drop_times = set()
historical_rise_times = set()

# Helper functions
def format_ticker(ticker):
    return ticker.upper()

def get_current_price(ticker):
    if client is None:
        logger.error("Schwab client not initialized")
        return None
    try:
        quote_response = client.quotes(ticker)
        if not quote_response.ok:
            logger.error(f"Quote response not OK: {quote_response.status_code} - {quote_response.text}")
            return None
        quote = quote_response.json()
        if quote and ticker in quote:
            return quote[ticker]['quote']['lastPrice']
        logger.error(f"No quote data found for ticker {ticker}")
        return None
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {e}")
        return None

def black_scholes_delta(S, K, T, r, sigma, is_put=False):
    """Calculate Black-Scholes delta"""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    if is_put:
        return norm.cdf(d1) - 1
    return norm.cdf(d1)

def black_scholes_vanna(S, K, T, r, sigma):
    """Calculate Black-Scholes vanna"""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    vanna = -norm.pdf(d1) * d2 / sigma
    return vanna

def black_scholes_charm(S, K, T, r, sigma, is_put=False):
    """Calculate Black-Scholes charm"""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    phi_d1 = norm.pdf(d1)
    if is_put:
        charm = phi_d1 * (r / (S * sigma * math.sqrt(T)) - d2 / (2 * T))
    else:
        charm = -phi_d1 * (r / (S * sigma * math.sqrt(T)) + d2 / (2 * T))
    return charm

def black_scholes_price(S, K, T, r, sigma, is_put=False):
    """Calculate Black-Scholes option price"""
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if is_put:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    
    return price

def implied_volatility(S, K, T, r, price, is_put=False):
    """Calculate implied volatility using Brent's method"""
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, is_put) - price
    
    try:
        vol = brentq(objective, 0.001, 5.0)
        return vol
    except ValueError:
        return 0.2  # Default to 20% volatility

def filter_market_hours(candles):
    """Filter candles to only include regular market hours (9:30 AM - 4:00 PM CDT)"""
    filtered_candles = []
    cdt = pytz.timezone('America/Chicago')
    for candle in candles:
        dt = datetime.fromtimestamp(candle['datetime']/1000, tz=pytz.UTC).astimezone(cdt)
        if dt.weekday() < 5:  # Monday-Friday
            market_open = dt.replace(hour=8, minute=30, second=0, microsecond=0)
            market_close = dt.replace(hour=15, minute=0, second=0, microsecond=0)
            if market_open <= dt <= market_close:
                filtered_candles.append(candle)
    return filtered_candles

def convert_to_heikin_ashi(candles):
    """Convert regular OHLC candles to Heikin-Ashi candles"""
    if not candles:
        return []
    
    ha_candles = []
    prev_ha_open = None
    prev_ha_close = None
    
    for candle in candles:
        # Calculate Heikin-Ashi values
        ha_close = (candle['open'] + candle['high'] + candle['low'] + candle['close']) / 4
        
        if prev_ha_open is None:
            # First candle: HA_Open = (Open + Close) / 2
            ha_open = (candle['open'] + candle['close']) / 2
        else:
            # Subsequent candles: HA_Open = (Previous HA_Open + Previous HA_Close) / 2
            ha_open = (prev_ha_open + prev_ha_close) / 2
        
        ha_high = max(candle['high'], ha_open, ha_close)
        ha_low = min(candle['low'], ha_open, ha_close)
        
        # Create new candle with Heikin-Ashi values
        ha_candle = {
            'datetime': candle['datetime'],
            'open': ha_open,
            'high': ha_high,
            'low': ha_low,
            'close': ha_close,
            'volume': candle['volume']
        }
        
        ha_candles.append(ha_candle)
        
        # Store values for next iteration
        prev_ha_open = ha_open
        prev_ha_close = ha_close
    
    return ha_candles


def find_vix_drop_times(spy_price_data, vix_price_data, put_wall):
    """Identify timestamps where SPY low is at/near Put Wall and VIX is dropping."""
    if spy_price_data is None or spy_price_data.empty or vix_price_data is None or vix_price_data.empty:
        logger.info("No SPY or VIX data for VIX drop analysis")
        return []
    
    # Merge SPY and VIX data on datetime
    merged_data = spy_price_data[['datetime', 'low', 'close']].merge(
        vix_price_data[['datetime', 'open', 'close', 'high']],
        on='datetime',
        how='inner',
        suffixes=('_spy', '_vix')
    )
    
    if merged_data.empty:
        logger.info("No overlapping SPY and VIX data for VIX drop analysis")
        return []
    
    drop_times = []
    for i in range(1, len(merged_data)):
        spy_low = merged_data['low'].iloc[i]
        vix_close = merged_data['close_vix'].iloc[i]
        vix_open = merged_data['open'].iloc[i]
        vix_high = merged_data['high'].iloc[i]
        vix_prev_high = merged_data['high'].iloc[i-1]
        vix_prev_close = merged_data['close_vix'].iloc[i-1]
        timestamp = merged_data['datetime'].iloc[i]
        
        # Check conditions: SPY low at/near Put Wall and VIX dropping
        is_spy_near_put_wall = spy_low <= put_wall
        is_vix_dropping = vix_close < vix_prev_close or vix_high <= vix_prev_high
        
        if is_spy_near_put_wall and is_vix_dropping:
            # Convert timestamp to timezone-naive
            naive_timestamp = timestamp.tz_localize(None)
            drop_times.append(naive_timestamp)
            logger.info(f"VIX drop condition met at {naive_timestamp}: SPY low={spy_low:.2f}, Put Wall={put_wall:.2f}, VIX close={vix_close:.2f}, VIX prev close={vix_prev_close:.2f}")
    
    return drop_times

def find_vix_rise_times(spy_price_data, vix_price_data, call_wall):
    """Identify timestamps where SPY price is at/above POC and VIX is rising, only if there was a previous entry."""
    
    if spy_price_data is None or spy_price_data.empty or vix_price_data is None or vix_price_data.empty:
        logger.info("No SPY or VIX data for VIX rise analysis")
        return []
    
    # Merge SPY and VIX data on datetime
    merged_data = spy_price_data[['datetime', 'high', 'close']].merge(
        vix_price_data[['datetime', 'open', 'close','low']],
        on='datetime',
        how='inner',
        suffixes=('_spy', '_vix')
    )
    
    if merged_data.empty:
        logger.info("No overlapping SPY and VIX data for VIX rise analysis")
        return []
    
    rise_times = []
    for i in range(1, len(merged_data)):
        spy_high = merged_data['high'].iloc[i]
        vix_close = merged_data['close_vix'].iloc[i]
        vix_open = merged_data['open'].iloc[i]
        vix_low = merged_data['low'].iloc[i]
        vix_prev_low = merged_data['low'].iloc[i-1]
        vix_prev_close = merged_data['close_vix'].iloc[i-1]
        timestamp = merged_data['datetime'].iloc[i]
        
        # Check conditions: SPY at/above POC and VIX rising
        is_spy_at_poc = spy_high > call_wall
        is_vix_rising = vix_close > vix_prev_close or vix_low >= vix_prev_low
        
        if is_spy_at_poc and is_vix_rising:
            # Convert timestamp to timezone-naive
            naive_timestamp = timestamp.tz_localize(None)
            rise_times.append(naive_timestamp)
            logger.info(f"VIX rise condition met at {naive_timestamp}: SPY high={spy_high:.2f}, Call Wall={call_wall:.2f}, VIX close={vix_close:.2f}, VIX prev close={vix_prev_close:.2f}")
    
    return rise_times

def get_price_history(ticker, frequency=1):
    if client is None:
        logger.error("Schwab client not initialized")
        return None
    try:
        cdt = datetime.now(CDT)
        current_date = cdt.date()
        start_date = datetime.combine(current_date - timedelta(days=5), datetime.min.time(), tzinfo=CDT)
        end_date = datetime.combine(current_date + timedelta(days=1), datetime.min.time(), tzinfo=CDT)
        
        response = client.price_history(
            symbol=ticker,
            periodType="day",
            period=5,
            frequencyType="minute",
            frequency=frequency,
            startDate=int(start_date.timestamp() * 1000),
            endDate=int(end_date.timestamp() * 1000),
            needExtendedHoursData=True
        )
        
        if not response.ok:
            logger.error(f"Price history response not OK: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        if not data or 'candles' not in data:
            logger.error("No candles in price history data")
            return None
        
        candles = data['candles']
        if not candles:
            logger.error(f"No candles returned for {ticker}")
            return None
        
        # Filter for market hours
        candles = filter_market_hours(candles)
        if not candles:
            logger.error(f"No market hours data for {ticker}")
            return None
        
        # Sort candles by timestamp and remove duplicates
        unique_candles = {}
        for candle in candles:
            candle_time = datetime.fromtimestamp(candle['datetime']/1000, tz=pytz.UTC).astimezone(CDT)
            unique_candles[candle_time] = candle
        
        sorted_candles = sorted(unique_candles.items(), key=lambda x: x[0])
        all_candles = [candle for _, candle in sorted_candles]
        
        # Filter for current day's candles
        current_day_candles = []
        for candle in all_candles:
            candle_time = datetime.fromtimestamp(candle['datetime']/1000, tz=pytz.UTC).astimezone(CDT)
            if candle_time.date() == current_date:
                current_day_candles.append(candle)
        
        # If no current day candles, use the most recent trading day's candles
        if not current_day_candles:
            most_recent_day = max(datetime.fromtimestamp(candle['datetime']/1000, tz=pytz.UTC).astimezone(CDT).date() for candle in all_candles)
            current_day_candles = [candle for candle in all_candles if datetime.fromtimestamp(candle['datetime']/1000, tz=pytz.UTC).astimezone(CDT).date() == most_recent_day]
        
        if not current_day_candles:
            logger.error(f"No current or recent trading day data for {ticker}")
            return None
        
        # Convert to Heikin-Ashi
        ha_candles = convert_to_heikin_ashi(current_day_candles)
        df = pd.DataFrame(ha_candles)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True).dt.tz_convert(CDT).dt.tz_localize(None)
        df = df.sort_values('datetime')
        logger.info(f"Fetched {len(df)} {frequency}-minute Heikin-Ashi candles for {ticker} in market hours")
        return df
    except Exception as e:
        logger.error(f"Error fetching price history for {ticker}: {e}")
        return None

def calculate_rsi(prices, period=8):
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([float('inf'), -float('inf')], 0).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_options_chain(ticker, expiry_date=None):
    if client is None:
        logger.error("Schwab client not initialized")
        return pd.DataFrame(), pd.DataFrame(), None, None
    try:
        if expiry_date is None:
            expirations = get_option_expirations(ticker)
            if not expirations:
                logger.error("No expirations found")
                return pd.DataFrame(), pd.DataFrame(), None, None
            expiry_date = min(expirations)
            logger.info(f"Using nearest expiry: {expiry_date}")

        response = client.option_chains(
            symbol=ticker,
            fromDate=expiry_date,
            toDate=expiry_date,
            contractType='ALL'
        )
        
        if not response.ok:
            logger.error(f"Options chain response not OK: {response.status_code} - {response.text}")
            return pd.DataFrame(), pd.DataFrame(), None, None
        
        chain = response.json()
        S = float(chain.get('underlyingPrice', 0))
        if S == 0:
            S = get_current_price(ticker)
            if S is None:
                logger.error("Could not fetch current price")
                return pd.DataFrame(), pd.DataFrame(), None, None
        
        # Calculate time to expiration
        today = datetime.now(CDT).date()
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
        T = max((expiry - today).days / 365.0, 1/1440)  # Minimum 1 minute
        r = 0.05  # Risk-free rate
        
        calls_data = []
        puts_data = []
        
        for exp_date, strikes in chain.get('callExpDateMap', {}).items():
            for strike, options in strikes.items():
                for option in options:
                    market_price = float(option.get('last', 0))
                    if market_price <= 0:
                        market_price = (float(option.get('bid', 0)) + float(option.get('ask', 0))) / 2
                    K = float(option['strikePrice'])
                    vol = implied_volatility(S, K, T, r, market_price, is_put=False) if market_price > 0 else 0.2
                    delta = black_scholes_delta(S, K, T, r, vol, is_put=False)
                    vanna = black_scholes_vanna(S, K, T, r, vol)
                    charm = black_scholes_charm(S, K, T, r, vol, is_put=False)
                    
                    option_data = {
                        'strike': K,
                        'volume': int(option.get('totalVolume', 0)),
                        'oi': int(option.get('openInterest', 0)),
                        'delta': delta,
                        'vanna': vanna,
                        'charm': charm,
                        'vol': vol
                    }
                    calls_data.append(option_data)
        
        for exp_date, strikes in chain.get('putExpDateMap', {}).items():
            for strike, options in strikes.items():
                for option in options:
                    market_price = float(option.get('last', 0))
                    if market_price <= 0:
                        market_price = (float(option.get('bid', 0)) + float(option.get('ask', 0))) / 2
                    K = float(option['strikePrice'])
                    vol = implied_volatility(S, K, T, r, market_price, is_put=True) if market_price > 0 else 0.2
                    delta = black_scholes_delta(S, K, T, r, vol, is_put=True)
                    vanna = black_scholes_vanna(S, K, T, r, vol)
                    charm = black_scholes_charm(S, K, T, r, vol, is_put=True)
                    
                    option_data = {
                        'strike': K,
                        'volume': int(option.get('totalVolume', 0)),
                        'oi': int(option.get('openInterest', 0)),
                        'delta': delta,
                        'vanna': vanna,
                        'charm': charm,
                        'vol': vol
                    }
                    puts_data.append(option_data)
        
        calls = pd.DataFrame(calls_data)
        puts = pd.DataFrame(puts_data)
        
        calls = calls.groupby('strike').agg({'volume': 'sum', 'oi': 'sum', 'delta': 'mean', 'vanna': 'mean', 'charm': 'mean', 'vol': 'mean'}).reset_index()
        puts = puts.groupby('strike').agg({'volume': 'sum', 'oi': 'sum', 'delta': 'mean', 'vanna': 'mean', 'charm': 'mean', 'vol': 'mean'}).reset_index()
        
        logger.info(f"Fetched {len(calls)} call strikes and {len(puts)} put strikes")
        return calls, puts, S, expiry_date
    except Exception as e:
        logger.error(f"Error fetching options chain for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame(), None, None

def get_option_expirations(ticker):
    if client is None:
        logger.error("Schwab client not initialized")
        return []
    try:
        response = client.option_expiration_chain(ticker)
        if not response.ok:
            logger.error(f"Expiration chain response not OK: {response.status_code} - {response.text}")
            return []
        data = response.json()
        expirations = sorted([item['expirationDate'] for item in data.get('expirationList', [])])
        logger.info(f"Fetched {len(expirations)} expiration dates")
        return expirations
    except Exception as e:
        logger.error(f"Error fetching expirations for {ticker}: {e}")
        return []

def get_nearest_strikes(calls, puts, S, num_strikes=17):
    all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
    if not all_strikes:
        logger.error("No strikes available")
        return pd.DataFrame()
    
    closest_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - S))
    half = num_strikes // 2
    start_idx = max(0, closest_idx - half)
    end_idx = min(len(all_strikes), closest_idx + half + (num_strikes % 2))
    
    selected_strikes = all_strikes[start_idx:end_idx]
    while len(selected_strikes) < num_strikes and end_idx < len(all_strikes):
        end_idx += 1
        selected_strikes = all_strikes[start_idx:end_idx]
    while len(selected_strikes) < num_strikes and start_idx > 0:
        start_idx -= 1
        selected_strikes = all_strikes[start_idx:end_idx]
    
    calls = calls[calls['strike'].isin(selected_strikes)]
    puts = puts[puts['strike'].isin(selected_strikes)]
    
    combined = pd.DataFrame({'strike': selected_strikes})
    combined = combined.merge(calls, on='strike', how='left').rename(columns={'volume': 'call_volume', 'oi': 'call_oi', 'delta': 'call_delta', 'vanna': 'call_vanna', 'charm': 'call_charm', 'vol': 'call_iv'})
    combined = combined.merge(puts, on='strike', how='left').rename(columns={'volume': 'put_volume', 'oi': 'put_oi', 'delta': 'put_delta', 'vanna': 'put_vanna', 'charm': 'put_charm', 'vol': 'put_iv'})
    combined = combined.fillna({'call_volume': 0, 'put_volume': 0, 'call_oi': 0, 'put_oi': 0, 'call_delta': 0, 'put_delta': 0, 'call_vanna': 0, 'put_vanna': 0, 'call_charm': 0, 'put_charm': 0, 'call_iv': 0, 'put_iv': 0})
    combined['net'] = combined['put_volume'] - combined['call_volume']
    combined['net_oi'] = combined['put_oi'] - combined['call_oi']
    combined['net_total'] = combined['net'] + combined['net_oi']
    combined['net_delta_weighted'] = (combined['put_delta'].abs() * combined['put_volume']) - (combined['call_delta'] * combined['call_volume'])
    combined['net_delta_weighted_oi'] = (combined['put_delta'].abs() * combined['put_oi']) - (combined['call_delta'] * combined['call_oi'])
    combined['net_vanna'] = combined['put_vanna'] * combined['put_volume'] - combined['call_vanna'] * combined['call_volume']
    combined['net_vanna_oi'] = combined['put_vanna'] * combined['put_oi'] - combined['call_vanna'] * combined['call_oi']
    combined['net_charm'] = combined['put_charm'] * combined['put_volume'] - combined['call_charm'] * combined['call_volume']
    combined['net_charm_oi'] = combined['put_charm'] * combined['put_oi'] - combined['call_charm'] * combined['call_oi']
    
    # Calculate Put Wall, Call Wall, and POC for volume
    put_wall = combined[combined['net'] > 0]['strike'].iloc[combined[combined['net'] > 0]['net'].argmax()] if (combined['net'] > 0).any() else combined.loc[combined['net'].idxmax()]['strike']
    call_wall = combined[combined['net'] < 0]['strike'].iloc[combined[combined['net'] < 0]['net'].argmin()] if (combined['net'] < 0).any() else combined.loc[combined['net'].idxmin()]['strike']
    
    # Filter strikes between Put Wall and Call Wall
    min_strike = min(put_wall, call_wall)
    max_strike = max(put_wall, call_wall)
    poc_candidates = combined[(combined['strike'] >= min_strike) & (combined['strike'] <= max_strike)]
    poc = poc_candidates.loc[poc_candidates['net'].abs().idxmin()]['strike'] if not poc_candidates.empty else combined.loc[combined['net'].abs().idxmin()]['strike']
    
    # Calculate Put Wall, Call Wall, and POC for OI
    put_wall_oi = combined[combined['net_oi'] > 0]['strike'].iloc[combined[combined['net_oi'] > 0]['net_oi'].argmax()] if (combined['net_oi'] > 0).any() else combined.loc[combined['net_oi'].idxmax()]['strike']
    call_wall_oi = combined[combined['net_oi'] < 0]['strike'].iloc[combined[combined['net_oi'] < 0]['net_oi'].argmin()] if (combined['net_oi'] < 0).any() else combined.loc[combined['net_oi'].idxmin()]['strike']
    
    min_strike_oi = min(put_wall_oi, call_wall_oi)
    max_strike_oi = max(put_wall_oi, call_wall_oi)
    poc_candidates_oi = combined[(combined['strike'] >= min_strike_oi) & (combined['strike'] <= max_strike_oi)]
    poc_oi = poc_candidates_oi.loc[poc_candidates_oi['net_oi'].abs().idxmin()]['strike'] if not poc_candidates_oi.empty else combined.loc[combined['net_oi'].abs().idxmin()]['strike']
    
    # Calculate Put Wall, Call Wall, and POC for total
    put_wall_total = combined[combined['net_total'] > 0]['strike'].iloc[combined[combined['net_total'] > 0]['net_total'].argmax()] if (combined['net_total'] > 0).any() else combined.loc[combined['net_total'].idxmax()]['strike']
    call_wall_total = combined[combined['net_total'] < 0]['strike'].iloc[combined[combined['net_total'] < 0]['net_total'].argmin()] if (combined['net_total'] < 0).any() else combined.loc[combined['net_total'].idxmin()]['strike']
    
    min_strike_total = min(put_wall_total, call_wall_total)
    max_strike_total = max(put_wall_total, call_wall_total)
    poc_candidates_total = combined[(combined['strike'] >= min_strike_total) & (combined['strike'] <= max_strike_total)]
    poc_total = poc_candidates_total.loc[poc_candidates_total['net_total'].abs().idxmin()]['strike'] if not poc_candidates_total.empty else combined.loc[combined['net_total'].abs().idxmin()]['strike']
    
    combined['put_wall'] = put_wall
    combined['call_wall'] = call_wall
    combined['poc'] = poc
    combined['put_wall_oi'] = put_wall_oi
    combined['call_wall_oi'] = call_wall_oi
    combined['poc_oi'] = poc_oi
    combined['put_wall_total'] = put_wall_total
    combined['call_wall_total'] = call_wall_total
    combined['poc_total'] = poc_total
    
    logger.info(f"Selected {len(combined)} strikes, Volume: Put Wall: {put_wall}, Call Wall: {call_wall}, POC: {poc} | OI: Put Wall: {put_wall_oi}, Call Wall: {call_wall_oi}, POC: {poc_oi} | Total: Put Wall: {put_wall_total}, Call Wall: {call_wall_total}, POC: {poc_total}")
    return combined.sort_values('strike')

def create_net_volume_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net'],
        marker_color=colors,
        name='Net (Puts - Calls)'
    ))
    
    # Put Wall (largest positive net volume)
    put_wall = combined['put_wall'].iloc[0]
    put_wall_net = combined[combined['strike'] == put_wall]['net'].iloc[0]
    fig.add_vline(
        x=put_wall,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Put Wall: {put_wall_net:.0f} @ {put_wall:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='white',
            borderwidth=1,
            y=1.05
        )
    )
    
    # Call Wall (largest negative net volume)
    call_wall = combined['call_wall'].iloc[0]
    call_wall_net = combined[combined['strike'] == call_wall]['net'].iloc[0]
    fig.add_vline(
        x=call_wall,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Call Wall: {call_wall_net:.0f} @ {call_wall:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='green',
            borderwidth=1,
            y=0.95
        )
    )
    
    # POC (closest to zero net volume between Put Wall and Call Wall)
    poc = combined['poc'].iloc[0]
    poc_net = combined[combined['strike'] == poc]['net'].iloc[0]
    fig.add_vline(
        x=poc,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"POC: {poc_net:.0f} @ {poc:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='purple',
            borderwidth=1,
            y=0.85
        )
    )
    
    # Current Price
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Net Volume", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net Volume (Puts - Calls)", font=dict(size=14, color='white')),
        height=400,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_net_oi_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net_oi']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net_oi'],
        marker_color=colors,
        name='Net OI (Puts - Calls)'
    ))
    
    # Put Wall OI (largest positive net OI)
    put_wall_oi = combined['put_wall_oi'].iloc[0]
    put_wall_net_oi = combined[combined['strike'] == put_wall_oi]['net_oi'].iloc[0]
    fig.add_vline(
        x=put_wall_oi,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Put Wall OI: {put_wall_net_oi:.0f} @ {put_wall_oi:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='white',
            borderwidth=1,
            y=1.05
        )
    )
    
    # Call Wall OI (largest negative net OI)
    call_wall_oi = combined['call_wall_oi'].iloc[0]
    call_wall_net_oi = combined[combined['strike'] == call_wall_oi]['net_oi'].iloc[0]
    fig.add_vline(
        x=call_wall_oi,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Call Wall OI: {call_wall_net_oi:.0f} @ {call_wall_oi:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='green',
            borderwidth=1,
            y=0.95
        )
    )
    
    # POC OI (closest to zero net OI between Put Wall OI and Call Wall OI)
    poc_oi = combined['poc_oi'].iloc[0]
    poc_net_oi = combined[combined['strike'] == poc_oi]['net_oi'].iloc[0]
    fig.add_vline(
        x=poc_oi,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"POC OI: {poc_net_oi:.0f} @ {poc_oi:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='purple',
            borderwidth=1,
            y=0.85
        )
    )
    
    # Current Price
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Net Open Interest", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net OI (Puts - Calls)", font=dict(size=14, color='white')),
        height=400,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_net_total_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net_total']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net_total'],
        marker_color=colors,
        name='Net Total (Puts - Calls)'
    ))
    
    # Put Wall Total (largest positive net total)
    put_wall_total = combined['put_wall_total'].iloc[0]
    put_wall_net_total = combined[combined['strike'] == put_wall_total]['net_total'].iloc[0]
    fig.add_vline(
        x=put_wall_total,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Put Wall Total: {put_wall_net_total:.0f} @ {put_wall_total:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='white',
            borderwidth=1,
            y=1.05
        )
    )
    
    # Call Wall Total (largest negative net total)
    call_wall_total = combined['call_wall_total'].iloc[0]
    call_wall_net_total = combined[combined['strike'] == call_wall_total]['net_total'].iloc[0]
    fig.add_vline(
        x=call_wall_total,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Call Wall Total: {call_wall_net_total:.0f} @ {call_wall_total:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='green',
            borderwidth=1,
            y=0.95
        )
    )
    
    # POC Total (closest to zero net total between Put Wall Total and Call Wall Total)
    poc_total = combined['poc_total'].iloc[0]
    poc_net_total = combined[combined['strike'] == poc_total]['net_total'].iloc[0]
    fig.add_vline(
        x=poc_total,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"POC Total: {poc_net_total:.0f} @ {poc_total:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='purple',
            borderwidth=1,
            y=0.85
        )
    )
    
    # Current Price
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Total Exposure (Volume + OI)", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net Total (Puts - Calls)", font=dict(size=14, color='white')),
        height=400,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_net_delta_weighted_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net_delta_weighted']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net_delta_weighted'],
        marker_color=colors,
        name='Net Delta-Weighted (Abs(Put Delta) * Vol - Call Delta * Vol)'
    ))
    
    max_net_row = combined.loc[combined['net_delta_weighted'].idxmax()]
    fig.add_vline(
        x=max_net_row['strike'],
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Max Net: {max_net_row['net_delta_weighted']:.0f} @ {max_net_row['strike']:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='white',
            borderwidth=1,
            y=1.05
        )
    )
    
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Net Delta-Weighted Volume", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net Delta-Weighted Volume", font=dict(size=14, color='white')),
        height=300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_net_delta_weighted_oi_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net_delta_weighted_oi']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net_delta_weighted_oi'],
        marker_color=colors,
        name='Net Delta-Weighted OI (Abs(Put Delta) * OI - Call Delta * OI)'
    ))
    
    max_net_row = combined.loc[combined['net_delta_weighted_oi'].idxmax()]
    fig.add_vline(
        x=max_net_row['strike'],
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Max Net OI: {max_net_row['net_delta_weighted_oi']:.0f} @ {max_net_row['strike']:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='white',
            borderwidth=1,
            y=1.05
        )
    )
    
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Net Delta-Weighted Open Interest", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net Delta-Weighted OI", font=dict(size=14, color='white')),
        height=300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_net_vanna_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net_vanna']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net_vanna'],
        marker_color=colors,
        name='Net Vanna (Put Vanna * Vol - Call Vanna * Vol)'
    ))
    
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Net Vanna (Volume)", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net Vanna", font=dict(size=14, color='white')),
        height=300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_net_vanna_oi_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net_vanna_oi']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net_vanna_oi'],
        marker_color=colors,
        name='Net Vanna OI (Put Vanna * OI - Call Vanna * OI)'
    ))
    
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Net Vanna (OI)", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net Vanna OI", font=dict(size=14, color='white')),
        height=300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_net_charm_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net_charm']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net_charm'],
        marker_color=colors,
        name='Net Charm (Put Charm * Vol - Call Charm * Vol)'
    ))
    
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Net Charm (Volume)", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net Charm", font=dict(size=14, color='white')),
        height=300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_net_charm_oi_chart(combined, S):
    fig = go.Figure()
    
    colors = ['green' if val >= 0 else 'red' for val in combined['net_charm_oi']]
    fig.add_trace(go.Bar(
        x=combined['strike'],
        y=combined['net_charm_oi'],
        marker_color=colors,
        name='Net Charm OI (Put Charm * OI - Call Charm * OI)'
    ))
    
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="Net Charm (OI)", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Net Charm OI", font=dict(size=14, color='white')),
        height=300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_spy_candlestick_chart(spy_price_data, put_wall, call_wall, poc, vix_price_data):
    global historical_drop_times
    global historical_rise_times
    if spy_price_data is None or spy_price_data.empty:
        logger.error("No price data for SPY candlestick chart")
        return go.Figure()
    
    fig = go.Figure(data=[go.Candlestick(
        x=spy_price_data['datetime'],
        open=spy_price_data['open'],
        high=spy_price_data['high'],
        low=spy_price_data['low'],
        close=spy_price_data['close'],
        increasing_line_color='white',
        decreasing_line_color='cyan'
    )])
    
    fig.add_hline(
        y=put_wall,
        line_dash="solid",
        line_color="#0DFF00",  # Brighter green (DodgerBlue) for better visibility
        line_width=5,  # Increase line thickness
        annotation_text=f"Put Wall: {put_wall:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color='white', family='Arial'),  # Larger, bold font
        annotation=dict(
            bgcolor='black',
            bordercolor='white',
            borderwidth=1,
            xanchor="left",
            xshift=-10
        )
    )

    fig.add_hline(
        y=call_wall,
        line_dash="solid",
        line_color="#FF0606",
        line_width = 3,
        annotation_text=f"Call Wall: {call_wall:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color='white', family='Arial'),
        annotation=dict(
            bgcolor='black',
            bordercolor='white',
            borderwidth=1,
            xanchor="left",
            xshift=-10
        )
    )

    fig.add_hline(
        y=poc,
        line_dash="dash",
        line_color="#FF9100",
        line_width = 3,
        annotation_text=f"POC: {poc:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color='white', family='Arial'),
        annotation=dict(
            bgcolor='black',
            bordercolor='white',
            borderwidth=1,
            xanchor="left",
            xshift=-10
        )
    )
    
    # Add vertical lines for VIX drop conditions
    new_drop_times = find_vix_drop_times(spy_price_data, vix_price_data, put_wall)
    historical_drop_times.update(new_drop_times)
    for drop_time in historical_drop_times:
        fig.add_vline(
            x=drop_time,
            line_dash="dashdot",
            line_color="#B700FF",
            line_width=4
        )
    
    # Add vertical lines for VIX rise conditions
    new_rise_times = find_vix_rise_times(spy_price_data, vix_price_data, call_wall)
    historical_rise_times.update(new_rise_times)
    for rise_time in historical_rise_times:
        fig.add_vline(
            x=rise_time,
            line_dash="dashdot",
            line_color="#F2FF00",
            line_width=4
        )

    fig.update_layout(
        title=dict(text="1-Minute SPY Heikin-Ashi", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Time", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Price", font=dict(size=14, color='white')),
        height=500,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), tickformat='%H:%M', rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_vix_candlestick_chart(price_data):
    if price_data is None or price_data.empty:
        logger.error("No price data for VIX candlestick chart")
        return go.Figure()
    
    fig = go.Figure(data=[go.Candlestick(
        x=price_data['datetime'],
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        increasing_line_color='yellow',
        decreasing_line_color='purple'
    )])
    
    fig.update_layout(
        title=dict(text="1-Minute VIX Heikin-Ashi", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Time", font=dict(size=14, color='white')),
        yaxis_title=dict(text="VIX", font=dict(size=14, color='white')),
        height=500,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), tickformat='%H:%M', rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_rsi_chart(price_data):
    if price_data is None or price_data.empty:
        logger.error("No price data for RSI chart")
        return go.Figure()
    
    rsi = calculate_rsi(price_data['close'], period=8)
    dates = price_data['datetime']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=rsi,
        mode='lines',
        name='RSI',
        line=dict(color='#00FFFF', width=3)
    ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left", annotation_font=dict(size=14, color='white'))
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom left", annotation_font=dict(size=14, color='white'))
    
    fig.update_layout(
        height=300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), tickformat='%H:%M'),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), range=[0, 100])
    )
    
    return fig

def create_iv_skew_chart(combined, S):
    fig = go.Figure()
    
    # Prepare a single IV series: use put IV for strikes < S, call IV for strikes > S, average for ATM
    combined = combined.sort_values('strike')
    iv_series = []
    for _, row in combined.iterrows():
        if row['strike'] < S:
            iv = row['put_iv']  # Use put IV for OTM puts
        elif row['strike'] > S:
            iv = row['call_iv']  # Use call IV for OTM calls
        else:
            iv = (row['put_iv'] + row['call_iv']) / 2  # Average for ATM
        iv_series.append(iv * 100)  # Convert to percentage
    
    # Plot the connected IV smile
    fig.add_trace(go.Scatter(
        x=combined['strike'],
        y=iv_series,
        mode='lines+markers',
        name='IV Smile',
        line=dict(color='cyan')
    ))
    
    # Highlight OTM Puts portion (left side)
    puts_df = combined[combined['strike'] < S]
    fig.add_trace(go.Scatter(
        x=puts_df['strike'],
        y=[row['put_iv'] * 100 for _, row in puts_df.iterrows()],
        mode='markers',
        name='OTM Puts IV',
        marker=dict(color='red')
    ))
    
    # Highlight OTM Calls portion (right side)
    calls_df = combined[combined['strike'] > S]
    fig.add_trace(go.Scatter(
        x=calls_df['strike'],
        y=[row['call_iv'] * 100 for _, row in calls_df.iterrows()],
        mode='markers',
        name='OTM Calls IV',
        marker=dict(color='green')
    ))
    
    # Current Price vertical bar
    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Price: {S:.2f}",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(size=14, color='white', family='Arial'),
            bgcolor='black',
            bordercolor='yellow',
            borderwidth=1,
            y=-0.1
        )
    )
    
    fig.update_layout(
        title=dict(text="IV Skew", font=dict(size=18, color='white')),
        xaxis_title=dict(text="Strike", font=dict(size=14, color='white')),
        yaxis_title=dict(text="Implied Volatility (%)", font=dict(size=14, color='white')),
        height=400,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'), dtick=1),
        yaxis=dict(gridcolor='gray', zerolinecolor='gray', tickfont=dict(size=12, color='white'))
    )
    
    return fig

def create_volume_table(combined):
    max_net_strike = combined.loc[combined['net'].idxmax()]['strike']
    
    table_html = '<table style="width:100%; border-collapse: collapse; background-color: black; color: white;">'
    table_html += '<tr><th>Strike</th><th>Call Volume</th><th>Put Volume</th><th>Net Vol (Puts - Calls)</th><th>Call OI</th><th>Put OI</th><th>Net OI (Puts - Calls)</th><th>Net Total (Vol + OI)</th><th>Net Delta-Weighted Vol</th><th>Net Delta-Weighted OI</th><th>Net Vanna Vol</th><th>Net Vanna OI</th><th>Net Charm Vol</th><th>Net Charm OI</th></tr>'
    
    for _, row in combined.iterrows():
        bg_color = 'background-color: yellow; color: black;' if row['strike'] == max_net_strike else ''
        table_html += f'<tr style="{bg_color}">'
        table_html += f'<td>{row["strike"]:.2f}</td>'
        table_html += f'<td>{row["call_volume"]:.0f}</td>'
        table_html += f'<td>{row["put_volume"]:.0f}</td>'
        table_html += f'<td>{row["net"]:.0f}</td>'
        table_html += f'<td>{row["call_oi"]:.0f}</td>'
        table_html += f'<td>{row["put_oi"]:.0f}</td>'
        table_html += f'<td>{row["net_oi"]:.0f}</td>'
        table_html += f'<td>{row["net_total"]:.0f}</td>'
        table_html += f'<td>{row["net_delta_weighted"]:.0f}</td>'
        table_html += f'<td>{row["net_delta_weighted_oi"]:.0f}</td>'
        table_html += f'<td>{row["net_vanna"]:.0f}</td>'
        table_html += f'<td>{row["net_vanna_oi"]:.0f}</td>'
        table_html += f'<td>{row["net_charm"]:.0f}</td>'
        table_html += f'<td>{row["net_charm_oi"]:.0f}</td>'
        table_html += '</tr>'
    
    table_html += '</table>'
    return table_html

def capture_combined_screenshot(combined, S, spy_price_data, vix_price_data, put_wall, call_wall, poc, expiry_date, iv_skew_fig):
    try:
        volume_fig = create_net_volume_chart(combined, S)
        oi_fig = create_net_oi_chart(combined, S)
        total_fig = create_net_total_chart(combined, S)
        delta_weighted_fig = create_net_delta_weighted_chart(combined, S)
        delta_weighted_oi_fig = create_net_delta_weighted_oi_chart(combined, S)
        vanna_oi_fig = create_net_vanna_oi_chart(combined, S)
        charm_oi_fig = create_net_charm_oi_chart(combined, S)
        iv_skew_fig = create_iv_skew_chart(combined, S)
        spy_candle_fig = create_spy_candlestick_chart(spy_price_data, put_wall, call_wall, poc, vix_price_data) if spy_price_data is not None and not spy_price_data.empty else None
        vix_candle_fig = create_vix_candlestick_chart(vix_price_data) if vix_price_data is not None and not vix_price_data.empty else None
        rsi_fig = create_rsi_chart(spy_price_data) if spy_price_data is not None and not spy_price_data.empty else None
        
        # Generate individual images
        volume_img = pio.to_image(volume_fig, format='png', width=800, height=400)
        oi_img = pio.to_image(oi_fig, format='png', width=800, height=400)
        total_img = pio.to_image(total_fig, format='png', width=800, height=400)
        delta_weighted_img = pio.to_image(delta_weighted_fig, format='png', width=800, height=300)
        delta_weighted_oi_img = pio.to_image(delta_weighted_oi_fig, format='png', width=800, height=300)
        vanna_oi_img = pio.to_image(vanna_oi_fig, format='png', width=800, height=300)
        charm_oi_img = pio.to_image(charm_oi_fig, format='png', width=800, height=300)
        iv_skew_img = pio.to_image(iv_skew_fig, format='png', width=800, height=300)
        
        spy_candle_img = pio.to_image(spy_candle_fig, format='png', width=1600, height=1200) if spy_candle_fig else None
        rsi_img = pio.to_image(rsi_fig, format='png', width=1600, height=444) if rsi_fig else None
        vix_candle_img = pio.to_image(vix_candle_fig, format='png', width=1600, height=900) if vix_candle_fig else None
        
        # Left column images (options charts)
        left_images = [
            Image.open(io.BytesIO(volume_img)),
            Image.open(io.BytesIO(oi_img)),
            Image.open(io.BytesIO(total_img)),
            Image.open(io.BytesIO(delta_weighted_img)),
            Image.open(io.BytesIO(delta_weighted_oi_img)),
            Image.open(io.BytesIO(vanna_oi_img)),
            Image.open(io.BytesIO(charm_oi_img)),
            Image.open(io.BytesIO(iv_skew_img))
        ]
        
        # Right column images (SPY, VIX, RSI)
        right_images = []
        if spy_candle_img:
            right_images.append(Image.open(io.BytesIO(spy_candle_img)))
        if rsi_img:
            right_images.append(Image.open(io.BytesIO(rsi_img)))
        if vix_candle_img:
            right_images.append(Image.open(io.BytesIO(vix_candle_img)))
        
# Calculate heights
        left_height = sum(img.height for img in left_images)
        right_height = sum(img.height for img in right_images)
        max_column_height = max(left_height, right_height)
        
        left_column_width = 800  # Left column width for options charts
        right_column_width = 1600  # Right column width for SPY, VIX, RSI
        total_width = left_column_width + right_column_width  # Total width is 2000 pixels
        
        # Load logo and resize preserving aspect ratio
        logo = Image.open('static/logo.png')
        aspect_ratio = logo.width / logo.height
        logo_height = 777  # Target height, adjust as needed
        logo_width = int(logo_height * aspect_ratio)
        if logo_width > total_width:
            logo_width = total_width
            logo_height = int(logo_width / aspect_ratio)
        logo = logo.resize((logo_width, logo_height), Image.LANCZOS)
        
        # Space for title
        title_height = 50  # Adjust as needed
        
        # Create combined image with space for logo and title at top, then columns below
        total_height = logo_height + title_height + max_column_height
        combined_img = Image.new('RGB', (total_width, total_height), color='black')
        
        # Paste logo centered at top
        logo_x = (total_width - logo_width) // 2
        combined_img.paste(logo, (logo_x, 0))
        
        # Draw title below logo, centered
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("arial.ttf", 53)  # Adjust font path if needed
        except IOError:
            font = ImageFont.load_default()
        text = f"ScalpNet - Data for expiration: {expiry_date}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (total_width - text_width) // 2
        text_y = logo_height + (title_height - text_height) // 2
        draw.text((text_x, text_y), text, fill="white", font=font)
        
        # Paste left column charts
        y_offset = logo_height + title_height
        for img in left_images:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height
        
# Paste right column charts
        y_offset = logo_height + title_height
        for img in right_images:
            combined_img.paste(img, (left_column_width, y_offset))  # Start right column at 800
            y_offset += img.height
        img_buffer = io.BytesIO()
        combined_img.save(img_buffer, format='PNG')
        logger.info("Combined chart screenshot captured successfully")
                # ---- CLEANUP FIX ----
        for fig in [volume_fig, oi_fig, total_fig, delta_weighted_fig, delta_weighted_oi_fig, 
                    vanna_oi_fig, charm_oi_fig, iv_skew_fig, spy_candle_fig, rsi_fig, vix_candle_fig]:
            try:
                if fig is not None:
                    fig.data = []  # clear figure data
                    del fig
            except Exception:
                pass

        # Close Pillow images to free file descriptors
        for img in left_images + right_images:
            try:
                img.close()
            except Exception:
                pass

        # Close logo
        try:
            logo.close()
        except Exception:
            pass

        return img_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error capturing combined screenshot: {e}")
        return None
    
def send_discord_alert():
    global last_alert_time
    global previous_put_wall
    global historical_drop_times
    logger.info("Starting Discord alert process")
    
    # Check if current time is within 8:30 AM - 3:00 PM CDT
    cdt = pytz.timezone('America/Chicago')
    current_time_cdt = datetime.now(cdt)
    if current_time_cdt.weekday() >= 5:  # Weekend
        logger.info("Skipping Discord alerts: Outside market days")
        return
    
    
    market_open = current_time_cdt.replace(hour=8, minute=30, second=0, microsecond=0)
    market_close = current_time_cdt.replace(hour=15, minute=0, second=0, microsecond=0)
    if not (market_open <= current_time_cdt <= market_close):
        logger.info("Skipping Discord alerts: Outside 8:30 AM - 3:00 PM CDT")
        return
    try:
        ticker = 'SPY'
        calls, puts, S, expiry_date = fetch_options_chain(ticker)
        if calls.empty or puts.empty or S is None:
            logger.error("No valid options data available for Discord alert")
            return
        
        combined = get_nearest_strikes(calls, puts, S, 17)
        if combined.empty:
            logger.error("No strikes selected for Discord alert")
            return
        
        put_wall = combined['put_wall'].iloc[0]
        call_wall = combined['call_wall'].iloc[0]
        poc = combined['poc'].iloc[0]
        put_wall_net = combined[combined['strike'] == put_wall]['net'].iloc[0]
        call_wall_net = combined[combined['strike'] == call_wall]['net'].iloc[0]
        poc_net = combined[combined['strike'] == poc]['net'].iloc[0]
        
        put_wall_oi = combined['put_wall_oi'].iloc[0]
        call_wall_oi = combined['call_wall_oi'].iloc[0]
        poc_oi = combined['poc_oi'].iloc[0]
        put_wall_net_oi = combined[combined['strike'] == put_wall_oi]['net_oi'].iloc[0]
        call_wall_net_oi = combined[combined['strike'] == call_wall_oi]['net_oi'].iloc[0]
        poc_net_oi = combined[combined['strike'] == poc_oi]['net_oi'].iloc[0]
        
        put_wall_total = combined['put_wall_total'].iloc[0]
        call_wall_total = combined['call_wall_total'].iloc[0]
        poc_total = combined['poc_total'].iloc[0]
        put_wall_net_total = combined[combined['strike'] == put_wall_total]['net_total'].iloc[0]
        call_wall_net_total = combined[combined['strike'] == call_wall_total]['net_total'].iloc[0]
        poc_net_total = combined[combined['strike'] == poc_total]['net_total'].iloc[0]
        
        # Check for put wall change
        if previous_put_wall is not None and put_wall != previous_put_wall:
            change_payload = {
                'content': f"**Put Wall Change Alert**\nPut Wall changed from ${previous_put_wall:.2f} to ${put_wall:.2f}"
            }
            for webhook_url in DISCORD_WEBHOOK_URLS:
                try:
                    response = requests.post(webhook_url, data={'payload_json': json.dumps(payload)}, files=files)
# Discord returns 204 (no content) for JSON-only posts and 200 (OK) when files are attached
                    if response.status_code in (200, 204):
                         logger.info(f"✅ Regular Discord alert sent successfully to {webhook_url} ({response.status_code})")
                    else:
                         logger.error(f"❌ Failed to send regular Discord alert to {webhook_url}: {response.status_code} - {response.text}")

                except Exception as e:
                    logger.error(f"Error sending put wall change alert to {webhook_url}: {e}")
        previous_put_wall = put_wall
        
        spy_price_data = get_price_history(ticker, frequency=1)
        vix_price_data = get_price_history('$VIX', frequency=1)
        rsi_available = spy_price_data is not None and not spy_price_data.empty
        rsi_text = f"RSI: {calculate_rsi(spy_price_data['close'], period=8).iloc[-1]:.2f}" if rsi_available else "RSI: Not available"
        # Generate IV skew chart
        iv_skew_fig = create_iv_skew_chart(combined, S)

        screenshot = capture_combined_screenshot(combined, S, spy_price_data, vix_price_data, put_wall, call_wall, poc, expiry_date, iv_skew_fig)
        if not screenshot:
            logger.error("Failed to capture combined screenshot")
            return
        
        # Prepare regular Discord payload
        payload = {
            'content': f"**SPY Options Update 0-DTE**\nCurrent Price: ${S:.2f}\n**Volume:**\nPut Wall: ${put_wall:.2f} (Net Volume: {put_wall_net:.0f})\nCall Wall: ${call_wall:.2f} (Net Volume: {call_wall_net:.0f})\nPOC: ${poc:.2f} (Net Volume: {poc_net:.0f})\n**OI:**\nPut Wall OI: ${put_wall_oi:.2f} (Net OI: {put_wall_net_oi:.0f})\nCall Wall OI: ${call_wall_oi:.2f} (Net OI: {call_wall_net_oi:.0f})\nPOC OI: ${poc_oi:.2f} (Net OI: {poc_net_oi:.0f})\n**Total (Vol + OI):**\nPut Wall Total: ${put_wall_total:.2f} (Net Total: {put_wall_net_total:.0f})\nCall Wall Total: ${call_wall_total:.2f} (Net Total: {call_wall_net_total:.0f})\nPOC Total: ${poc_total:.2f} (Net Total: {poc_net_total:.0f})\n{rsi_text}"
        }
        logger.info(f"Prepared regular Discord payload: {payload['content']}")
        
        files = {
            'file': ('charts.png', screenshot, 'image/png')
        }
        
        # Send regular alert to all Discord webhooks
        for webhook_url in DISCORD_WEBHOOK_URLS:
            try:
                response = requests.post(webhook_url, data={'payload_json': json.dumps(payload)}, files=files)
                if response.status_code == 204:
                    logger.info(f"Regular Discord alert sent successfully to {webhook_url}")
                else:
                    logger.error(f"Failed to send regular Discord alert to {webhook_url}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error sending regular Discord alert to {webhook_url}: {e}")
        
        # Check for special alert condition
        drop_times = find_vix_drop_times(spy_price_data, vix_price_data, put_wall)
        if drop_times and S <= put_wall and time.time() - last_alert_time >= 300:  # 5 minutes
            latest_drop = max(drop_times)
            current_time = datetime.now(CDT)
            if (current_time - latest_drop).total_seconds() < 300:  # Recent VIX drop within 5 minutes
                special_payload = {
                    'content': f"**SPY Rebound Alert**\nPrice at or below Put Wall: ${S:.2f}\nVIX Dropping\nEntry Price: ${S:.2f}\nExit Price (POC): ${poc:.2f}"
                }
                logger.info(f"Prepared special Discord payload: {special_payload['content']}")
                for webhook_url in DISCORD_WEBHOOK_URLS:
                    try:
                        special_response = requests.post(webhook_url, json=special_payload)
                        if special_response.status_code in (200, 204):
                            logger.info(f"✅ Special Discord alert sent successfully to {webhook_url} ({special_response.status_code})")
                        else:
                            logger.error(f"❌ Failed to send special Discord alert to {webhook_url}: {special_response.status_code} - {special_response.text}")

                    except Exception as e:
                        logger.error(f"Error sending special Discord alert to {webhook_url}: {e}")
                last_alert_time = time.time()
    except Exception as e:
        logger.error(f"Error in send_discord_alert: {e}")

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>SPY Options Volume Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background: black; padding: 20px; color: white; }
        #chart, #oi-chart, #total-chart, #iv-skew-chart, #delta-weighted-chart, #delta-weighted-oi-chart, #vanna-oi-chart, #charm-oi-chart, #vomma-oi-chart { width: 100%; height: 400px; }
        #spy-candle-chart, #vix-candle-chart { width: 100%; height: 500px; }
        #rsi-chart { width: 100%; height: 400px; }
        table { border: 1px solid #444; margin-top: 20px; background: black; color: white; }
        th, td { padding: 8px; text-align: center; border: 1px solid #444; }
        th { background: #222; }
        h1 { color: white; }
    </style>
</head>
<body>
    <div style="text-align: center;">
        <img src="{{ url_for('static', filename='logo.png') }}" alt="ScalpNet Logo" style="max-width: 300px;">
        <h1 id="title"></h1>
    </div>
    <h1>SPY Put/Call Volume</h1>
    <div id="chart"></div>
    <h1>SPY Put/Call Open Interest</h1>
    <div id="oi-chart"></div>
    <h1>SPY Put/Call Total Exposure (Volume + OI)</h1>
    <div id="total-chart"></div>
    <h1>IV Skew/Smile</h1>
    <div id="iv-skew-chart"></div>
    <h1>Net Delta-Weighted Volume</h1>
    <div id="delta-weighted-chart"></div>
    <h1>Net Delta-Weighted Open Interest</h1>
    <div id="delta-weighted-oi-chart"></div>
    <h1>Net Vanna (OI)</h1>
    <div id="vanna-oi-chart"></div>
    <h1>Net Charm (OI)</h1>
    <div id="charm-oi-chart"></div>
    <h1>1-Minute SPY Heikin-Ashi Candlestick</h1>
    <div id="spy-candle-chart"></div>
    <h1>1-Minute VIX Heikin-Ashi Candlestick</h1>
    <div id="vix-candle-chart"></div>
    <h1>1-Minute SPY RSI (8-period)</h1>
    <div id="rsi-chart"></div>
    <div id="table"></div>
    <script>
        function updateDashboard() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('title').innerHTML = `ScalpNet - Data for expiration: ${data.expiry_date}`;
                    Plotly.newPlot('chart', JSON.parse(data.chart).data, JSON.parse(data.chart).layout);
                    Plotly.newPlot('oi-chart', JSON.parse(data.oi_chart).data, JSON.parse(data.oi_chart).layout);
                    Plotly.newPlot('total-chart', JSON.parse(data.total_chart).data, JSON.parse(data.total_chart).layout);
                    Plotly.newPlot('iv-skew-chart', JSON.parse(data.iv_skew_chart).data, JSON.parse(data.iv_skew_chart).layout);
                    Plotly.newPlot('delta-weighted-chart', JSON.parse(data.delta_weighted_chart).data, JSON.parse(data.delta_weighted_chart).layout);
                    Plotly.newPlot('delta-weighted-oi-chart', JSON.parse(data.delta_weighted_oi_chart).data, JSON.parse(data.delta_weighted_oi_chart).layout);
                    Plotly.newPlot('vanna-oi-chart', JSON.parse(data.vanna_oi_chart).data, JSON.parse(data.vanna_oi_chart).layout);
                    Plotly.newPlot('charm-oi-chart', JSON.parse(data.charm_oi_chart).data, JSON.parse(data.charm_oi_chart).layout);
                    Plotly.newPlot('spy-candle-chart', JSON.parse(data.spy_candle_chart).data, JSON.parse(data.spy_candle_chart).layout);
                    Plotly.newPlot('vix-candle-chart', JSON.parse(data.vix_candle_chart).data, JSON.parse(data.vix_candle_chart).layout);
                    Plotly.newPlot('rsi-chart', JSON.parse(data.rsi_chart).data, JSON.parse(data.rsi_chart).layout);
                    document.getElementById('table').innerHTML = data.table;
                })
                .catch(error => console.error('Error updating dashboard:', error));
        }
        updateDashboard();
        setInterval(updateDashboard, 15000);  // Update every 15 seconds
    </script>
</body>
</html>
    ''')

@app.route('/data')
def get_data():
    ticker = 'SPY'
    calls, puts, S, expiry_date = fetch_options_chain(ticker)
    if calls.empty or puts.empty or S is None:
        logger.error("No data available for /data endpoint")
        return jsonify({'error': 'No data available'})
    
    combined = get_nearest_strikes(calls, puts, S, 17)
    chart = create_net_volume_chart(combined, S).to_json()
    oi_chart = create_net_oi_chart(combined, S).to_json()
    total_chart = create_net_total_chart(combined, S).to_json()
    iv_skew_chart = create_iv_skew_chart(combined, S).to_json()
    delta_weighted_chart = create_net_delta_weighted_chart(combined, S).to_json()
    delta_weighted_oi_chart = create_net_delta_weighted_oi_chart(combined, S).to_json()
    vanna_oi_chart = create_net_vanna_oi_chart(combined, S).to_json()
    charm_oi_chart = create_net_charm_oi_chart(combined, S).to_json()
    table = create_volume_table(combined)
    
    spy_price_data = get_price_history(ticker, frequency=1)
    vix_price_data = get_price_history('$VIX', frequency=1)
    put_wall = combined['put_wall'].iloc[0] if not combined.empty else S
    call_wall = combined['call_wall'].iloc[0] if not combined.empty else S
    poc = combined['poc'].iloc[0] if not combined.empty else S
    spy_candle_chart = create_spy_candlestick_chart(spy_price_data, put_wall, call_wall, poc, vix_price_data).to_json() if spy_price_data is not None and not spy_price_data.empty else go.Figure().to_json()
    vix_candle_chart = create_vix_candlestick_chart(vix_price_data).to_json() if vix_price_data is not None and not vix_price_data.empty else go.Figure().to_json()
    rsi_chart = create_rsi_chart(spy_price_data).to_json() if spy_price_data is not None and not spy_price_data.empty else go.Figure().to_json()
    
    return jsonify({
        'chart': chart,
        'oi_chart': oi_chart,
        'total_chart': total_chart,
        'iv_skew_chart': iv_skew_chart,
        'delta_weighted_chart': delta_weighted_chart,
        'delta_weighted_oi_chart': delta_weighted_oi_chart,
        'vanna_oi_chart': vanna_oi_chart,
        'charm_oi_chart': charm_oi_chart,
        'spy_candle_chart': spy_candle_chart,
        'vix_candle_chart': vix_candle_chart,
        'table': table,
        'rsi_chart': rsi_chart,
        'expiry_date': expiry_date
    })

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(send_discord_alert, 'interval', minutes=2, next_run_time=datetime.now(CDT))
scheduler.start()
logger.info("Scheduler started")

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5002, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler")
        scheduler.shutdown()