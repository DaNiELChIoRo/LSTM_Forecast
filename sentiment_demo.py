#!/usr/bin/env python3
"""
Market Sentiment Analysis Demo
This script demonstrates the sentiment analysis features without running the full LSTM model.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_fear_greed_index():
    """Fetch Fear & Greed Index from alternative.me API"""
    try:
        url = "https://api.alternative.me/fng/?limit=10"
        response = requests.get(url)
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            current_fng = float(data['data'][0]['value'])
            timestamp = data['data'][0]['timestamp']
            return current_fng / 100.0, timestamp
        else:
            return 0.5, None
    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return 0.5, None

def get_vix_data():
    """Fetch VIX (Volatility Index) data"""
    try:
        vix = yf.download('^VIX', period='1mo', interval='1d')
        if not vix.empty:
            latest_vix = vix['Close'].iloc[-1]
            normalized_vix = min(max((latest_vix - 10) / 70, 0), 1)
            return 1 - normalized_vix, latest_vix  # Invert so high VIX = low sentiment
        else:
            return 0.5, None
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return 0.5, None

def get_market_breadth_sentiment():
    """Analyze market breadth using major indices"""
    try:
        indices = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'}
        performance_data = {}
        sentiment_scores = []
        
        for symbol, name in indices.items():
            data = yf.download(symbol, period='5d', interval='1d')
            if not data.empty and len(data) >= 2:
                performance = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
                performance_data[name] = performance * 100  # Convert to percentage
                sentiment = 0.5 + (performance * 2)  # Convert to sentiment score
                sentiment_scores.append(max(min(sentiment, 1), 0))
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
        return avg_sentiment, performance_data
    except Exception as e:
        print(f"Error calculating market breadth: {e}")
        return 0.5, {}

def get_crypto_sentiment():
    """Get cryptocurrency market sentiment"""
    try:
        btc = yf.download('BTC-USD', period='7d', interval='1d')
        if not btc.empty and len(btc) >= 2:
            btc_performance = (btc['Close'].iloc[-1] - btc['Close'].iloc[0]) / btc['Close'].iloc[0]
            btc_price = btc['Close'].iloc[-1]
            sentiment = 0.5 + (btc_performance * 1.5)
            return max(min(sentiment, 1), 0), btc_performance * 100, btc_price
        return 0.5, 0, 0
    except Exception as e:
        print(f"Error fetching crypto sentiment: {e}")
        return 0.5, 0, 0

def analyze_market_sentiment():
    """Comprehensive market sentiment analysis"""
    
    print("🔍 MARKET SENTIMENT ANALYSIS")
    print("=" * 50)
    
    # Get Fear & Greed Index
    fear_greed, fg_timestamp = get_fear_greed_index()
    print(f"📊 Fear & Greed Index: {fear_greed:.3f} ({fear_greed*100:.1f}/100)")
    
    if fear_greed < 0.25:
        fg_interpretation = "🔴 EXTREME FEAR"
    elif fear_greed < 0.45:
        fg_interpretation = "🟠 FEAR"
    elif fear_greed < 0.55:
        fg_interpretation = "🟡 NEUTRAL"
    elif fear_greed < 0.75:
        fg_interpretation = "🟢 GREED"
    else:
        fg_interpretation = "🔥 EXTREME GREED"
    
    print(f"   Interpretation: {fg_interpretation}")
    
    # Get VIX data
    vix_sentiment, vix_value = get_vix_data()
    print(f"\n📈 VIX Sentiment: {vix_sentiment:.3f}")
    if vix_value:
        print(f"   Current VIX: {vix_value:.2f}")
        if vix_value < 15:
            vix_interpretation = "🟢 LOW VOLATILITY (Complacency)"
        elif vix_value < 25:
            vix_interpretation = "🟡 NORMAL VOLATILITY"
        elif vix_value < 35:
            vix_interpretation = "🟠 HIGH VOLATILITY (Concern)"
        else:
            vix_interpretation = "🔴 EXTREME VOLATILITY (Panic)"
        print(f"   Interpretation: {vix_interpretation}")
    
    # Get market breadth
    breadth_sentiment, performance_data = get_market_breadth_sentiment()
    print(f"\n📊 Market Breadth Sentiment: {breadth_sentiment:.3f}")
    print("   Recent Performance (5-day):")
    for index, perf in performance_data.items():
        color = "🟢" if perf > 0 else "🔴"
        print(f"     {color} {index}: {perf:+.2f}%")
    
    # Get crypto sentiment
    crypto_sentiment, btc_performance, btc_price = get_crypto_sentiment()
    print(f"\n₿ Crypto Sentiment: {crypto_sentiment:.3f}")
    print(f"   Bitcoin 7-day performance: {btc_performance:+.2f}%")
    print(f"   Current BTC price: ${btc_price:,.2f}")
    
    # Calculate composite sentiment
    weights = {'fear_greed': 0.3, 'vix': 0.25, 'breadth': 0.25, 'crypto': 0.2}
    
    composite = (
        fear_greed * weights['fear_greed'] +
        vix_sentiment * weights['vix'] +
        breadth_sentiment * weights['breadth'] +
        crypto_sentiment * weights['crypto']
    )
    
    print(f"\n🎯 COMPOSITE SENTIMENT SCORE: {composite:.3f}")
    
    if composite < 0.3:
        overall_sentiment = "🔴 VERY BEARISH"
        recommendation = "High caution advised. Consider defensive positions."
    elif composite < 0.45:
        overall_sentiment = "🟠 BEARISH"
        recommendation = "Cautious approach recommended. Look for oversold opportunities."
    elif composite < 0.55:
        overall_sentiment = "🟡 NEUTRAL"
        recommendation = "Balanced approach. Monitor key levels closely."
    elif composite < 0.7:
        overall_sentiment = "🟢 BULLISH"
        recommendation = "Positive outlook. Consider growth positions."
    else:
        overall_sentiment = "🔥 VERY BULLISH"
        recommendation = "Strong optimism, but watch for overheating signs."
    
    print(f"Overall Market Sentiment: {overall_sentiment}")
    print(f"Trading Recommendation: {recommendation}")
    
    # Create visualization
    create_sentiment_visualization(fear_greed, vix_sentiment, breadth_sentiment, 
                                 crypto_sentiment, composite)
    
    return {
        'fear_greed': fear_greed,
        'vix': vix_sentiment,
        'breadth': breadth_sentiment,
        'crypto': crypto_sentiment,
        'composite': composite,
        'interpretation': overall_sentiment,
        'recommendation': recommendation
    }

def create_sentiment_visualization(fear_greed, vix_sentiment, breadth_sentiment, 
                                 crypto_sentiment, composite):
    """Create a visualization of sentiment indicators"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Individual sentiment indicators
    indicators = ['Fear & Greed', 'VIX', 'Market Breadth', 'Crypto', 'Composite']
    values = [fear_greed, vix_sentiment, breadth_sentiment, crypto_sentiment, composite]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    bars = ax1.bar(indicators, values, color=colors, alpha=0.7)
    ax1.set_title('Market Sentiment Indicators', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Sentiment Score (0-1)')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, label='Bearish Threshold')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Bullish Threshold')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sentiment gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    ax2 = plt.subplot(2, 1, 2, projection='polar')
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location('W')
    
    # Create color gradient for gauge
    colors_gauge = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    n_segments = len(colors_gauge)
    
    for i, color in enumerate(colors_gauge):
        theta_start = i * np.pi / n_segments
        theta_end = (i + 1) * np.pi / n_segments
        theta_segment = np.linspace(theta_start, theta_end, 20)
        r_segment = np.ones_like(theta_segment)
        ax2.fill_between(theta_segment, 0, r_segment, color=color, alpha=0.6)
    
    # Add sentiment needle
    composite_angle = composite * np.pi
    ax2.arrow(composite_angle, 0, 0, 0.8, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=3)
    
    ax2.set_ylim(0, 1)
    ax2.set_title('Composite Sentiment Gauge', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax2.set_xticklabels(['Bearish', 'Cautious', 'Neutral', 'Optimistic', 'Bullish'])
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('market_sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Sentiment visualization saved as 'market_sentiment_analysis.png'")

if __name__ == "__main__":
    print("🚀 Starting Market Sentiment Analysis Demo...")
    print("This will analyze current market conditions using multiple indicators.\n")
    
    try:
        sentiment_data = analyze_market_sentiment()
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Check the generated chart: market_sentiment_analysis.png")
        print(f"\n💡 How this enhances LSTM predictions:")
        print(f"   • Sentiment score ({sentiment_data['composite']:.3f}) is added as a feature")
        print(f"   • Model learns correlation between sentiment and price movements")
        print(f"   • Helps predict market reactions to sentiment shifts")
        print(f"   • Improves accuracy during high volatility periods")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check your internet connection and try again.")
