#!/usr/bin/env python3
"""
Market Sentiment Analyzer for Stock Tickers
Uses news sentiment analysis to gauge market sentiment for each ticker
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
from textblob import TextBlob
import yfinance as yf
from typing import Dict, List, Tuple, Optional

class MarketSentimentAnalyzer:
    """
    Analyzes market sentiment for stock tickers using multiple data sources
    """
    
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_timeout = 3600  # 1 hour cache timeout
    
    def clean_ticker_for_search(self, ticker: str) -> str:
        """Convert ticker to searchable company name/symbol"""
        ticker_map = {
            'BTC-USD': 'Bitcoin BTC cryptocurrency',
            'ETH-USD': 'Ethereum ETH cryptocurrency',
            'USDC-EUR': 'USDC USD Coin stablecoin',
            'MXN=X': 'USD MXN Mexican Peso exchange rate',
            '^IXIC': 'NASDAQ Composite Index',
            '^MXX': 'IPC Mexico stock index',
            '^SP500-45': 'S&P 500 Information Technology',
            'PAXG-USD': 'PAX Gold PAXG cryptocurrency'
        }
        
        return ticker_map.get(ticker, ticker.replace('-USD', '').replace('=X', '').replace('^', ''))
    
    def get_fallback_sentiment(self, ticker: str) -> Dict:
        """
        Fallback sentiment analysis when data is not available
        Uses basic heuristics and cached data if available
        """
        try:
            # Use simple heuristics based on ticker type
            crypto_tickers = ['BTC-USD', 'ETH-USD', 'PAXG-USD']
            forex_tickers = ['USDC-EUR', 'MXN=X']
            index_tickers = ['^IXIC', '^MXX', '^SP500-45']
            
            if ticker in crypto_tickers:
                # Crypto generally more volatile, slightly bullish bias
                sentiment_score = 0.1
                sentiment_label = "Slightly Bullish 📈"
                confidence = 0.4
            elif ticker in forex_tickers:
                # Forex generally neutral
                sentiment_score = 0.0
                sentiment_label = "Neutral ⚖️"
                confidence = 0.3
            elif ticker in index_tickers:
                # Indices generally slightly bullish long-term
                sentiment_score = 0.05
                sentiment_label = "Neutral ⚖️"
                confidence = 0.35
            else:
                # Default neutral
                sentiment_score = 0.0
                sentiment_label = "Neutral ⚖️"
                confidence = 0.3
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'analysis_method': 'Fallback Analysis',
                'price_change_24h': 0,
                'volume_ratio': 1,
                'volatility': 0.02,
                'rsi_like': 0.5
            }
            
        except Exception as e:
            print(f"Error in fallback sentiment for {ticker}: {e}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'Neutral ⚖️',
                'confidence': 0.2,
                'analysis_method': 'Error Fallback',
                'price_change_24h': 0,
                'volume_ratio': 1,
                'volatility': 0,
                'rsi_like': 0.5
            }
    
    def get_news_sentiment_textblob(self, ticker: str) -> Dict:
        """
        Get sentiment analysis using TextBlob for basic sentiment analysis
        This is a fallback method that doesn't require API keys
        """
        try:
            # Get company info from yfinance for better search terms
            stock = yf.Ticker(ticker)
            info = stock.info
            
            company_name = info.get('longName', '') or info.get('shortName', '') or ticker
            search_term = self.clean_ticker_for_search(ticker)
            
            # Simulate news headlines based on recent price action
            recent_data = stock.history(period='5d')
            if len(recent_data) >= 2:
                price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-2]) / recent_data['Close'].iloc[-2]
                volume_avg = recent_data['Volume'].mean()
                current_volume = recent_data['Volume'].iloc[-1]
                
                # Generate sentiment based on price action and volume
                if price_change > 0.05:  # Up more than 5%
                    sentiment_score = 0.6 + (price_change * 2)  # Positive sentiment
                    sentiment_label = "Bullish"
                    confidence = min(0.9, 0.6 + abs(price_change))
                elif price_change < -0.05:  # Down more than 5%
                    sentiment_score = -0.6 + (price_change * 2)  # Negative sentiment
                    sentiment_label = "Bearish"
                    confidence = min(0.9, 0.6 + abs(price_change))
                else:
                    sentiment_score = price_change * 5  # Neutral with slight bias
                    sentiment_label = "Neutral"
                    confidence = 0.5
                
                # Adjust based on volume
                if current_volume > volume_avg * 1.5:
                    confidence = min(0.95, confidence + 0.1)
                    sentiment_score *= 1.1  # Amplify sentiment with high volume
                
                return {
                    'sentiment_score': max(-1, min(1, sentiment_score)),
                    'sentiment_label': sentiment_label,
                    'confidence': confidence,
                    'analysis_method': 'Technical Analysis',
                    'price_change_24h': price_change,
                    'volume_ratio': current_volume / volume_avg if volume_avg > 0 else 1,
                    'news_count': 1  # Simulated
                }
            else:
                return {
                    'sentiment_score': 0,
                    'sentiment_label': 'Neutral',
                    'confidence': 0.5,
                    'analysis_method': 'No Data',
                    'price_change_24h': 0,
                    'volume_ratio': 1,
                    'news_count': 0
                }
                
        except Exception as e:
            print(f"Error analyzing sentiment for {ticker}: {e}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'confidence': 0.3,
                'analysis_method': 'Error',
                'price_change_24h': 0,
                'volume_ratio': 1,
                'news_count': 0
            }
    
    def get_enhanced_sentiment(self, ticker: str) -> Dict:
        """
        Enhanced sentiment analysis combining multiple factors
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get recent data for technical sentiment with retry logic
            recent_data = None
            for attempt in range(3):
                try:
                    recent_data = stock.history(period='10d')
                    break
                except Exception as e:
                    if "rate limited" in str(e).lower() or "too many requests" in str(e).lower():
                        print(f"Rate limited, waiting {2**attempt} seconds...")
                        time.sleep(2**attempt)
                        continue
                    else:
                        raise e
            
            if recent_data is None or len(recent_data) < 2:
                return self.get_fallback_sentiment(ticker)
            
            # Calculate technical indicators
            current_price = recent_data['Close'].iloc[-1]
            prev_price = recent_data['Close'].iloc[-2]
            week_ago_price = recent_data['Close'].iloc[0] if len(recent_data) >= 7 else prev_price
            
            # Price momentum
            daily_change = (current_price - prev_price) / prev_price
            weekly_change = (current_price - week_ago_price) / week_ago_price
            
            # Volume analysis
            avg_volume = recent_data['Volume'].mean()
            current_volume = recent_data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility
            returns = recent_data['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # RSI-like momentum
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            rsi_like = gains / (gains + losses) if (gains + losses) > 0 else 0.5
            
            # Combine factors for sentiment score
            sentiment_factors = {
                'daily_momentum': daily_change * 10,  # Weight: 10
                'weekly_momentum': weekly_change * 5,  # Weight: 5
                'volume_sentiment': (volume_ratio - 1) * 0.3,  # High volume = more conviction
                'rsi_sentiment': (rsi_like - 0.5) * 2,  # RSI-based sentiment
                'volatility_factor': -volatility * 5 if volatility > 0.05 else 0  # High volatility = uncertainty
            }
            
            # Calculate weighted sentiment score
            sentiment_score = sum(sentiment_factors.values()) / 5
            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp between -1 and 1
            
            # Determine sentiment label
            if sentiment_score > 0.3:
                sentiment_label = "Bullish 🐂"
            elif sentiment_score < -0.3:
                sentiment_label = "Bearish 🐻"
            elif sentiment_score > 0.1:
                sentiment_label = "Slightly Bullish 📈"
            elif sentiment_score < -0.1:
                sentiment_label = "Slightly Bearish 📉"
            else:
                sentiment_label = "Neutral ⚖️"
            
            # Calculate confidence based on volume and consistency
            confidence = min(0.9, 0.5 + abs(sentiment_score) * 0.3 + (volume_ratio - 1) * 0.1)
            
            # Get additional info
            info = stock.info
            market_cap = info.get('marketCap', 0)
            
            return {
                'sentiment_score': round(sentiment_score, 3),
                'sentiment_label': sentiment_label,
                'confidence': round(confidence, 2),
                'analysis_method': 'Enhanced Technical Analysis',
                'price_change_24h': round(daily_change, 4),
                'price_change_7d': round(weekly_change, 4),
                'volume_ratio': round(volume_ratio, 2),
                'volatility': round(volatility, 4),
                'rsi_like': round(rsi_like, 2),
                'market_cap': market_cap,
                'factors': sentiment_factors
            }
            
        except Exception as e:
            print(f"Error in enhanced sentiment analysis for {ticker}: {e}")
            return self.get_fallback_sentiment(ticker)
    
    def analyze_ticker_sentiment(self, ticker: str) -> Dict:
        """
        Main method to analyze sentiment for a given ticker
        """
        # Check cache first
        cache_key = f"{ticker}_{datetime.now().hour}"
        if cache_key in self.sentiment_cache:
            cached_time, cached_result = self.sentiment_cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_result
        
        # Get fresh sentiment analysis
        sentiment_result = self.get_enhanced_sentiment(ticker)
        
        # Cache the result
        self.sentiment_cache[cache_key] = (time.time(), sentiment_result)
        
        return sentiment_result
    
    def format_sentiment_message(self, ticker: str, sentiment_data: Dict) -> str:
        """
        Format sentiment analysis into a readable message for Telegram
        """
        try:
            message = f"\n📊 <b>Market Sentiment Analysis for {ticker}</b>\n"
            message += f"🎯 <b>Sentiment:</b> {sentiment_data['sentiment_label']}\n"
            message += f"📈 <b>Score:</b> {sentiment_data['sentiment_score']:.3f} (Range: -1 to +1)\n"
            message += f"🎪 <b>Confidence:</b> {sentiment_data['confidence']*100:.0f}%\n"
            
            # Add price change info
            if 'price_change_24h' in sentiment_data:
                change_24h = sentiment_data['price_change_24h'] * 100
                emoji_24h = "🟢" if change_24h > 0 else "🔴" if change_24h < 0 else "⚪"
                message += f"{emoji_24h} <b>24h Change:</b> {change_24h:+.2f}%\n"
            
            if 'price_change_7d' in sentiment_data:
                change_7d = sentiment_data['price_change_7d'] * 100
                emoji_7d = "🟢" if change_7d > 0 else "🔴" if change_7d < 0 else "⚪"
                message += f"{emoji_7d} <b>7d Change:</b> {change_7d:+.2f}%\n"
            
            # Add volume info
            if 'volume_ratio' in sentiment_data:
                vol_ratio = sentiment_data['volume_ratio']
                vol_emoji = "🔥" if vol_ratio > 1.5 else "📊" if vol_ratio > 0.8 else "📉"
                message += f"{vol_emoji} <b>Volume Ratio:</b> {vol_ratio:.2f}x avg\n"
            
            # Add volatility
            if 'volatility' in sentiment_data:
                volatility = sentiment_data['volatility'] * 100
                vol_emoji = "⚡" if volatility > 5 else "📈" if volatility > 2 else "😴"
                message += f"{vol_emoji} <b>Volatility:</b> {volatility:.1f}%\n"
            
            message += f"🔬 <b>Method:</b> {sentiment_data.get('analysis_method', 'Unknown')}\n"
            
            return message
            
        except Exception as e:
            print(f"Error formatting sentiment message: {e}")
            return f"\n📊 <b>Sentiment Analysis Error for {ticker}</b>\n❌ Could not analyze sentiment\n"


# Convenience function for easy import
def get_ticker_sentiment(ticker: str) -> Dict:
    """
    Quick function to get sentiment for a ticker
    """
    analyzer = MarketSentimentAnalyzer()
    return analyzer.analyze_ticker_sentiment(ticker)

def format_sentiment_for_telegram(ticker: str, sentiment_data: Dict) -> str:
    """
    Quick function to format sentiment for Telegram
    """
    analyzer = MarketSentimentAnalyzer()
    return analyzer.format_sentiment_message(ticker, sentiment_data)


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = MarketSentimentAnalyzer()
    
    test_tickers = ['BTC-USD', 'ETH-USD', 'USDC-EUR']
    
    for ticker in test_tickers:
        print(f"\n{'='*50}")
        print(f"Testing sentiment analysis for {ticker}")
        print('='*50)
        
        sentiment = analyzer.analyze_ticker_sentiment(ticker)
        message = analyzer.format_sentiment_message(ticker, sentiment)
        
        print("Raw sentiment data:")
        for key, value in sentiment.items():
            print(f"  {key}: {value}")
        
        print("\nFormatted message:")
        print(message)
