# LSTM Forecast - System Architecture

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
5. [Model Architecture](#model-architecture)
6. [Data Flow Pipeline](#data-flow-pipeline)
7. [External Integrations](#external-integrations)
8. [Configuration Management](#configuration-management)
9. [Performance Metrics](#performance-metrics)
10. [Deployment & Infrastructure](#deployment--infrastructure)

---

## Project Overview

**LSTM Forecast** is a sophisticated stock price prediction system that combines deep learning with market sentiment analysis to generate 10-day price forecasts for cryptocurrencies, indices, and forex pairs.

### Key Features
- **Hybrid Neural Architecture**: CNN-LSTM-Transformer model with ~1.1M parameters
- **Multi-Source Sentiment Analysis**: Technical indicators, market breadth, volatility metrics
- **Automated Hyperparameter Tuning**: Parallel search with performance tracking
- **Telegram Integration**: Automated forecast delivery with charts
- **Model Persistence**: Performance-based model caching (MAE ≤ 10%)
- **Robust Data Collection**: Exponential backoff retry with rate limiting

### Supported Assets
```python
'USDC-EUR'   # USD Coin / Euro
'MXN=X'      # USD / Mexican Peso
'^MXX'       # IPC Mexico Index
'BTC-USD'    # Bitcoin
'ETH-USD'    # Ethereum
'PAXG-USD'   # PAX Gold
'^IXIC'      # NASDAQ Composite
'^SP500-45'  # S&P 500 Information Technology
```

### Technology Stack
- **Deep Learning**: TensorFlow 2.10+, Keras 2.10+
- **Data Processing**: NumPy 1.x, pandas 1.4+, scikit-learn 1.1+
- **Data Collection**: yfinance 0.2.66
- **Messaging**: python-telegram-bot 20.0+
- **Sentiment Analysis**: TextBlob, requests
- **Visualization**: Matplotlib 3.5+

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         LSTM FORECAST SYSTEM                     │
└─────────────────────────────────────────────────────────────────┘

┌───────────────┐      ┌──────────────────┐      ┌───────────────┐
│  Yahoo Finance│─────▶│  Data Collection │─────▶│ Preprocessing │
│   API (yfinance)      │   (main.py)      │      │  (MinMaxScaler)
└───────────────┘      └──────────────────┘      └───────┬───────┘
                                                          │
                       ┌──────────────────┐              │
                       │ Config Manager   │◀─────────────┤
                       │  (Hyperparams)   │              │
                       └─────────┬────────┘              │
                                 │                       │
                                 ▼                       ▼
┌───────────────┐      ┌──────────────────┐      ┌───────────────┐
│  Sentiment    │─────▶│  Hybrid Model    │◀─────│  60-Day Window│
│  Analyzer     │      │  (CNN-LSTM-Trans)│      │   Dataset     │
└───────────────┘      └─────────┬────────┘      └───────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  10-Day Forecast │
                       │   Predictions    │
                       └─────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌──────────────────┐      ┌──────────────────┐
          │  Visualization   │      │  Model Storage   │
          │  (matplotlib)    │      │  (.h5, .pkl)     │
          └─────────┬────────┘      └──────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Telegram Bot     │
          │  Notifications   │
          └──────────────────┘
```

### Component Interaction Flow

```
[User/Jenkins] ─────▶ [main.py]
                         │
                         ├─▶ [download_ticker_data_with_retry()]
                         │       └─▶ Yahoo Finance API
                         │
                         ├─▶ [config_manager.load_optimized_config()]
                         │       └─▶ tuning_results/*.json
                         │
                         ├─▶ [load_model_and_scaler()]
                         │       └─▶ models/{ticker}_*.{h5,pkl}
                         │
                         ├─▶ [create_dataset()] ─▶ [split_data()]
                         │
                         ├─▶ [create_hybrid_cnn_lstm_transformer_model()]
                         │       └─▶ Train or use cached model
                         │
                         ├─▶ [sentiment_analyzer.get_ticker_sentiment()]
                         │       └─▶ Technical indicator calculation
                         │
                         ├─▶ [Iterative 10-day prediction loop]
                         │
                         ├─▶ [matplotlib visualization]
                         │       └─▶ Save PNG charts
                         │
                         └─▶ [telegram_sender.send_telegram()]
                                 └─▶ Telegram Bot API
```

---

## Project Structure

### Directory Layout

```
LSTM_Forecast/
├── main.py                      # Core forecasting engine (803 lines)
├── main_with_sentiment.py       # Alternative with enhanced sentiment (552 lines)
├── sentiment_analyzer.py        # Market sentiment analysis (361 lines)
├── config_manager.py            # Configuration management (205 lines)
├── telegram_sender.py           # Telegram integration (51 lines)
├── fine_tuning_script.py        # Hyperparameter optimization (551 lines)
├── run_tuning.py                # Tuning CLI runner (78 lines)
├── sentiment_demo.py            # Sentiment demo utility (260 lines)
│
├── test_main.py                 # Unit tests (306 lines)
├── test_architectures.py        # Architecture tests (153 lines)
├── test_single_ticker.py        # Single ticker tests (142 lines)
├── test_main_script.py          # Integration tests (104 lines)
├── test_file_paths.py           # File handling tests (82 lines)
│
├── config.ini                   # Telegram bot credentials
├── requirements.txt             # Production dependencies
├── requirements_tuning.txt      # Tuning dependencies
├── SETUP_GUIDE.md               # Virtual environment setup
├── README.md                    # Main documentation
├── FINE_TUNING_README.md        # Tuning guide
├── LOG_README.md                # Logging documentation
│
├── setup_venv.sh                # Virtual environment setup script
├── run_with_venv.sh             # Jenkins execution script
├── update_dependencies.sh       # Dependency update script
│
├── models/                      # Trained models (~158 MB)
│   ├── {ticker}_model.h5        # Keras model weights (11-12 MB each)
│   ├── {ticker}_scaler.pkl      # MinMaxScaler state
│   └── {ticker}_mae_info.pkl    # Performance metadata
│
├── tuning_results/              # Hyperparameter optimization results
│   ├── best_configs_{ticker}.json
│   ├── tuning_results_{ticker}_{timestamp}.json
│   ├── optimized_config_{ticker}.json
│   └── tuning_report_{ticker}_{timestamp}.txt
│
├── tuned_models/                # Models from tuning runs
├── venv/                        # Python virtual environment
└── *_predictions.png            # Generated forecast charts
```

### File Statistics
- **Total Python Files**: 13 files
- **Total Lines of Code**: 3,648 lines
- **Test Coverage**: 787 lines across 5 test files
- **Model Storage**: 23 saved models (~158 MB)

---

## Core Components

### 1. Main Forecasting Engine (`main.py`)

**Primary Responsibilities:**
- Data collection from Yahoo Finance
- Model training and caching
- Iterative 10-day prediction generation
- Visualization and reporting
- Telegram notification orchestration

**Key Functions:**

#### `download_ticker_data_with_retry(ticker, period='max', max_retries=5)`
*Lines 46-123*

Robust data fetching with exponential backoff and rate limiting.

```python
# Features:
- Exponential backoff: delay * (2 ** attempt) * random(0.5, 1.5)
- Inter-ticker delays: random 10-20 seconds
- Validation: Minimum 61 days required
- Error handling: Graceful degradation on failure
```

#### `create_hybrid_cnn_lstm_transformer_model(input_shape, architecture_type='hybrid')`
*Lines 163-277*

Creates the neural network architecture based on configuration.

```python
# Architectures available:
- 'hybrid': CNN-LSTM-Transformer (default, best performance)
- 'cnn_lstm': CNN-LSTM hybrid (faster training)
- 'transformer_only': Pure Transformer (experimental)
- 'original_lstm': Bidirectional LSTM stack (legacy)
```

#### `process_ticker_data(ticker, data, display_ticker, model=None, scaler=None)`
*Lines 319-574*

Complete training and prediction pipeline.

```python
# Pipeline stages:
1. Data normalization (MinMaxScaler)
2. Dataset creation (60-day windows)
3. Train/test split (80/20)
4. Model training or loading
5. Evaluation (MAE, RMSE, MAPE, SMAPE, MASE)
6. 10-day iterative forecasting
7. Visualization generation
8. Model persistence (if MAE ≤ 10%)
```

#### `create_dataset(dataset, days_range=60)`
*Lines 577-583*

Sliding window dataset creation.

```python
# Input: [1000, 1] price array
# Output:
#   X: [940, 60, 1] - Input sequences
#   y: [940, 1] - Target next-day prices
```

#### `evalModel(model, X, y, y_train)`
*Lines 667-705*

Multi-metric evaluation system.

```python
# Metrics calculated:
- MAE: Mean Absolute Error (primary threshold: ≤10%)
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- SMAPE: Symmetric MAPE
- MASE: Mean Absolute Scaled Error (vs naive forecast)
```

### 2. Sentiment Analysis Module (`sentiment_analyzer.py`)

**Class: `MarketSentimentAnalyzer`**

**Primary Method: `get_enhanced_sentiment(ticker, days=10)`**
*Lines 168-263*

Technical analysis-based sentiment calculation.

```python
# Sentiment factors (weighted):
1. Daily momentum (10x): (close - open) / open
2. Weekly momentum (5x): (current - week_ago) / week_ago
3. Volume sentiment (0.3x): (vol - avg_vol) / avg_vol
4. RSI-like (2x): Recent price momentum
5. Volatility (-5x): Negative factor for high volatility

# Output:
{
    'sentiment_score': float (-1 to +1),
    'sentiment_label': 'Bullish' | 'Bearish' | 'Neutral',
    'confidence': float (0 to 1),
    'price_change_24h': float,
    'volume_ratio': float,
    'volatility': float,
    'rsi_like': float
}
```

**Fallback Method: `get_fallback_sentiment(ticker)`**
*Lines 41-95*

Heuristic-based sentiment when API fails.

```python
# Fallback logic:
- Crypto tickers (BTC, ETH): Default 0.2 (mild bullish)
- Stablecoins (USDC): Neutral 0.0
- Gold (PAXG): Defensive 0.1
- Indices: Market-based heuristics
```

### 3. Configuration Manager (`config_manager.py`)

**Class: `ConfigManager`**

Manages per-ticker optimized hyperparameters.

**Key Methods:**

```python
load_best_configs() -> List[OptimizedConfig]
    # Loads all optimized configs from tuning_results/

get_config_for_ticker(ticker) -> Dict
    # Returns best config for specific ticker

save_optimized_config(ticker, config, mae)
    # Persists new optimized configuration

auto_apply_best_config(ticker) -> bool
    # Updates main.py ARCHITECTURE_CONFIG dynamically
```

**Dataclass: `OptimizedConfig`**

```python
@dataclass
class OptimizedConfig:
    ticker: str
    mae: float
    config: Dict
    timestamp: str
    training_time: float
```

### 4. Hyperparameter Tuning System (`fine_tuning_script.py`)

**Class: `HyperparameterTuner`**

Automated parallel hyperparameter search.

**Search Space:**
```python
{
    'cnn_filters': [[32,64], [64,128], [128,256]],
    'cnn_kernels': [[3,5], [5,7], [3,7]],
    'lstm_units': [[32,64,32], [50,100,50], [64,128,64]],
    'transformer_heads': [4, 8, 12],
    'transformer_key_dim': [32, 64, 128],
    'dense_units': [[64,32], [128,64], [256,128]],
    'dropout_rate': [0.1, 0.2, 0.3],
    'batch_size': [8, 16, 32],
    'learning_rate': [0.0001, 0.001, 0.01],
    'weight_decay': [0.001, 0.01, 0.1]
}
```

**Methods:**

```python
run_hyperparameter_search(ticker, num_configs=50)
    # Runs parallel config evaluation
    # Returns: Top 10 configs sorted by MAE

generate_configurations(num_configs)
    # Random sampling from search space
    # Ensures diversity

train_and_evaluate_config(ticker, config)
    # Single config training pipeline
    # Returns: MAE, training_time, test_predictions

apply_best_config_to_main(ticker)
    # Updates main.py with best config
    # Creates backup before modification
```

### 5. Telegram Integration (`telegram_sender.py`)

**Async Functions:**

```python
async send_telegram(message: str)
    # Sends HTML-formatted text message
    # Uses BOT_TOKEN and CHAT_ID from config.ini

async send_image_to_telegram(image_path: str, caption: str = "")
    # Uploads PNG charts
    # Supports multiple images per ticker
```

**Message Format:**
```
🔮 HYBRID Forecast Report for {ticker}

📈 Next 10 Days Predictions:
[date]: $price, [date]: $price, ...

📊 Model Performance Metrics:
  • MAE: X.XX%
  • MAPE: X.XX%
  • MASE: X.XX
  • SMAPE: X.XX%
  • RMSE: X.XX%

📊 Market Sentiment Analysis:
  • Sentiment: Bullish/Bearish/Neutral
  • Score: X.XXX
  • Confidence: XX%

📸 Charts attached below ⬇️
```

---

## Model Architecture

### Hybrid CNN-LSTM-Transformer Architecture

**Input Shape**: `(batch_size, 60, 1)` - 60 days of normalized prices

#### Architecture Diagram

```
Input (60, 1)
      │
      ├──────────────────┬──────────────────┐
      │                  │                  │
      ▼                  ▼                  │
┌─────────────┐   ┌─────────────┐         │
│ CNN Branch  │   │ LSTM Branch │         │
│             │   │             │         │
│ Conv1D(64)  │   │ BiLSTM(50)  │         │
│ BN          │   │ BiLSTM(100) │         │
│ Conv1D(128) │   │ BiLSTM(50)  │         │
│ BN          │   │ Dropout(0.2)│         │
│ Dropout(0.2)│   │             │         │
└──────┬──────┘   └──────┬──────┘         │
       │                 │                 │
       └────────┬────────┘                 │
                ▼                          │
         ┌─────────────┐                   │
         │ Concatenate │◀──────────────────┘
         └──────┬──────┘
                ▼
      ┌─────────────────┐
      │ Transformer     │
      │                 │
      │ MultiHeadAtt(8) │
      │ LayerNorm + Res │
      │ FFN(256→dim)    │
      │ LayerNorm + Res │
      └────────┬────────┘
               ▼
      ┌─────────────────┐
      │ GlobalMaxPool1D │
      └────────┬────────┘
               ▼
      ┌─────────────────┐
      │ Dense(128,relu) │
      │ Dropout(0.3)    │
      │ Dense(64,relu)  │
      │ Dense(1)        │
      └────────┬────────┘
               ▼
        Output (1) - Predicted Price
```

### Detailed Layer Configuration

#### 1. CNN Branch (Local Pattern Extraction)
```python
Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
BatchNormalization()
Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')
BatchNormalization()
Dropout(rate=0.2)

# Output shape: (batch, 60, 128)
# Purpose: Extract local price patterns (3-day and 5-day windows)
```

#### 2. LSTM Branch (Temporal Dependencies)
```python
Bidirectional(LSTM(50, return_sequences=True))   # 100 units total
Bidirectional(LSTM(100, return_sequences=True))  # 200 units total
Bidirectional(LSTM(50, return_sequences=True))   # 100 units total
Dropout(rate=0.2)

# Output shape: (batch, 60, 100)
# Purpose: Capture long-term temporal dependencies
```

#### 3. Feature Fusion
```python
Concatenate([cnn_output, lstm_output], axis=-1)

# Combined shape: (batch, 60, 228)
# Purpose: Merge local patterns with temporal features
```

#### 4. Transformer Branch (Global Attention)
```python
# Multi-Head Attention
MultiHeadAttention(
    num_heads=8,
    key_dim=64,
    dropout=0.1
)(combined_features, combined_features)

# Add & Norm (Residual Connection 1)
LayerNormalization()(attention_output + combined_features)

# Feed-Forward Network
Dense(256, activation='relu')
Dropout(0.1)
Dense(combined_features_dim, activation='relu')

# Add & Norm (Residual Connection 2)
LayerNormalization()(ffn_output + norm1_output)

# Output shape: (batch, 60, 228)
# Purpose: Global attention across all timesteps
```

#### 5. Prediction Head
```python
GlobalMaxPooling1D()           # (batch, 228)
Dense(128, activation='relu')  # (batch, 128)
Dropout(0.3)
Dense(64, activation='relu')   # (batch, 64)
Dense(1)                       # (batch, 1) - Final prediction

# Purpose: Compress to single price prediction
```

### Model Configuration

```python
# Compilation
optimizer = AdamW(
    learning_rate=0.001,
    weight_decay=0.01
)
loss = 'mean_squared_error'
metrics = ['mae', 'mse']

# Training
batch_size = 16
epochs = 150  # with EarlyStopping(patience=15)
validation_split = 0.2

# Callbacks
EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)
```

### Model Statistics

- **Total Parameters**: ~1.1 million (trainable)
- **Model Size**: 11-12 MB per saved model
- **Training Time**: 5-15 minutes per ticker (GPU)
- **Inference Time**: <1 second for 10-day forecast
- **Memory Usage**: ~2-3 GB during training

### Alternative Architectures

#### CNN-LSTM (Without Transformer)
```python
architecture_type='cnn_lstm'
# Faster training, slightly lower accuracy
# Use when: Speed > Accuracy
```

#### Transformer Only
```python
architecture_type='transformer_only'
# Pure attention mechanism
# Use when: Experimenting with attention patterns
```

#### Original LSTM
```python
architecture_type='original_lstm'
# Legacy bidirectional LSTM stack
# Use when: Baseline comparison needed
```

---

## Data Flow Pipeline

### Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA COLLECTION                                        │
└─────────────────────────────────────────────────────────────────┘
download_ticker_data_with_retry(ticker, period='max')
├─ yfinance.download() with retry logic
├─ Exponential backoff: 10s × 2^attempt × random(0.5, 1.5)
├─ Inter-ticker delay: random(10, 20) seconds
├─ Validation: ≥61 days required
└─ Output: DataFrame[Date, Open, High, Low, Close, Volume]

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: PREPROCESSING                                          │
└─────────────────────────────────────────────────────────────────┘
process_ticker_data()
├─ Extract 'Close' prices → (N, 1) array
├─ MinMaxScaler(feature_range=(0, 1))
│  └─ Formula: (price - min) / (max - min)
├─ create_dataset(scaled_data, days_range=60)
│  └─ X: [price[i:i+60] for i in range(len-60)]
│  └─ y: [price[i+60] for i in range(len-60)]
└─ split_data(X, y, train_size=0.8)
   └─ X_train, y_train, X_test, y_test

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: MODEL TRAINING/LOADING                                 │
└─────────────────────────────────────────────────────────────────┘
Check existing model:
├─ load_model_and_scaler(ticker)
│  ├─ Load: {ticker}_model.h5, {ticker}_scaler.pkl, {ticker}_mae_info.pkl
│  ├─ Validate: MAE ≤ 10%
│  └─ Return model if valid
└─ If no model or invalid MAE:
   ├─ load_optimized_config_for_ticker(ticker)
   ├─ create_hybrid_cnn_lstm_transformer_model()
   ├─ model.compile(optimizer=AdamW, loss='mse')
   ├─ model.fit(X_train, y_train, epochs=150, callbacks=[...])
   ├─ evalModel() → Calculate MAE, RMSE, MAPE, SMAPE, MASE
   └─ save_model_and_scaler() if MAE ≤ 10%

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: SENTIMENT ANALYSIS                                     │
└─────────────────────────────────────────────────────────────────┘
get_ticker_sentiment(ticker)
├─ Download last 10 days of data
├─ Calculate technical indicators:
│  ├─ Daily momentum: (close - open) / open × 10
│  ├─ Weekly momentum: (current - week_ago) / week_ago × 5
│  ├─ Volume ratio: (volume - avg_volume) / avg_volume × 0.3
│  ├─ RSI-like: Recent price momentum × 2
│  └─ Volatility: -std(returns) × 5
├─ Weighted sentiment score: sum(factors) → (-1 to +1)
├─ Determine label:
│  ├─ > 0.2: Bullish
│  ├─ < -0.2: Bearish
│  └─ else: Neutral
└─ Calculate confidence: min(abs(score), 1.0)

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: PREDICTION GENERATION                                  │
└─────────────────────────────────────────────────────────────────┘
Iterative 10-day forecasting:
├─ current_batch = scaled_data[-60:].reshape(1, 60, 1)
├─ predictions = []
└─ For day in range(10):
   ├─ next_pred = model.predict(current_batch)
   ├─ predictions.append(scaler.inverse_transform(next_pred))
   ├─ current_batch = np.append(current_batch[:, 1:, :], next_pred)
   └─ current_batch = current_batch.reshape(1, 60, 1)

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6: VISUALIZATION & OUTPUT                                 │
└─────────────────────────────────────────────────────────────────┘
matplotlib visualization:
├─ Create prediction dates (last_date + 1 to last_date + 10)
├─ Plot 1: predictions_only
│  └─ Save: {ticker}_predictions.png
├─ Plot 2: historical (60 days) + predictions (10 days)
│  └─ Save: full_{ticker}_predictions.png
└─ Telegram notification:
   ├─ send_telegram(formatted_message)
   │  └─ Includes: predictions, metrics, sentiment
   └─ send_image_to_telegram(chart_paths)
      └─ Uploads both PNG files
```

### Data Shapes Throughout Pipeline

```python
# Raw data
yfinance → DataFrame(N rows × 6 columns)

# After preprocessing
Close prices → array(N, 1)
Scaled → array(N, 1) in range [0, 1]

# Dataset creation
X → array(N-60, 60, 1)   # Input sequences
y → array(N-60, 1)       # Target prices

# Train/test split (80/20)
X_train → array(0.8*(N-60), 60, 1)
y_train → array(0.8*(N-60), 1)
X_test → array(0.2*(N-60), 60, 1)
y_test → array(0.2*(N-60), 1)

# Prediction loop
current_batch → array(1, 60, 1)  # Single 60-day window
prediction → array(1, 1)          # Single price prediction

# Final output
predictions → array(10, 1)        # 10-day forecast
```

---

## External Integrations

### 1. Yahoo Finance API (yfinance 0.2.66)

**Purpose**: Historical price data collection

**Usage Pattern**:
```python
data = yf.download(
    ticker,
    period='max',           # Maximum available history
    interval='1d',          # Daily bars
    auto_adjust=True,       # Adjust for splits/dividends
    prepost=False,          # No pre/post market
    threads=False           # Single-threaded for stability
)
```

**Rate Limiting Strategy**:
```python
# Retry logic
max_retries = 5
base_delay = 10  # seconds
delay = base_delay * (2 ** attempt) * random.uniform(0.5, 1.5)

# Inter-ticker delays
time.sleep(random.uniform(10, 20))  # Between ticker downloads
```

**Error Handling**:
- Network errors: Retry with exponential backoff
- Empty data: Skip ticker with warning
- Invalid ticker: Log error and continue
- Rate limit (429): Exponential backoff up to 320s

### 2. Telegram Bot API (python-telegram-bot 20.0+)

**Configuration** (`config.ini`):
```ini
[token]
BOT_TOKEN = your_bot_token_from_BotFather

[chat]
CHAT_ID = your_chat_id_from_userinfobot
```

**Message Types**:

1. **Text Messages** (HTML formatted):
```python
await bot.send_message(
    chat_id=CHAT_ID,
    text=html_formatted_message,
    parse_mode='HTML'
)
```

2. **Image Messages**:
```python
await bot.send_photo(
    chat_id=CHAT_ID,
    photo=open(image_path, 'rb'),
    caption=caption
)
```

**Message Format Example**:
```html
🔮 <b>HYBRID</b> Forecast Report for <b>BTC-USD</b>

📈 <b>Next 10 Days Predictions:</b>
2024-11-20: $95,234.56
2024-11-21: $96,123.45
...

📊 <b>Model Performance Metrics:</b>
  • MAE: 2.34%
  • MAPE: 2.12%
  • MASE: 0.89
  • SMAPE: 2.08%
  • RMSE: 3.21%

📊 <b>Market Sentiment Analysis:</b>
  • Sentiment: <b>Bullish</b>
  • Score: 0.456
  • Confidence: 45.6%

📸 Charts attached below ⬇️
```

### 3. Sentiment Data Sources

#### Primary: Technical Analysis
```python
# Data from yfinance (last 10 days)
- Price momentum (daily, weekly)
- Volume analysis (vs 10-day average)
- Volatility (std of returns)
- RSI-like momentum indicators
```

#### Alternative: Market Indicators (`main_with_sentiment.py`)
```python
# Fear & Greed Index
url = 'https://api.alternative.me/fng/'
response = requests.get(url)
fear_greed_value = response.json()['data'][0]['value']

# VIX Index (volatility)
vix_data = yf.download('^VIX', period='5d')

# Major Indices
sp500 = yf.download('^GSPC', period='5d')
dow = yf.download('^DJI', period='5d')
nasdaq = yf.download('^IXIC', period='5d')

# Crypto Sentiment (Bitcoin as leading indicator)
btc = yf.download('BTC-USD', period='5d')
```

**Sentiment Weighting**:
```python
weights = {
    'fear_greed': 0.30,    # Alternative.me index
    'vix': 0.25,           # CBOE volatility (inverse)
    'put_call': 0.20,      # Put/Call ratio proxy
    'breadth': 0.15,       # Market breadth
    'crypto': 0.10         # Bitcoin performance
}
```

### 4. Jenkins CI/CD Integration

**Job Configuration**:
```bash
#!/bin/bash
set -e

# Navigate to project
cd /Users/danielmenesesleon/PycharmProjects/LSTM_Forecast

# Run with isolated virtual environment
./run_with_venv.sh
```

**Execution Flow**:
```
Jenkins Job Trigger (schedule/manual)
       ↓
Execute Shell: run_with_venv.sh
       ↓
Activate venv: source venv/bin/activate
       ↓
Run: python main.py
       ↓
Process all 8 tickers sequentially
       ↓
Send Telegram notifications
       ↓
Deactivate venv
       ↓
Report build status (SUCCESS/FAILURE)
```

---

## Configuration Management

### 1. Global Configuration (`main.py`)

```python
ARCHITECTURE_CONFIG = {
    # Model architecture
    'type': 'hybrid',              # 'hybrid' | 'cnn_lstm' | 'transformer_only' | 'original_lstm'

    # CNN parameters
    'cnn_filters': [64, 128],      # Number of filters in Conv1D layers
    'cnn_kernels': [3, 5],         # Kernel sizes for local pattern detection

    # LSTM parameters
    'lstm_units': [50, 100, 50],   # Units in each Bidirectional LSTM layer

    # Transformer parameters
    'transformer_heads': 8,         # Number of attention heads
    'transformer_key_dim': 64,      # Dimension of attention keys

    # Dense layers
    'dense_units': [128, 64],       # Units in final dense layers

    # Regularization
    'dropout_rate': 0.2,            # Dropout rate throughout model

    # Training
    'batch_size': 16,               # Batch size for training
    'epochs': 150,                  # Maximum epochs (EarlyStopping may stop earlier)
    'learning_rate': 0.001,         # AdamW learning rate
    'weight_decay': 0.01            # AdamW weight decay for regularization
}
```

### 2. Ticker-Specific Configuration (`config_manager.py`)

**Configuration Loading**:
```python
# Load optimized config for specific ticker
config = config_manager.get_config_for_ticker('BTC-USD')

# Auto-apply to main.py
config_manager.auto_apply_best_config('BTC-USD')
```

**Storage Format** (`tuning_results/optimized_config_{ticker}.json`):
```json
{
  "ticker": "BTC-USD",
  "mae": 0.0234,
  "timestamp": "2024-11-17T14:30:00",
  "training_time": 642.5,
  "config": {
    "cnn_filters": [64, 128],
    "cnn_kernels": [3, 5],
    "lstm_units": [50, 100, 50],
    "transformer_heads": 8,
    "transformer_key_dim": 64,
    "dense_units": [128, 64],
    "dropout_rate": 0.2,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 0.01
  }
}
```

### 3. Telegram Credentials (`config.ini`)

```ini
[token]
# Get your bot token from @BotFather on Telegram
BOT_TOKEN = 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

[chat]
# Get your chat ID by messaging @userinfobot on Telegram
CHAT_ID = -1234567890
```

**Security**:
- ⚠️ `config.ini` is in `.gitignore` (contains secrets)
- Never commit credentials to version control
- Use environment variables for production

### 4. Model Persistence Configuration

**Save Conditions**:
```python
# Only save model if performance threshold met
if mae <= 0.10:  # 10% threshold
    save_model_and_scaler(model, scaler, ticker, mae)
```

**Files Saved** (atomic operation):
```python
models/{ticker}_model.h5        # Keras model weights
models/{ticker}_scaler.pkl      # MinMaxScaler fit parameters
models/{ticker}_mae_info.pkl    # Performance metadata
```

**Load Strategy**:
```python
# Load existing model
model, scaler, saved_mae = load_model_and_scaler(ticker)

# Validate performance
if saved_mae > 0.10:
    # Retrain if performance degraded
    model = None
```

---

## Performance Metrics

### Evaluation Metrics

#### 1. MAE (Mean Absolute Error) - Primary Metric
```python
mae = mean_absolute_error(y_true, y_pred)
mae_percentage = (mae / mean(y_true)) * 100

# Threshold: ≤ 10% for model saving
# Typical range: 2-10% for major assets
```

#### 2. RMSE (Root Mean Squared Error)
```python
rmse = sqrt(mean_squared_error(y_true, y_pred))
rmse_percentage = (rmse / mean(y_true)) * 100

# Penalizes larger errors more heavily
# Useful for detecting outlier predictions
```

#### 3. MAPE (Mean Absolute Percentage Error)
```python
mape = mean(abs((y_true - y_pred) / y_true)) * 100

# Percentage-based accuracy
# Easy to interpret (e.g., 5% average error)
```

#### 4. SMAPE (Symmetric MAPE)
```python
smape = mean(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred))) * 100

# Symmetric version of MAPE
# Handles direction changes better
```

#### 5. MASE (Mean Absolute Scaled Error)
```python
naive_forecast_error = mean(abs(y_train[1:] - y_train[:-1]))
mase = mae / naive_forecast_error

# Scaled against naive "next day = today" forecast
# < 1.0: Better than naive
# > 1.0: Worse than naive
```

### Typical Performance Benchmarks

| Asset Class | MAE | MAPE | MASE | Training Time |
|-------------|-----|------|------|---------------|
| **Bitcoin (BTC-USD)** | 2-5% | 2-4% | 0.5-0.8 | 10-15 min |
| **Ethereum (ETH-USD)** | 3-6% | 3-5% | 0.6-0.9 | 10-15 min |
| **NASDAQ (^IXIC)** | 2-4% | 2-3% | 0.4-0.7 | 8-12 min |
| **Forex (MXN=X)** | 1-3% | 1-2% | 0.3-0.6 | 5-10 min |
| **Gold (PAXG-USD)** | 2-4% | 2-3% | 0.5-0.8 | 8-12 min |

### Model Performance Statistics

```python
# From actual testing (17 Nov 2024)
Saved Models: 23 files
Total Model Storage: ~158 MB
Average Model Size: ~11.5 MB per ticker

Training Performance:
- Average MAE: 4.2%
- Best MAE: 1.8% (forex pairs)
- Worst MAE: 8.5% (volatile crypto)

Inference Performance:
- Single prediction: <10ms
- 10-day forecast: <100ms
- Full ticker processing: 5-20 minutes (including training)
```

### System Resource Usage

```python
# Training (per ticker)
CPU Usage: 60-90% (multi-core)
Memory: 2-3 GB
GPU Memory: 4-6 GB (if available)
Disk I/O: Moderate (data loading)

# Inference
CPU Usage: 10-20%
Memory: 500 MB - 1 GB
GPU Memory: 1-2 GB (if available)
Disk I/O: Low

# Full Pipeline (8 tickers)
Total Runtime: 40-120 minutes
Peak Memory: ~3 GB
Network Usage: ~50-100 MB (data download)
```

---

## Deployment & Infrastructure

### Virtual Environment Setup

**Creation** (`setup_venv.sh`):
```bash
#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Execution** (`run_with_venv.sh`):
```bash
#!/bin/bash
source venv/bin/activate
python main.py
deactivate
```

**Dependency Update** (`update_dependencies.sh`):
```bash
#!/bin/bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --upgrade
deactivate
```

### Dependencies

**Core Dependencies** (`requirements.txt`):
```
# Deep Learning
tensorflow>=2.10.0
keras>=2.10.0

# Data Processing
numpy>=1.21.0,<2.0.0    # Pinned to avoid NumPy 2.x incompatibility
pandas>=1.4.0
scikit-learn>=1.1.0
numexpr>=2.8.0,<2.10.0  # NumPy 1.x compatible
bottleneck>=1.3.0,<2.0.0
h5py>=3.7.0,<4.0.0

# Data Collection
yfinance==0.2.66        # Specific version for rate limiting

# Visualization
matplotlib>=3.5.0

# Messaging
python-telegram-bot>=20.0

# Sentiment Analysis
textblob>=0.17.1
requests>=2.28.0

# Configuration
configparser

# Async
asyncio
```

**Why NumPy < 2.0?**
```
NumPy 2.x introduced breaking changes in binary API.
Libraries compiled with NumPy 1.x (pandas, tensorflow, h5py, numexpr, bottleneck)
cannot run with NumPy 2.x without recompilation.

Solution: Pin numpy<2.0.0 until all dependencies support NumPy 2.x
```

### File Storage Strategy

**Directory Permissions**:
```bash
models/           # 755 (rwxr-xr-x)
tuning_results/   # 755 (rwxr-xr-x)
config.ini        # 600 (rw-------)  # Sensitive credentials
*.sh              # 755 (rwxr-xr-x)  # Executable scripts
```

**Disk Space Requirements**:
```
Virtual Environment: ~500 MB - 1 GB
Models Storage: ~12 MB × number of tickers
Tuning Results: ~1-5 MB per ticker
Prediction Images: ~100-200 KB per ticker × 2 images
Total (8 tickers): ~2-3 GB
```

### Environment Variables (Optional)

```bash
# Alternative to config.ini for production
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Modify telegram_sender.py to read from env:
import os
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
```

### Jenkins CI/CD Configuration

**Build Triggers**:
- Scheduled (daily at 6 PM local time)
- Manual trigger
- Git push to main branch (optional)

**Build Steps**:
```bash
#!/bin/bash
set -e  # Exit on error

cd /Users/danielmenesesleon/PycharmProjects/LSTM_Forecast

# Pull latest code
git pull origin main

# Run with virtual environment
./run_with_venv.sh
```

**Post-Build Actions**:
- Email notification on failure
- Telegram notification (handled by script)
- Archive prediction images (optional)

**Build Environment**:
```
Platform: macOS (Darwin 25.2.0)
Python: 3.12
Shell: /bin/bash
Working Directory: ~/.jenkins/workspace/LSTM
```

### Logging Strategy

**Log Levels**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_forecast.log'),
        logging.StreamHandler()
    ]
)
```

**Log Categories**:
- Data download attempts and failures
- Model training progress
- Prediction generation
- Telegram notification status
- Error tracebacks

**Log Rotation** (optional):
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'lstm_forecast.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5
)
```

---

## Security Considerations

### Credential Management

**Sensitive Files** (in `.gitignore`):
```
config.ini          # Telegram bot credentials
*.secret
*.key
*.pem
.env
```

**Recommendations**:
1. Never commit `config.ini` to version control
2. Use environment variables for production
3. Rotate Telegram bot tokens periodically
4. Limit bot permissions to send-only

### API Rate Limiting

**Yahoo Finance**:
- 5 retries with exponential backoff
- 10-20 second delays between tickers
- Single-threaded downloads
- Respects HTTP 429 (Too Many Requests)

**Telegram Bot**:
- 30 messages/second limit (not reached)
- 20 MB photo upload limit
- Async send to avoid blocking

### Data Validation

**Input Validation**:
```python
# Minimum data requirement
if len(data) < 61:
    raise ValueError(f"Insufficient data: {len(data)} days (need 61+)")

# Data quality checks
if data.isnull().any():
    data = data.fillna(method='ffill')  # Forward fill missing values

# Price sanity checks
if (data['Close'] <= 0).any():
    raise ValueError("Invalid price data (negative or zero)")
```

**Model Validation**:
```python
# Performance threshold
if mae > 0.10:
    logger.warning(f"Model MAE {mae:.2%} exceeds 10% threshold - not saving")
    return None

# Shape validation
assert X_train.shape[1:] == (60, 1), "Invalid input shape"
assert model.output_shape == (None, 1), "Invalid output shape"
```

---

## Future Enhancements

### Planned Features

1. **Multi-Asset Portfolio Optimization**
   - Correlation analysis between tickers
   - Portfolio weight recommendations
   - Risk-adjusted return optimization

2. **Enhanced Sentiment Sources**
   - Twitter/X sentiment scraping
   - Reddit WallStreetBets analysis
   - News headline sentiment (NewsAPI)
   - Google Trends integration

3. **Real-Time Predictions**
   - WebSocket integration for live prices
   - Intraday predictions (hourly forecasts)
   - Alert system for significant price movements

4. **Advanced Model Architectures**
   - Ensemble models (voting/stacking)
   - Temporal Fusion Transformer (TFT)
   - GRU variants for faster training
   - Attention mechanism visualization

5. **Enhanced Monitoring**
   - Prometheus metrics export
   - Grafana dashboards
   - Model drift detection
   - Prediction confidence intervals

6. **API Service**
   - REST API for predictions
   - FastAPI or Flask deployment
   - Authentication and rate limiting
   - Swagger/OpenAPI documentation

### Technical Debt

1. **Testing**
   - Increase unit test coverage (currently ~20%)
   - Add integration tests for full pipeline
   - Mock external API calls in tests
   - Automated regression testing

2. **Code Quality**
   - Refactor main.py (803 lines → modular components)
   - Type hints throughout codebase
   - Docstring standardization (Google/NumPy style)
   - Linting (pylint, flake8)

3. **Performance**
   - GPU utilization optimization
   - Parallel ticker processing
   - Model quantization for faster inference
   - Caching sentiment analysis results

4. **Documentation**
   - API documentation (if REST API added)
   - Architecture diagrams (PlantUML/Mermaid)
   - Contribution guidelines
   - Deployment runbook

---

## Troubleshooting

### Common Issues

#### 1. NumPy Compatibility Error
```
ImportError: numpy.dtype size changed, may indicate binary incompatibility
```

**Solution**:
```bash
rm -rf venv/
./setup_venv.sh  # Reinstalls with numpy<2.0.0
```

#### 2. Yahoo Finance Rate Limiting
```
Error downloading ticker X: HTTP 429 Too Many Requests
```

**Solution**: Automatic retry with exponential backoff (built-in)

#### 3. Telegram Send Failure
```
telegram.error.NetworkError: Connection refused
```

**Solution**: Check internet connectivity and bot token validity

#### 4. Model Not Saving
```
Warning: Model MAE 12.34% exceeds 10% threshold - not saving
```

**Solution**: Run hyperparameter tuning to find better configuration

#### 5. Insufficient Data
```
ValueError: Insufficient data: 45 days (need 61+)
```

**Solution**: Use different ticker or adjust `days_range` parameter

---

## References

### Documentation
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [python-telegram-bot Documentation](https://python-telegram-bot.readthedocs.io/)

### Research Papers
- Transformer Architecture: "Attention Is All You Need" (Vaswani et al., 2017)
- LSTM Networks: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- CNN for Time Series: "Convolutional Neural Networks for Time Series Classification" (Fawaz et al., 2019)

### Related Projects
- [LSTM Stock Predictor](https://github.com/topics/lstm-stock-prediction)
- [TensorFlow Time Series](https://www.tensorflow.org/tutorials/structured_data/time_series)

---

**Document Version**: 1.0
**Last Updated**: 2024-11-17
**Maintainer**: LSTM Forecast Team
**License**: [Project License]
