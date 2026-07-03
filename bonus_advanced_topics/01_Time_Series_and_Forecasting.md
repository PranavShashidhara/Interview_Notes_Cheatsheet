# Time Series and Forecasting

## Overview

**Time Series**: sequence of observations ordered by time. Examples: stock prices, weather, website traffic.

**Forecasting**: predict future values based on past patterns.

**Challenges**: trend, seasonality, autocorrelation, non-stationarity, regime changes.

---

## Time Series Components

### Additive Model
Y(t) = Trend(t) + Seasonal(t) + Residual(t)

### Multiplicative Model
Y(t) = Trend(t) × Seasonal(t) × Residual(t)

**Trend**: long-term direction (increasing, decreasing, flat)
**Seasonality**: repeating patterns at fixed intervals (daily, weekly, yearly)
**Residual**: random noise + unexplained variation

---

## Classical Methods

### Moving Average (MA)
Average over last k periods: forecast(t) = mean(Y(t-k), ..., Y(t-1))

```python
import pandas as pd
df['MA_7'] = df['value'].rolling(window=7).mean()
```

**Pros**: simple, fast
**Cons**: lags behind trend; equal weight on all past observations

### Exponential Smoothing
Weighted average: more recent observations get higher weight.

```
forecast(t+1) = α·Y(t) + (1-α)·forecast(t)
```

where α ∈ [0,1] (smoothing parameter)

**Pros**: adaptive; computational efficient
**Cons**: assumes no trend or seasonality (Holt-Winters extends this)

### ARIMA (AutoRegressive Integrated Moving Average)

**AR (AutoRegressive)**: Y(t) = c + φ₁·Y(t-1) + ... + φₚ·Y(t-p) + ε(t)
- Current value depends on past values

**I (Integrated)**: differencing to make series stationary
- Y'(t) = Y(t) - Y(t-1) (first difference)

**MA (Moving Average)**: Y(t) = ε(t) + θ₁·ε(t-1) + ... + θq·ε(t-q)
- Current value depends on past forecast errors

**ARIMA(p,d,q)**:
- p: AR order (number of lagged values)
- d: differencing order (1-2 typical)
- q: MA order (number of lagged errors)

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df['value'], order=(1, 1, 1))
result = model.fit()
forecast = result.get_forecast(steps=30).predicted_mean
```

**Pros**: statistically grounded; widely used; handles many patterns
**Cons**: requires stationarity; hyperparameter tuning; poor for long-term forecasts

### Seasonal ARIMA (SARIMA)

ARIMA + seasonal component: SARIMA(p,d,q)(P,D,Q,s)
- (P,D,Q): seasonal AR, I, MA orders
- s: seasonal period (e.g., 12 for monthly data with yearly seasonality)

### Prophet (Facebook)

Additive model: Y(t) = Trend(t) + Seasonal(t) + Holiday(t) + Residual(t)

```python
from fbprophet import Prophet

model = Prophet()
model.fit(df)  # df columns: 'ds' (date), 'y' (value)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

**Pros**: handles missing data, outliers, holidays; robust; minimal tuning
**Cons**: less flexible; slower than ARIMA

---

## Deep Learning for Time Series

### RNN / LSTM
Maintain hidden state across time steps.

```python
model = Sequential([
    LSTM(64, activation='relu', input_shape=(lookback, n_features)),
    LSTM(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

**Input shape**: (batch, timesteps, features)
- lookback: how many past steps to use for prediction

**Pros**: learns long-range dependencies; flexible
**Cons**: slower to train; needs more data; harder to tune

### Attention & Transformers
Self-attention weights past timesteps for current prediction.

```python
# Transformer Encoder
encoder = TransformerEncoder(...)
output = encoder(X)  # (batch, seq_len, d_model)
forecast = Dense(1)(output[:, -1, :])  # last timestep to output
```

**Advantages**: parallelizable; captures long-range patterns better than RNN

### TCN (Temporal Convolutional Network)
Dilated convolutions over time; parallelizable like Transformers but simpler.

```python
model = Sequential([
    Conv1D(64, kernel_size=3, dilation_rate=1, padding='same'),
    Conv1D(64, kernel_size=3, dilation_rate=2, padding='same'),
    Conv1D(64, kernel_size=3, dilation_rate=4, padding='same'),
    GlobalAveragePooling1D(),
    Dense(1)
])
```

---

## Evaluation Metrics

### Point Forecasts
- **MAE (Mean Absolute Error)**: average |actual - forecast|; robust to outliers
- **RMSE (Root Mean Squared Error)**: sqrt(mean((actual - forecast)²)); penalizes large errors
- **MAPE (Mean Absolute Percentage Error)**: average |actual - forecast| / |actual|; scale-independent

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

### Directional Accuracy
Does forecast predict correct direction (up/down)?

```python
direction_correct = np.sign(y_true - y_true.shift(1)) == np.sign(y_pred - y_true.shift(1))
accuracy = direction_correct.mean()
```

### Diebold-Mariano Test
Statistical test if two forecasts differ significantly.

---

## Practical Considerations

### Stationarity (ADF Test)
Most classical methods assume stationary series (mean, variance, autocorr constant over time).

```python
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df['value'])
p_value = adf_result[1]
if p_value > 0.05:
    print("Series is non-stationary; apply differencing")
```

### Autocorrelation (ACF/PACF)
Helps choose ARIMA(p,d,q) orders.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['value'], lags=40)  # q order
plot_pacf(df['value'], lags=40)  # p order
```

### Train/Test Split
**Don't shuffle!** Use time ordering.

```python
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]
```

### Walk-Forward Validation
Simulate real deployment: train on past, test on next k periods, move forward.

```python
for t in range(len(train), len(df) - k):
    model.fit(df[:t])
    forecast = model.predict(df[t:t+k])
    evaluate(forecast, df[t:t+k])
```

---

## Interview Key Points

- **Stationarity: why matters?** ARIMA assumes stationary; non-stationary means mean/variance change over time; need differencing.
- **ARIMA vs Prophet?** ARIMA: statistically grounded, requires tuning. Prophet: simpler, handles holidays/events, more robust.
- **RNN vs CNN for time series?** RNN: sequential, maintains hidden state. CNN: parallel (fast), dilated convolutions capture multiple timescales. TCN combines both.
- **Why LSTM over RNN?** LSTM has gates (forget, input, output) to control information flow; mitigates vanishing gradient.
- **Trend vs seasonality?** Trend: long-term direction. Seasonality: repeating patterns. Both need handling for good forecasts.
- **Autocorrelation: how to use?** ACF shows lags correlated with current value (suggests AR order). PACF shows direct dependence (partial correlation).
- **Train/test split in time series?** No shuffling! Use temporal order; walk-forward validation mimics real deployment.
- **How to forecast far future (year ahead)?** Classical methods (ARIMA, Prophet) degrade. Scenarios/ensemble; focus on trend/seasonality; human judgment.
