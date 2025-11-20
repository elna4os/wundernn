Predict market next state:

- 32-dimensional anonymized time series features with low autocorrelation and SNR
- Each time series consists of 1000 timesteps
- Predictions should be provided for steps >= 101
- First 100 steps can be used to "warm up" the model
- Features like EMA and spectral entropy may be useful (according to Granger causality test), but adding them to the model can be tricky

---

**Install dependencies**:

```bash
poetry install --no-root
```

---
