# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 08:43:40 2025

@author: hp
"""

#Fourier smoothing function
def fourier_smooth(doy, y, T=365, K=8):
    # Clean and sort
    df = pd.DataFrame({"t": doy, "y": y}).dropna().sort_values("t")
    t = df["t"].values
    yv = df["y"].values

    # Get Fourier coefficients from scrfft
    f, a, b = scrfft(t, yv)

    # Evaluate truncated Fourier series at all days 1..T
    tt = np.arange(1, T+1)
    y_hat = np.zeros_like(tt, dtype=float)
    # constant term
    y_hat += a[0]
    # add harmonics up to K
    for k in range(1, K+1):
        y_hat += a[k]*np.cos(2*np.pi*f[k]*tt) + b[k]*np.sin(2*np.pi*f[k]*tt)
    return tt, y_hat