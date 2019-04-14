import pandas as pd
import numpy as np
# import datetime
from pandas.compat import lmap
from statsmodels.tsa.arima_model import ARMA
from scipy.linalg import cholesky
import matplotlib.pyplot as plt


def autocorrelation_and_significance(series, ax=None, **kwds):
    """Autocorrelation and significant lags for time series."""
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(1, n), ylim=(-1.0, 1.0))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = np.arange(n) + 1
    y = lmap(r, x)
    z95 = 1.959963984540054
    z95l = -z95/np.sqrt(n)
    z95h = z95/np.sqrt(n)
    ax.axhline(y=z95h, linestyle=':', color='red', )
    ax.axhline(y=0.0, color='black')
    ax.axhline(y=z95l, linestyle=':', color='red')
    ax.set_xlabel("lag", fontsize=15)
    ax.set_ylabel("autocorrelation", fontsize=15)
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    df = pd.DataFrame({'autocorrelation': y,
                       'lag': x})
    lags = np.sort(np.append(df[df['autocorrelation'] > z95h]['lag'],
                             df[df['autocorrelation'] < z95l]['lag']))
    return df, lags, ax
