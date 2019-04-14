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


def optimal_params_ar_model(data, lags_to_test, cap=4, test_criteria='BIC', **kwds):
    """Optimal lags using Bayes or Akaike Information Criteria.

    Given a time series, and significatnlags returned by the
    autocorrelation_and_significance function, this function test n values (cap)
    to find out if auto regresive models of order > 1 are worth exploring.
    Test criteria can be Akaike ('AIC') or Baye ('BIC').

    """
    ax = plt.gca()
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    num_lags = len(lags_to_test)
    information_criteria = np.zeros(num_lags)
    for lag in list(lags_to_test)[:cap]:
        mod = ARMA(data.values, order=(lag, 0))
        res = mod.fit()
        if test_criteria == 'BIC':
            information_criteria[lag] = res.bic
            ax.set_title('Bayes Information Criterion', fontsize=20)
            ax.set_ylabel('BIC', fontsize=15)
        elif test_criteria == 'AIC':
            information_criteria[lag] = res.aic
            ax.set_title('Akaike Information Criterion', fontsize=20)
            ax.set_ylabel('AIC', fontsize=15)

    ax.set_xlabel('Lag', fontsize=15)
    ax.plot(lags_to_test[:cap], information_criteria[:cap], **kwds)
    ax.legend(loc='best')

    return ax


def ar_param_dictionary(train_df, order):
    """Parameters of autoregreesive models.

    Given a train df, this functions fits Auto Regressive models of any oder
    given for the different time series in the train df, and stores a summary
    of results, the AR1 value, and the volatility (syandard deviation) of
    observations.

    """
    ar_params = {}
    for i, col in enumerate(train_df):
        asset = train_df[col]
        mod = ARMA(asset, order=(order, 0))
        res = mod.fit()
        ar_params[i] = {'name': col,
                        'summary': res.summary(),
                        'AR1': res.arparams[0],
                        'vol': res.sigma2}
    return ar_params