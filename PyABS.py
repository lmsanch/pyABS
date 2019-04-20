# PyABS, light version of aspects of proprietary code
import pandas as pd
import numpy as np
from pandas.compat import lmap
from statsmodels.tsa.arima_model import ARMA
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    Given a time series, and significant lags returned by the
    autocorrelation_and_significance function, this function tests n values (cap)
    to find out if auto regresive models of order > 1 are worth exploring.
    Test criteria can be Akaike ('AIC') or Bayes ('BIC').
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
    """Parameters of autoregressive models.

    Given a train df, this functions fits auto regressive models of any oder
    given for the different time series in the train df and stores a summary
    of results, the AR1 value and the volatility (standard deviation) of
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


def simulate_correlated_random_numbers(corr_matrix, n=1000):
    """Multivariate random normal.

    A generalization of the one-dimensional normal distribution to higher
    dimensions, using Cholesky decomposition.
    https://math.stackexchange.com/questions/2079137/generating-multivariate-normal-samples-why-cholesky
    """
    upper_cholesky = cholesky(corr_matrix)
    rnd_numbers = np.random.normal(0.0, 1.0, size=(n, corr_matrix.shape[0]))
    ans = rnd_numbers@upper_cholesky
    return(ans)


def simulate_single_set_interest_rates(train_df, date_ix, ar_params_dict, vol_stress=1):
    """Simulate 1 path of multiple future interest rates.

    Given an historical time series of interest rates, an index of future dates,
    and a dictionary of autoregressive parameters for interest rates,
    this function generates a path of correlated interest rates.
    """
    corr_matrix = train_df.corr().as_matrix()
    fut_rates = {}
    for i in range(train_df.shape[1]):
        fut_rates[i] = np.zeros(len(date_ix))
        fut_rates[i][0] = train_df.iloc[-1, i]

    corr_rnd = simulate_correlated_random_numbers(corr_matrix)
    for k in range(train_df.shape[1]):
        for z in range(len(date_ix)):
            # skip the first value, since it is the seed value for the sim
            if z != 0:
                fut_rates[k][z] = fut_rates[k][z-1]*ar_params_dict[k]['AR1'] + ar_params_dict[k]['vol'] * corr_rnd[:, k][z]
                fut_rates[k][z] = (fut_rates[k][z])*(vol_stress)
                # for cases ofa simulatd negative spread, set the sread to the
                # previous positive spread
                if fut_rates[k][z] < 0:
                    fut_rates[k][z] = fut_rates[k][z-1]
    rates_df = pd.DataFrame(fut_rates, index=date_ix)
    rates_df.columns = train_df.columns
    return rates_df


def simulate_several_sets_correlated_rates(df_train, sims, date_index, ar_params_dict):
    """Simulate many paths of multiple future interest rates.

    Given an historical time series of interest rates in a df, the number of
    simulations to perform, an index of future dates, and a dictionary of
    autoregressive parameters for interest rates, this function generates a
    dictionary of data frames contaning paths for all correlated interest rates
    in the historical time series, plus a dictionary of dfs with the rates for
    specifc rates, for further analysis.
    """
    assets = df_train.columns.tolist()
    all_sims = {}
    for i in tqdm(range(sims)):
        all_sims[i] = simulate_single_set_interest_rates(df_train, date_index, ar_params_dict)
    master_sim = pd.DataFrame(pd.concat(all_sims, axis=1))
    master_sim.columns = master_sim.columns.get_level_values(1)
    asset_sim = {}
    for asset in assets:
        asset_sim[asset] = master_sim.filter(like=asset, axis=1)
        asset_sim[asset].columns = list(np.arange(sims))
    return all_sims, asset_sim


def estimate_1yr_transition(initial_rating='AAA'):
    """Simulate 1 period rating transition.

    This function estimates the transition from any given rating
    'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC' to the same rating plus the 'D'
    (default) state, based on observed trasitions for ABS in 1 year, excluding
    mortgages. These are approximations, and each asset class should have its
    own transition matrix.
    """
    data = np.array([[9.081e-01, 8.330e-02, 6.500e-03, 9.000e-04, 6.000e-04, 3.000e-04, 2.000e-04, 1.000e-04],
                     [7.000e-03, 9.065e-01, 7.790e-02, 6.400e-03, 6.000e-04, 1.300e-03, 2.000e-04, 1.000e-04],
                     [9.000e-04, 2.270e-02, 9.105e-01, 5.520e-02, 7.400e-03, 2.600e-03, 1.000e-04, 6.000e-04],
                     [2.000e-04, 3.300e-03, 5.950e-02, 8.693e-01, 5.300e-02, 1.170e-02, 1.200e-03, 1.800e-03],
                     [3.000e-04, 1.400e-03, 6.700e-03, 7.730e-02, 8.053e-01, 8.840e-02, 1.000e-02, 1.060e-02],
                     [0.000e+00, 1.100e-03, 2.400e-03, 4.300e-03, 6.480e-02, 8.346e-01, 4.070e-02, 5.210e-02],
                     [2.200e-03, 2.200e-03, 2.200e-03, 1.300e-02, 2.380e-02, 1.124e-01, 6.486e-01, 1.956e-01]])
    initial_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
    transition_to = initial_ratings.copy()
    transition_to.append('D')
    p_transition = pd.DataFrame(data, index=initial_ratings, columns=transition_to).transpose().to_dict()
    final = np.random.choice(list(p_transition[initial_rating].keys()), 1, p=list(p_transition[initial_rating].values()))[0]
    return final


def estimate_transition_vector(initial_rating, years):
    """Simulate rates upgrades or downgrades in n years.

    This function simulates the movement of the initial rating over time,
    by using the function recursively, i.e.: the initial rating feeds the
    function, and the output of the function feeds the function again until
    n periods have been completed. This is a one state Markov process.
    """
    input_list = [initial_rating]
    if input_list == []:
        return 0
    else:
        for i in range(years-1):
            new_rating = estimate_1yr_transition(initial_rating=input_list[-1])
            input_list.append(new_rating)
            if new_rating == 'D':
                return input_list
    return input_list


def simulate_purchase_per_sim_rate_scenario(purchase_weeks, sims, rates_sim_dict, spreads_dict, date_index, to_invest=[30, 20, 20, 20, 10]):
    """Simulate purchases per scenarios.

    This function simulates the purchase of assets under the different interest
    rates scenarios.
    """
    purchase_dict = {}
    for i in tqdm(range(sims)):
        capital = []
        assets = np.random.choice(list(p_issuance.keys()), purchase_weeks, p=list(p_issuance.values()))
        terms = np.random.choice(list(p_term.keys()), purchase_weeks, p=list(p_term.values()))
        for asset, term in list(zip(assets, terms)):
            capital.append(f_risk_capital[asset][term])
        data_dict = {'rate_label': list(assets),
                     'term': list(terms),
                     'f_risk_capital': capital}
        purchase_dict[i] = pd.DataFrame(data_dict)
        purchase_dict[i]['purchase_week'] = purchase_dict[i].index+1
        purchase_dict[i]['purchase_date'] = date_index[purchase_dict[i]['purchase_week']]
        purchase_dict[i]['maturity_date'] = date_index[purchase_dict[i]['purchase_week']+(purchase_dict[i]['term']*52)]
        purchase_dict[i]['asset'] = purchase_dict[i]['term'].astype(str) + '_yr_' + purchase_dict[i]['rate_label']
        purchase_dict[i]['benchmark_asset'] = '3_yr_' + purchase_dict[i]['rate_label']

    for i in range(sims):
        for ix, col in purchase_dict[i].iterrows():
            purchase_dict[i].at[ix, 'benchmark_ABS_spread'] = rates_sim_dict[i].iloc[col['purchase_week']][col['benchmark_asset']]
            purchase_dict[i].at[ix, 'libor'] = rates_sim_dict[i].iloc[col['purchase_week']]['libor']
            purchase_dict[i].at[ix, 'risk_capital'] = to_invest[ix]
            purchase_dict[i].at[ix, 'fed_loan'] = (to_invest[ix]/col['f_risk_capital'])-to_invest[ix]
            purchase_dict[i].at[ix, 'final_rating'] = estimate_transition_vector('AAA', col['term'])[-1]
        purchase_dict[i]['total_purchase'] = purchase_dict[i]['fed_loan'] + purchase_dict[i]['risk_capital']
        purchase_dict[i]['spread_over_libor'] = purchase_dict[i]['benchmark_ABS_spread'] - (purchase_dict[i]['asset'].map(spreads_dict))
        r1 = ((purchase_dict[i]['libor'] + purchase_dict[i]['spread_over_libor']))*purchase_dict[i]['f_risk_capital']
        r2 = ((purchase_dict[i]['spread_over_libor'])-100)*(1-purchase_dict[i]['f_risk_capital'])
        purchase_dict[i]['exp_annual_r'] = ((r1+r2)/(purchase_dict[i]['f_risk_capital']))/10000
        purchase_dict[i] = purchase_dict[i][['asset',
                                             'purchase_week',
                                             'purchase_date',
                                             'maturity_date',
                                             'term',
                                             'f_risk_capital',
                                             'risk_capital',
                                             'fed_loan',
                                             'total_purchase',
                                             'libor',
                                             'spread_over_libor',
                                             'exp_annual_r',
                                             'final_rating']]
        for col in ['risk_capital', 'fed_loan', 'total_purchase', 'libor', 'spread_over_libor']:
            purchase_dict[i][col] = purchase_dict[i][col].astype(int)

    return purchase_dict


def plot_sim_and_real(asset, assets_sims, sims, df_test, cap=10, percentiles = [25,50,75]):
    cols = assets_sims[asset].columns
    # plot the percentile bands
    for i in range(len(percentiles)):
        assets_sims[asset]['p'+ str(percentiles[i])] = assets_sims[asset][cols].apply(lambda x: np.percentile(x, percentiles[i]), axis=1)
        assets_sims[asset]['p'+ str(percentiles[i])][:cap].plot(color='k', linestyle='dotted', linewidth=3)
    # plot the simulation paths
    for i in range(sims):
        assets_sims[asset][i][:cap].plot(color='blue', alpha=0.01)
    # plot the real rates
    df_weekly_test[asset][:cap].plot(figsize=(20,10), color='orange', linewidth=4)


f_risk_capital = {'auto_AAA':         {1: 0.10,
                                       2: 0.11,
                                       3: 0.12},
                  'student_loan_AAA': {1: 0.08,
                                       2: 0.09,
                                       3: 0.10},
                  'helc_AAA':         {1: 0.12,
                                       2: 0.13,
                                       3: 0.14},
                  'credit_card_AAA':  {1: 0.05,
                                       2: 0.05,
                                       3: 0.06}}


p_term = {1: 0.20,
          2: 0.30,
          3: 0.50}


p_issuance = {'auto_AAA':         0.20,
              'student_loan_AAA': 0.20,
              'helc_AAA':         0.30,
              'credit_card_AAA':  0.30}
