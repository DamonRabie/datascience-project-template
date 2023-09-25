import logging
import os
import warnings
from itertools import product

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

warnings.filterwarnings('ignore')


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def plot_components(model: Prophet, forecast: pd.DataFrame):
    return model.plot_components(forecast)


def plot_test_forecast(test_df: pd.DataFrame, forecast: pd.DataFrame):
    import plotly.express as plx

    check = pd.merge(test_df[['ds', 'y']], forecast, on='ds')

    return plx.line(data_frame=check, x='ds', y=['y', 'yhat_lower', 'yhat', 'yhat_upper'])


def plot_forecast(model: Prophet, forecast: pd.DataFrame, figsize: tuple[int]):
    from prophet.plot import plot_plotly

    fig = plot_plotly(model, forecast, figsize=figsize)

    return fig


def test_metric(actual, prediction, metric):
    """
    available metrics: rmse, mse, mae, mape
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

    if metric == 'rmse':
        return mean_squared_error(actual, prediction, squared=False)
    elif metric == 'mse':
        return mean_squared_error(actual, prediction, squared=True)
    elif metric == 'mae':
        return mean_absolute_error(actual, prediction)
    elif metric == 'mape':
        return mean_absolute_percentage_error(actual, prediction)
    else:
        return None


def tune_hyper_parameter(train_df: pd.DataFrame, test_df: pd.DataFrame, holiday_df: pd.DataFrame, pred_horizon: int,
                         param_grid: dict, method: str, metric: str) -> pd.DataFrame:
    """
    available methods: cross_validation, test_set
    available metrics: rmse, mse, mae, mape
    """

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

    # Store results
    metric_results = []

    # Use cross validation to evaluate all parameters
    for params in all_params:
        if method == 'cross_validation':
            df_cv, df_p = cross_validation_prophet(train_df=train_df, model_params=params, holiday_df=holiday_df,
                                                   pred_horizon=f'{pred_horizon} days')

            metric_results.append(df_p[metric].values[0])
            logging.info(f"prophet model cross validation: {params} _ {df_p[metric].values[0]}")

        if method == 'test_set':
            model = create_model(train_df, params, holiday_df)

            forecast = model.predict(test_df)
            metric_results.append(test_metric(test_df['y'], forecast['yhat'], metric=metric))

    # Convert to Pandas DF
    tuning_results = pd.DataFrame(all_params)
    tuning_results[metric] = metric_results
    tuning_results.sort_values(metric, inplace=True)

    return tuning_results


def cross_validation_prophet(train_df: pd.DataFrame, model_params: dict, holiday_df: pd.DataFrame,
                             pred_horizon: str) -> (pd.DataFrame, pd.DataFrame):
    with suppress_stdout_stderr():
        model = create_model(train_df, model_params, holiday_df)
        df_cv = cross_validation(model, horizon=pred_horizon, parallel="processes")
        df_p = performance_metrics(df_cv)

    return df_cv, df_p


def create_model(train_df: pd.DataFrame, model_params: dict, holiday_df: pd.DataFrame):
    with suppress_stdout_stderr():
        # create prophet model
        if model_params is None:
            model = Prophet(holidays=holiday_df)
        else:
            model = Prophet(holidays=holiday_df, **model_params)

        # add extra regressors
        for col in train_df.columns:
            if col in ['ds', 'y']:
                continue
            model.add_regressor(col)

        # fit model
        model.fit(train_df)

    return model

# %%
