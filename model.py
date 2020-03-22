import datetime
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import ruptures as rpt
from cumsum import change_point_detection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import logging 

def get_logger():
    # From Official documentation
    logger = logging.getLogger('base')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'./logs from {datetime.datetime.isoformat(datetime.datetime.now())[:10]}.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
logger=get_logger()





def get_change_point(series, jump=5, n_bkps=5, pen=10):
    """

    series: numpy array please
    jump: размер сэмпла
    n_bkps: количество возвращаемых остановок
    pen: пенальти для Pelt

    """

    alg_dynp = rpt.Dynp(jump=jump).fit_predict(series, n_bkps=n_bkps)

    alg_pelt = rpt.Pelt(jump=jump).fit_predict(series, pen=pen)

    alg_bin = rpt.Binseg(jump=jump).fit_predict(series, n_bkps=n_bkps)

    alg_bot = rpt.BottomUp(jump=jump).fit_predict(series, n_bkps=n_bkps)

    alg_win = rpt.Window(jump=jump).fit_predict(series, n_bkps=n_bkps)

    alg_cumsum = change_point_detection(series.tolist())
    
    # Получили разладки от нескольких алгоритмов
    # Теперь найдём точки, которые предсказывались алгоритмами несколько раз
    res = {}
    for i in alg_dynp + alg_pelt + alg_bin + alg_bot + alg_win + alg_cumsum:
        if i in res:
            res[i] += 1
        else:
            res[i] = 1

    del res[0]
    del res[len(series)]

    itemMaxValue = max(res.items(), key=lambda x: x[1])
    listOfKeys = []
    for key, value in res.items():
        if value == itemMaxValue[1]:
            listOfKeys.append(key)
    return listOfKeys   

def is_change_point(series, jump=5, n_bkps=5, pen=10, lim=10):
    res = get_change_point(series, jump, n_bkps, pen)
    return res[-1] >= len(series) - lim

def get_model(data, target):
    params1 = {'alpha' : np.linspace(0.0001, 1, 100), 
               'l1_ratio' : np.linspace(0, 1, 100)}
    
    params2 = {'n_estimators' : range(10, 101, 10),
               'max_depth' : range(2,10)}

    params3 = {'learning_rate' : np.linspace(0.0001, 0.3, 50),
               'n_estimators' : range(10, 101, 10),
               'max_depth' : range(2, 10)}

    el = ElasticNet()
    gr_el = GridSearchCV(el, params1, cv=TimeSeriesSplit(), scoring='neg_mean_error',refit=True)
    gr_el.fit(data, target)

    rf = RandomForestRegressor()
    gr_rf = GridSearchCV(rf, params2, cv=TimeSeriesSplit(), scoring='neg_mean_error',refit=True)
    gr_rf.fit(data, target)

    lgb = LGBMRegressor()
    gr_lgb = GridSearchCV(lgb, params3, cv=TimeSeriesSplit(), scoring='neg_mean_error',refit=True)
    gr_lgb.fit(data, target)

    res_scores = {'elastic' : gr_el.best_score_, 
                  'random_forest' : gr_rf.best_score_,
                  'lgbm' : gr_lgb.best_score_}

    res_est = {'elastic' : gr_el.best_estimator_, 
               'random_forest' : gr_rf.best_estimator_,
               'lgbm' : gr_lgb.best_estimator_}

    return res_est[sorted(res_scores, key=lambda x: (-res_scores[x], x))[0]]

def is_weekday(date):
    with open("prod_cal.json", "r") as read_file:
        holidays = json.load(read_file)
    return (date.weekday() == 5)|(date.weekday() == 6)|(date.date().isoformat() in holidays[str(date.year)])

def get_dates_list(target_data_file='project_3_train+test.xlsx'):
    base_index = pd.DatetimeIndex(
        start=pd.read_excel(target_data_file)['Date'].min(),
        end=pd.read_excel(target_data_file)['Date'].max(),
        freq='D')

    base_index_series=pd.Series(base_index)
    return base_index_series[~base_index_series.apply(is_weekday)]

def plotMovingAverage(df, series, n):
    rolling_mean = series.rolling(window=n).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(n))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    plt.plot(df[n:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

def call_script(data_file_path, date):
    pass 
    forecast_on_next_day=None
    return None

def generate_features(
    df #датафрейм со всеми содержательными фичами, даты в качестве индекса 
    ):
    new_df=df.copy()
    pass#логика
    return new_df

def select_features(
    df,#датафрейм со всеми фичами, в т ч генерированными
    target, #серия с таргетом
    ):
    y = df.pop('target')
    #логика
    selected_features=[]
    return selected_features



def calibrate_hyper(
    df,
    target,
    model):
    model_spec = get_model_spec()
    trained_model=None
    return trained_model


def get_data(date_until,target_data_file='project_3_train+test.xlsx'):
    base_dates = get_dates_list(target_data_file)
    if base_dates.max()>date_until:
        base_dates = base_dates[base_dates<=date_until]
    base_df=pd.DataFrame(index=base_dates)
    
    target = pd.read_excel(target_data_file)
    target=target.set_index('Date')
    target=target.loc[base_df.index]

    
    
    data_moex = pd.read_csv('MOEX Russia Historical Data.csv', index_col='Date',sep=';').iloc[::-1].reset_index()
    data_usdrub = pd.read_csv('USD_RUB Historical Data.csv', index_col='Date').iloc[::-1].reset_index()
    data_brent = pd.read_csv('Brent Oil Futures Historical Data.csv', index_col ='Date').iloc[::-1].reset_index()

    data_moex['Date'] = pd.to_datetime(data_moex['Date'])
    data_usdrub['Date'] = pd.to_datetime(data_usdrub['Date'])
    data_brent['Date'] = pd.to_datetime(data_brent['Date'])
    
    data_moex = data_moex.rename(columns={'Price':'MOEX'})
    data_usdrub = data_usdrub.rename(columns={'Price':'USDRUB'})
    data_brent = data_brent.rename(columns={'Price':'BRENT'})
    
    data_moex1 = data_moex[['Date', 'MOEX']]
    data_usdrub1 = data_usdrub[['Date', 'USDRUB']]
    data_brent1 = data_brent[['Date', 'BRENT']]

    data_keyrate = pd.read_excel('keyrate_cbr.xlsx').iloc[::-1].reset_index(drop=True)
    data_liquidity = pd.read_excel('liquidity_cbr.xlsx').iloc[::-1].reset_index(drop=True)
    
    data_keyrate = data_keyrate.rename(columns={'Дата':'Date'})
    data_liquidity = data_liquidity.rename(columns={'Дата':'Date'})

    econ_data = base_df.join(other=[
        data_moex1.set_index('Date'),
        data_usdrub1.set_index('Date'),
        data_brent1.set_index('Date'),
        data_keyrate.set_index('Date'),
        data_liquidity.set_index('Date'),
    ], how='left')

    return econ_data, target


def get_model_spec():
    spec=dict()
    return spec



def report_metric(
    target, 
    predictions):
    metric = target-predictions
    print(f'metric is {metric.mean()}')
    return metric

STARTING_TICK=100
PREDICTIONS_FILEPATH='predictions.xlsx'



def prepare_complete_model_and_data(date,target_data_file):

    data, target = get_data(date, target_data_file)
    full_data = generate_features(data)
    selected_features = select_features(
        full_data.loc[:date],
        target.loc[   :date],)
    clean_data = full_data.loc[:,selected_features]
    model = get_model(
        clean_data.loc[:data],
        target.loc[    :data])
    return model, clean_data, target



def general_loop(target_data_file):
    datelist = get_dates_list(target_data_file)
    start_date = datelist[STARTING_TICK]
    model, data, target = prepare_complete_model_and_data(start_date,target_data_file)
    logger.info('Initial model and data are ready')
    predictions = pd.Series(index=target.index)
    for day_count, cur_date in enumerate(datelist[STARTING_TICK+1:]):
        predictions[cur_date] = model.predict(data.loc[cur_date])
        predictions.to_excel(PREDICTIONS_FILEPATH)
        if is_change_point(target[:cur_date]):
            logger.info(f'Change-point detected on {cur_date}')
            model, data, target = prepare_complete_model_and_data(cur_date, target_data_file)
    metric_results = report_metric(target, predictions)

