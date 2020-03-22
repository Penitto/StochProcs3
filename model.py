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
import warnings
warnings.filterwarnings('ignore')

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
    series = series.values
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

def get_model(data, target,use_ensemble=True):
    params1 = {'alpha' : np.logspace(0.0001, 1, 30), 
               'l1_ratio' : np.linspace(0, 1, 10)}
    
    params2 = {'n_estimators' : range(10, 101, 10),
               'max_depth' : range(2,10)}

    params3 = {'learning_rate' : np.logspace(0.0001, 0.5, 10),
               'n_estimators' : range(10, 101, 30),
               'max_depth' : [6,9,12],
               'verbose':[-1]}

    lgb = LGBMRegressor()
    
    gr_lgb = GridSearchCV(lgb, params3, cv=TimeSeriesSplit(), scoring='neg_mean_squared_error',refit=True)
    gr_lgb.fit(data, target)
    logger.info('Booster params discovered')

    el = ElasticNet(max_iter=5000)
    gr_el = GridSearchCV(el, params1, cv=TimeSeriesSplit(), scoring='neg_mean_squared_error',refit=True)
    gr_el.fit(data, target)
    logger.info('ElasticNet params discovered')

    rf = RandomForestRegressor()
    gr_rf = GridSearchCV(rf, params2, cv=TimeSeriesSplit(), scoring='neg_mean_squared_error',refit=True)
    gr_rf.fit(data, target)
    logger.info('RandomForest params discovered')


    res_scores = {'elastic' : gr_el.best_score_, 
                  'random_forest' : gr_rf.best_score_,
                  'lgbm' : gr_lgb.best_score_}

    res_est = {'elastic' : gr_el.best_estimator_, 
               'random_forest' : gr_rf.best_estimator_,
               'lgbm' : gr_lgb.best_estimator_}
    if use_ensemble:
        estimators = [
                    ('elastic', ElasticNet(**gr_el.best_params_)), 
                    ('random_forest', RandomForestRegressor(**gr_rf.best_params_)),
                    ('lgbm', LGBMRegressor(**gr_lgb.best_params_))]


        from sklearn.ensemble import StackingRegressor


        stacked = StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(n_estimators=30),
            passthrough=True)
        stacked.fit(data, target)
        logger.info('Ensemble fitted')
        return stacked
    return res_est[sorted(res_scores, key=lambda x: (-res_scores[x], x))[0]]

def add_tax_dates(df, example='Target'):
    """
        df: датафрейм, в который вносим данные
        example : название столбца, который можно взять как образец формы для np.ones_like
            default: 'Target'
    """

    with open("corporate_tax.json", "r") as read_file:
        corporate_tax = json.load(read_file)

    with open("value_added_tax.json", "r") as read_file:
        val_add_tax = json.load(read_file)

    # Налог на прибыль
    df['corp_tax'] = np.ones_like(df[example].values)
    corp = {i : [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in corporate_tax[i]] for i in corporate_tax}
    for i in df.index:
        # Если не налоговый день
        if not ((i >= corp[str(i.year)][0]) and (i <= corp[str(i.year)][1])):
            # Если до первого марта
            if corp[str(i.year)][0] > i:
                data['corp_tax'][i] = (i.date() - corp[str(i.year - 1)][1]).days / (corp[str(i.year)][0] - corp[str(i.year - 1)][1]).days
            else:
                data['corp_tax'][i] = (i.date() - corp[str(i.year)][1]).days / (corp[str(i.year + 1)][0] - corp[str(i.year)][1]).days

    # Налог добавленной стоимости
    df['val_add_tax'] = np.ones_like(df[example].values)
    val_add = {i : {j : [datetime.datetime.strptime(k, '%Y-%m-%d').date() for k in val_add_tax[i][j]] for j in val_add_tax[i]} for i in val_add_tax}
    for i in df.index:
        # Если не налоговый день
        if ((i > val_add[str(i.year)]['1'][1]) and (i < val_add[str(i.year)]['2'][0])) \
            or ((i > val_add[str(i.year)]['2'][1]) and (i < val_add[str(i.year)]['3'][0])) \
            or ((i > val_add[str(i.year)]['3'][1]) and (i < val_add[str(i.year)]['4'][0])) \
            or ((i > val_add[str(i.year)]['4'][1]) and (i < val_add[str(i.year + 1)]['1'][0])):
                if i.quarter == 1:
                    df['val_add_tax'][i] = (i.date() - val_add[str(i.year)]['1'][1]).days / (val_add[str(i.year)]['2'][0] - val_add[str(i.year)]['1'][1]).days
                elif i.quarter == 2:
                    df['val_add_tax'][i] = (i.date() - val_add[str(i.year)]['2'][1]).days / (val_add[str(i.year)]['3'][0] - val_add[str(i.year)]['2'][1]).days
                elif i.quarter == 3:
                    df['val_add_tax'][i] = (i.date() - val_add[str(i.year)]['3'][1]).days / (val_add[str(i.year)]['4'][0] - val_add[str(i.year)]['3'][1]).days
                else:
                    df['val_add_tax'][i] = (i.date() - val_add[str(i.year)]['4'][1]).days / (val_add[str(i.year + 1)]['1'][0] - val_add[str(i.year)]['4'][1]).days

    return df





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
    return df.columns





def get_data(target_data_file='project_3_train+test.xlsx'):
    base_dates = get_dates_list(target_data_file)
    # if base_dates.max()>date_until:
    #     base_dates = base_dates[base_dates<=date_until]
    base_df=pd.DataFrame(index=base_dates)
    
    target = pd.read_excel(target_data_file)
    target=target.set_index('Date')
    target=target.loc[base_df.index].iloc[:,0]

    
    
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

    data_keyrate = pd.read_excel('keyrate_cbr.xlsx').iloc[::-1].replace({'—':np.nan}).reset_index(drop=True)
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
    econ_data = econ_data.interpolate(method='polynomial', order=3)
    return econ_data, target




def report_metric(
    target, 
    predictions):
    metric = np.abs(target-predictions)
    print(f'metric is {metric.mean()}')
    return metric

STARTING_TICK=100
PREDICTIONS_FILEPATH='predictions.xlsx'



def prepare_complete_model_and_data(date,target_data_file):

    data, target = get_data(target_data_file)
    logger.info('Raw data fetched')
    full_data = generate_features(data)
    logger.info('New features generated')
    selected_features = select_features(
        full_data.loc[:date],
        target.loc[   :date],)
    clean_data = full_data.loc[:,selected_features]
    logger.info('Data selected')
    model = get_model(
        clean_data.loc[:date],
        target.loc[    :date])
    return model, clean_data, target



def general_loop(target_data_file='project_3_train+test.xlsx'):
    logger.info('Script started!')
    datelist = get_dates_list(target_data_file)
    start_date = datelist[STARTING_TICK]
    model, data, target = prepare_complete_model_and_data(start_date,target_data_file)
    logger.info('Initial model and data are ready')
    predictions = pd.Series(index=datelist)
    for day_count, cur_date in enumerate(datelist.values[STARTING_TICK+1:]):
        predictions.loc[cur_date] = model.predict(data.loc[cur_date].values.reshape(1,-1))
        predictions.to_excel(PREDICTIONS_FILEPATH)
        logger.info(f'Day {day_count} predicted, absolute error of {np.abs(predictions.loc[cur_date]-target.loc[cur_date])}')
        if is_change_point(target[:cur_date]):
            logger.info(f'Change-point detected on {cur_date}')
            model, data, target = prepare_complete_model_and_data(cur_date, target_data_file)
    metric_results = report_metric(target, predictions)

if __name__=='__main__':
    general_loop()