import datetime
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import ruptures as rpt
from cumsum import change_point_detection


def detect_change_point(series, jump, n_bkps, pen):
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

def is_weekday(date):
    with open("prod_cal.json", "r") as read_file:
        holidays = json.load(read_file)
    return True if (date.weekday() == 5) or (date.weekday() == 6) or (date.date().isoformat() in holidays[str(date.year)]) else False

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
    target_name='target' #название колонки с таргетом
    ):
    y = df.pop('target')
    #логика
    selected_features=[]
    return df[selected_features+[target_name]]


def adjustment_detection(idontknow):
    return None

    
def feature_selection_stability_control(idontknow):
    return None

def calibrate_hyper(df, model, model_spec):
    return model_spec


def get_data():
    df = None
    return df
