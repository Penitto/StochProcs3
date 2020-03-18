import datetime
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


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
