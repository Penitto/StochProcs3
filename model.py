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