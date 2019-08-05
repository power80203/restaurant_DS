import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import sin, cos, sqrt, atan2, radians

def computeGeoDistance(lat1, lon1, lat2, lon2):

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    # print("Result:", distance)
    # print("Should be:", 278.546, "km")

    return round(distance,4)

def missing_data(data_frame):
    total = data_frame.isnull().sum().sort_values(ascending=False)
    percent = (data_frame.isnull().sum()/data_frame.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    

    return missing_data

def deal_synonyms(i, dict_synonyms):
    if i.strip() in dict_synonyms:
        i = dict_synonyms[i.strip()]
    return i

if __name__ == "__main__":
    lat1 = 52.2296756
    lon1 = 21.0122287
    lat2 = 52.406374
    lon2 = 16.9251681
    computeGeoDistance(lat1, lon1, lat2, lon2)
        