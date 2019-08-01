import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))
import config


def storePayment():
    df_chefmozaccepts = pd.read_csv('%s/chefmozaccepts.csv'%config.raw_data_path, encoding = "latin1")

    return df_chefmozaccepts

def storeCuisine():
    df_chefmozcuisine = pd.read_csv('%s/chefmozcuisine.csv'%config.raw_data_path, encoding = "latin1")
    return df_chefmozcuisine

def storeHours():
    df_chefmozhours4 = pd.read_csv('%s/chefmozhours4.csv'%config.raw_data_path, encoding = "latin1")

    return df_chefmozhours4

def storeParking():
    df_chefmozparking = pd.read_csv('%s/chefmozparking.csv'%config.raw_data_path, encoding = "latin1")

    return df_chefmozparking

def storeGeo():
    df_geoplaces2 = pd.read_csv('%s/geoplaces2.csv'%config.raw_data_path, encoding = "latin1")
    df_geoplaces2 = df_geoplaces2.drop('name', axis =1)
    return df_geoplaces2

# user

def userCuisine():
    df_usercuisine = pd.read_csv('%s/usercuisine.csv'%config.raw_data_path, encoding = "latin1")
    return df_usercuisine

def userPayment():
    df_userpayment = pd.read_csv('%s/userpayment.csv'%config.raw_data_path, encoding = "latin1")
    return df_userpayment

def userProfile():
    df_userprofile = pd.read_csv('%s/userprofile.csv'%config.raw_data_path, encoding = "latin1")
    return df_userprofile

def rating():
   df_rating = pd.read_csv('%s/rating_final.csv'%config.raw_data_path, encoding = "latin1")
   return df_rating






if __name__ == "__main__":
    # cv1 = pd.read_csv('%s/geoplaces2.csv'%config.raw_data_path, encoding = "latin1")
    # cv1 = cv1.drop('name', axis =1)
    # print(cv1)
    a = chefmozaccepts()
    b = geoplaces()
    print(b)