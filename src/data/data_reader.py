import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))
import config


def chefmozaccepts():
    df_chefmozaccepts = pd.read_csv('%s/chefmozaccepts.csv'%config.raw_data_path, encoding = "latin1")

    return df_chefmozaccepts

def chefmozcuisine():
    df_chefmozcuisine = pd.read_csv('%s/chefmozcuisine.csv'%config.raw_data_path, encoding = "latin1")
    return df_chefmozcuisine

def chefmozhours4():
    df_chefmozhours4 = pd.read_csv('%s/chefmozhours4.csv'%config.raw_data_path, encoding = "latin1")

    return df_chefmozhours4

def chefmozparking():
    df_chefmozparking = pd.read_csv('%s/chefmozparking.csv'%config.raw_data_path, encoding = "latin1")

    return df_chefmozparking

def geoplaces():
    df_geoplaces2 = pd.read_csv('%s/geoplaces2.csv'%config.raw_data_path, encoding = "latin1")
    df_geoplaces2 = df_geoplaces2.drop('name', axis =1)
    return df_geoplaces2

# user

def usercuisine():
    df_usercuisine = pd.read_csv('%s/usercuisine.csv'%config.raw_data_path, encoding = "latin1")
    return df_usercuisine

def userpayment():
    df_userpayment = pd.read_csv('%s/userpayment.csv'%config.raw_data_path, encoding = "latin1")
    return df_userpayment

def userprofile():
    df_userprofile = pd.read_csv('%s/userprofile.csv'%config.raw_data_path, encoding = "latin1")
    return df_userprofile







if __name__ == "__main__":
    # cv1 = pd.read_csv('%s/geoplaces2.csv'%config.raw_data_path, encoding = "latin1")
    # cv1 = cv1.drop('name', axis =1)
    # print(cv1)
    a = chefmozaccepts()
    b = geoplaces()
    print(b)