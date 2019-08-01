import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))
import config
from src.data import data_reader


def userDataFactroy():
    df_profile = data_reader.userProfile()
    df_final_rating = data_reader.rating()
    df_userpayment = data_reader.userPayment()
    df_userRcusine = data_reader.userCuisine()
    print(df_final_rating.shape)
    df_user = df_final_rating.merge(df_profile, left_on='userID', right_on='userID', how = 'left')
    print(df_user.shape)

    # dealing with user payment
    user_payment_dict = dict()
    for i in df_userpayment.values:
        if i[0] not in user_payment_dict:
            user_payment_list = list()
            user_payment_dict[i[0]] = user_payment_list
            user_payment_dict[i[0]].append(i[1])
        else:
            user_payment_dict[i[0]].append(i[1])
        # print(user_payment_dict) 

    id_list = list()
    payment_list = list()

    for key, value in user_payment_dict.items():
        id_list.append(key)
        payment = [str(x) for x in value]
        str_temp = ','.join(payment)
        payment_list.append(str_temp)
    
    df_userpayment = pd.DataFrame({"userID":id_list, "Upayment":payment_list})
    
    df_user = df_user.merge(df_userpayment, left_on='userID', right_on='userID', how = 'left')
    print(df_user.shape)

    df_user.to_csv('%s/user_merged.csv'%config.interim_data_path, index = False)

    return df_user

def rastaurantDataFactroy():
    pass


if __name__ == "__main__":
    userDataFactroy()