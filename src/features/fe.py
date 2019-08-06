import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))
import config
from src.data import data_reader
from src.utli import utli

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
 
    number_payment = [len(i.split(',')) if len(i.split(',')) else 1 for i in payment_list]
    
    df_userpayment = pd.DataFrame({"userID":id_list, "Upayment":payment_list, 'num_of_Upayment' : number_payment})
    
    df_user = df_user.merge(df_userpayment, left_on='userID', right_on='userID', how = 'left')
    print(df_user.shape)

    # dealing with user cusine
    user_cusine_dict = dict()
    for i in df_userRcusine.values:
        if i[0] not in user_cusine_dict:
            user_cusine_list = list()
            user_cusine_dict[i[0]] = user_cusine_list
            user_cusine_dict[i[0]].append(i[1])
        else:
            user_cusine_dict[i[0]].append(i[1])
        # print(user_payment_dict) 

    id_list = list()
    cusine_list = list()

    for key, value in user_cusine_dict.items():
        id_list.append(key)
        tmp = [str(x) for x in value]
        str_temp = ','.join(tmp)
        cusine_list.append(str_temp)
    
    df_user_cusine = pd.DataFrame({"userID":id_list, "user_cusine":cusine_list})
    
    df_user = df_user.merge(df_user_cusine, left_on='userID', right_on='userID', how = 'left')
    print(df_user.shape)

    df_user.to_csv('%s/user_merged.csv'%config.interim_data_path, index = False)

    return df_user

def rastaurantDataFactroy():
    df_storeGeo =  data_reader.storeGeo()
    df_storeCuisine =  data_reader.storeCuisine()
    df_storeHours =  data_reader.storeHours()
    df_storeParking =  data_reader.storeParking()
    df_storePayment = data_reader.storePayment()


    df_storeGeo.drop('franchise', axis =1)
    df_storeGeo.drop('fax', axis =1)
    df_storeGeo.drop('zip', axis =1)
    df_storeGeo.drop('url', axis =1)

    df_storeGeo = df_storeGeo.rename(columns = {'the_geom_meter':'store_the_geom_meter',
                                      'address':'store_address', 'city': 'store_city',
                                      'state':'store_state', 'country':'store_country',
                                      'alcohol' : 'store_alcohol', 'smoking_area' : 'store_smoking_area',
                                      'dress_code' : 'store_dress_code', 'accessibility': 'store_accessibility',
                                      'price' : 'store_price', 'Rambience':'store_Rambience',
                                      'area':'store_area', 'other_services':'store_other_services'
                                      })



    #########################################################
    # dealing with multiple df_storeCuisine 

    store_storeCuisine_dict = dict()
    for cuisin in df_storeCuisine.values:
        if cuisin[0] not in store_storeCuisine_dict:
            store_storeCuisine_dict[cuisin[0]] = set()
            store_storeCuisine_dict[cuisin[0]].add(cuisin[1])
        else:
            store_storeCuisine_dict[cuisin[0]].add(cuisin[1])

    id_list = list()
    cuisin_list = list()
    for i, c in store_storeCuisine_dict.items():
        id_list.append(i)
        cuisin_list.append(c)

    number_of_store_cuisin = [ (len(i)) for i in cuisin_list ]

    df_storeCuisine = pd.DataFrame({'placeID':id_list,
                                    'store_cuisin' : cuisin_list,
                                    'number_of_store_cuisin' : number_of_store_cuisin,
                                  })

    df_store = df_storeGeo.merge(df_storeCuisine, on =  'placeID', how = 'left')
    print(df_store.shape)

    #########################################################
    # dealing with multiple row of one attri

    store_hours_dict = dict() # (shopid : {day : hour})

    for i in df_storeHours.values:
        if i[0] not in store_hours_dict:
            store_hoursByday_dict = dict()
            store_hours_dict[i[0]] = store_hoursByday_dict
            i[2] = i[2].strip().split(';')
            i[2].pop()
            i[1] = str(i[1]).split(';')
            i[1].pop()
            for day in i[2]:
                store_hours_list = set()
                store_hoursByday_dict[day] = store_hours_list
                for hr in i[1]:
                    store_hours_list.add(hr)
        else:
            i[2] = i[2].strip().split(';')
            i[2].pop()
            i[1] = str(i[1]).strip().split(';')
            i[1].pop()
            for day in i[2]:
                if day not in store_hours_dict[i[0]]:
                    store_hours_list = set()
                    store_hours_dict[i[0]][day] = store_hours_list
                    for hr in i[1]:
                        store_hours_list.add(hr)
                        
    full_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i in store_hours_dict.keys():
        if len(store_hours_dict[i].keys()) < 7:
            lack_list = list()
            for full in full_list:
                if full not in store_hours_dict[i].keys():
                    lack_list.append(full)
            for lack in lack_list:
                store_hours_dict[i][lack] = ''


    store_hours_dict_mon = [ store_hours_dict[key]['Mon'] for key in store_hours_dict.keys()]
    store_hours_dict_tue = [ store_hours_dict[key]['Tue'] for key in store_hours_dict.keys()]
    store_hours_dict_wed = [ store_hours_dict[key]['Wed'] for key in store_hours_dict.keys()]
    store_hours_dict_thu = [ store_hours_dict[key]['Thu'] for key in store_hours_dict.keys()]
    store_hours_dict_fri = [ store_hours_dict[key]['Fri'] for key in store_hours_dict.keys()]
    store_hours_dict_sat = [ store_hours_dict[key]['Sat'] for key in store_hours_dict.keys()]
    store_hours_dict_sun = [ store_hours_dict[key]['Sun'] for key in store_hours_dict.keys()]
    
    store_id = [x for x in store_hours_dict.keys()]
    df_storeHours = pd.DataFrame({"placeID":store_id,
                                  "mon_hours":store_hours_dict_mon,
                                  "tue_hours":store_hours_dict_tue,
                                  "wed_hours":store_hours_dict_wed,
                                  "thu_hours":store_hours_dict_thu,
                                  "fri_hours":store_hours_dict_fri,
                                  "sat_hours":store_hours_dict_sat,
                                  "sun_hours":store_hours_dict_sun,
                                })
    df_storeHours.to_csv('%s/df_storeHours.csv'%config.interim_data_path, index = False)

    # merge

    df_store = df_store.merge(df_storeHours, on =  'placeID', how = 'left')
    print(df_store.shape)

    #########################################################
    # dealing with parking issue of that dataset

    store_parking_dict = dict()
    for i in df_storeParking.values:
        if i[0] not in store_parking_dict:
            store_parking_dict[i[0]] = set()
            for park in i[1]:
                store_parking_dict[i[0]].add(i[1])
        else:
            store_parking_dict[i[0]].add(i[1])
    id_list = list()
    park_list = list()
    for store_id, park in store_parking_dict.items():
        id_list.append(store_id)
        park = list(park)
        if len(park) < 2:
            park_list.append(park[0])
            # print(store_id, park[0])
        else:
            park_temp = list()
            for p in park:
                park_temp.append(p)
            str1 = "@"
            park_temp = str1.join(park_temp)
            park_list.append(park_temp)
            # print(store_id, park_temp)
        

    df_store_parking = pd.DataFrame({'placeID': id_list,
                                     'park' : park_list})

    df_store_parking.to_csv('%s/df_store_parking.csv'%config.interim_data_path, index = False)
    # merge

    df_store = df_store.merge(df_store_parking, on =  'placeID', how = 'left')
    print(df_store.shape)
    #########################################################
    # dealing with storePayment issue of that dataset  

    df_storePayment_o_total = df_storePayment.shape[0]

    storePayment_dict = dict()
    for i in df_storePayment.values:
        if i[0] not in storePayment_dict:
            storePayment_dict[i[0]] = set()           
            storePayment_dict[i[0]].add(i[1])
        else:
            storePayment_dict[i[0]].add(i[1])

    id_list = list()
    payment_list = list()

    for store_id, payment in storePayment_dict.items():
        id_list.append(store_id)
        payment_list.append(payment)
    
    df_storePayment = pd.DataFrame({"placeID":id_list,
                                    "payment" : payment_list

                                   })
    
    #add payment

    payment_methods = [ len(i) if ( not isinstance(i, float)) and (len(i) >= 1) else 1 for i in df_storePayment['payment'].values ]

    df_payment_methods = pd.DataFrame({'payment_methods' : payment_methods})

    df_storePayment = pd.concat([df_storePayment , df_payment_methods], axis =1)

    # merge
    df_store = df_store.merge(df_storePayment, on =  'placeID', how = 'left')
    df_store['payment_methods'] = df_store['payment_methods'].fillna(1)
    print(df_store.shape)
    df_store.to_csv('%s/store_merged.csv'%config.interim_data_path, index = False)
    
    return df_store


def mergred_store_and_user():

    _rastaurantDataFactroy = rastaurantDataFactroy()
    _userDataFactroy = userDataFactroy()

    df_merged = _userDataFactroy.merge(_rastaurantDataFactroy, on = 'placeID', how = 'left')

    
    print(df_merged.shape)

    # # create new demension
    # distance = []
    # for i in df_merged.values:
    #     distance.append(utli.computeGeoDistance(i[5], i[6], i[24], i[25]))

    # df_distance = pd.DataFrame({'distance_between_user_and_store' : distance})

    # df_merged = pd.concat([df_merged, df_distance], axis =1)

    print(df_merged.shape)

    #1 filter out unemployment
    # df_merged = df_merged[df_merged['activity'] != 'unemployed']

    #2 replace synonyms for city
    city_synonyms_dict = dict()    
    city_synonyms = pd.read_csv('./src/features/city.csv')

    for i in city_synonyms.values:
        city_synonyms_dict[i[0]] = i[1]
    
 
    df_merged['store_city'] = df_merged['store_city'].apply(lambda x :utli.deal_synonyms(x, city_synonyms_dict))
    
    # add store cuision math

    tmp_cuisine_match = list()

    for match in range(df_merged['user_cusine'].shape[0]):
        if isinstance(df_merged['store_cuisin'][match], float):
            tmp_cuisine_match.append(0)
            continue
        if df_merged['user_cusine'][match] in df_merged['store_cuisin'][match]:
            tmp_cuisine_match.append(1)
        else:
            tmp_cuisine_match.append(0)
        
    df_cuisine_match = pd.DataFrame({'cuisine_match': tmp_cuisine_match})

    df_merged = pd.concat([df_merged, df_cuisine_match], axis = 1)
    
    # remove ? as nan
    df_merged = df_merged.replace('?', np.nan)

    try:
        df_merged.to_csv('%s/merged_data.csv'%config.interim_data_path, index = False)
    except Exception as e:
        print(e)
    

    return df_merged


        


        
    

    






        



    
    # df_store = df_store.merge(df_storeHours, on =  'placeID', how = 'left')
    # print(df_store.shape)


    

if __name__ == "__main__":
    print("#"*20)
    print("Start the ETL process")
    # userDataFactroy()
    # rastaurantDataFactroy()
    mergred_store_and_user()
    print("End the ETL process")