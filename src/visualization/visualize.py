import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set(style="whitegrid")
sys.path.append(os.path.abspath("."))
import config
from src.features import fe
from src.utli import utli

# read data
df_EDA = fe.mergred_store_and_user()


# check missing rate
dataReport = utli.missing_data(df_EDA)


"""
****user
userID	placeID	rating	food_rating	service_rating	latitude_x	longitude_x	
smoker ,drink_level	,dress_preference ,ambience	,transport	,marital_status	,hijos	,birth_year	
,interest ,personality	,religion	,activity	,color	,budget

## numeric
weight	height	Upayment	

****store
latitude_y	longitude_y	store_the_geom_meter store_address 

store_city, store_state, store_country, store_alcohol, store_smoking_area, store_dress_code	
store_accessibility, store_price, store_Rambience, store_area, store_other_services, park	

## numeric
cuisin	fri_hours	mon_hours	
sat_hours	sun_hours	thu_hours	tue_hours	wed_hours	park	payment	
distance_between_user_and_store
"""

# start to plot 

user_non_numeric = ['smoker', 'drink_level', 'dress_preference','ambience', 'transport', 'marital_status', 
                    'hijos',
                    'interest', 'personality', 'religion', 'activity','color', 'budget',
                     ]

store_non_numeric = ['store_city','store_alcohol', 
                   'store_smoking_area', 'store_dress_code', 'store_accessibility', 
                   'store_price', 'store_Rambience', 'store_area', 'store_other_services', 'park']


for var_temp_user in user_non_numeric:
    for var_temp_store in store_non_numeric:
        fig, ax = plt.subplots(figsize=(10, 8))
        g = sns.barplot(x = var_temp_user, y ='rating', hue = var_temp_store, data = df_EDA)
        plt.xlabel(var_temp_user)
        plt.ylabel('rating')
        plt.title('different %s %s by different %s'%(var_temp_user, 'rating',var_temp_store ))
        # g.legend(loc='upper left', frameon=False)
        plt.legend(bbox_to_anchor=(0.98, 1), loc="upper left")
        plt.savefig('%s/%s_and_rating_by_%s_barplot.jpg'%(config.fig_report_path, var_temp_user, var_temp_store)
                    ,pad = 0.5)
        plt.close()

# sns.regplot(x= 'weight', y ='rating',data = df_EDA)
# plt.show()

# sns.regplot(x= 'height', y ='rating',data = df_EDA)
# plt.show()
# corr chart


# kernel chat


# parallel chart




