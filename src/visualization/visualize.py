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

if 1:
    user_non_numeric = ['smoker', 'drink_level', 'dress_preference','ambience', 'transport', 'marital_status', 
                        'hijos',
                        'interest', 'personality', 'religion', 'activity','color', 'budget',
                        ]

    store_non_numeric = ['store_city','store_alcohol', 
                    'store_smoking_area', 'store_dress_code', 'store_accessibility', 
                    'store_price', 'store_Rambience', 'store_area', 'store_other_services', 'park']


    # for var_temp_user in user_non_numeric:
    #     for var_temp_store in store_non_numeric:
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         g = sns.barplot(x = var_temp_user, y ='rating', hue = var_temp_store, data = df_EDA)
    #         avg = df_EDA['rating'].mean()
    #         plt.xlabel(var_temp_user)
    #         plt.ylabel('rating')
    #         plt.title('{1} in different {0} by different {2}'.format(var_temp_user, 'rating',var_temp_store))
    #         # g.legend(loc='upper left', frameon=False)
    #         plt.legend(bbox_to_anchor=(0.95, 1), loc="upper left")
    #         plt.axhline(avg, color='r', linestyle='dashed', linewidth=2) # 繪製平均線
    #         plt.savefig('{}/rating_by_{}_and_{}_barplot.jpg'.format(config.fig_report_path, var_temp_user, var_temp_store)
    #                     ,pad = 0.5)
    #         plt.close()
        
    # for var_temp_user in user_non_numeric:
    #     for var_temp_store in store_non_numeric:
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         g = sns.barplot(x = var_temp_user, y ='food_rating', hue = var_temp_store, data = df_EDA)
    #         avg = df_EDA['food_rating'].mean()
    #         plt.xlabel(var_temp_user)
    #         plt.ylabel('food_rating')
    #         plt.title('{1} in different {0} by different {2}'.format(var_temp_user, 'food_rating',var_temp_store))
    #         # g.legend(loc='upper left', frameon=False)
    #         plt.legend(bbox_to_anchor=(0.95, 1), loc="upper left")
    #         plt.axhline(avg, color='r', linestyle='dashed', linewidth=1) # 繪製平均線 
    #         plt.savefig('{}/food_rating_by_{}_and_{}_barplot.jpg'.format(config.fig_report_path, var_temp_user, var_temp_store)
    #                     ,pad = 0.5)
    #         plt.close()
    
    # for var_temp_user in user_non_numeric:
    #     for var_temp_store in store_non_numeric:
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         g = sns.violinplot(x = var_temp_user, y ='food_rating', hue = var_temp_store, data = df_EDA)
    #         avg = df_EDA['food_rating'].mean()
    #         plt.xlabel(var_temp_user)
    #         plt.ylabel('food_rating')
    #         plt.title('{1} in different {0} by different {2}'.format(var_temp_user, 'food_rating',var_temp_store))
    #         g.legend(loc='upper left', frameon=False)
    #         plt.legend(bbox_to_anchor=(0.95, 1), loc="upper left")
    #         # plt.axhline(avg, color='r', linestyle='dashed', linewidth=1) # 繪製平均線 
    #         plt.savefig('{}/food_rating_by_{}_and_{}_barplot.jpg'.format(config.fig_report_path, var_temp_user, var_temp_store)
    #                     ,pad = 0.5)
    #         plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='food_rating', y='service_rating', hue = 'religion', data=df_EDA ,ax=ax) # ax = ax
plt.show()





co = df_EDA.corr()
mask = np.zeros_like(co) # minor mask to filter out a half of vars
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(co, mask=mask, vmax=0.5, square=True, annot= True, fmt = '.2f',linewidths = 0.2)
plt.savefig('%s/dfcorr.jpg'%(config.fig_report_path)
            ,pad = 0.5)
plt.show()


# sns.regplot(x= 'weight', y ='rating',data = df_EDA)
# plt.show()

# sns.regplot(x= 'height', y ='rating',data = df_EDA)
# plt.show()




    



        
    




# corr chart


# kernel chat


# parallel chart

