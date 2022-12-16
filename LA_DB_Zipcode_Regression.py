from sodapy import Socrata
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial as P
from numpy.polynomial.polynomial import polyval as Pval
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from shapely.geometry import Point
import geopandas as gpd
import contextily as cx
from datetime import date, datetime, timedelta
#import dateutil.parser

def getKeys(path):
    with open(path, 'r', encoding='latin-1') as f:
        keys = f.readline()
    return keys

def Return_Results_Df(query_str):
    # Return the dataframe from a socrata pull. Optional JSON query string to filter the pull
    global Token # Socrata project token
    global Usr # Socrata username
    global Pss # Socrata password
    global LA_DB # database pointer
    global Permit_DS # dataset pointer
    global pull_limit
    client = Socrata(LA_DB, Token, username=Usr, password=Pss)
    results = client.get(Permit_DS, where=query_str, limit=pull_limit)
    results_df = pd.DataFrame.from_records(results)
    return results_df

def Get_Lat_Lon_Info(LocSeries): 
    #Given a dataframe series from a socrata pull (location_1), return a numpy array with [lat,lon] form.
    LatLon_ = LocSeries.array
    LatLon_Data = np.zeros((1,2))
    for la in range(0,len(LocSeries)):
        latlon_ = LatLon_[la]
        if latlon_==latlon_:
            LatLon_Data = np.row_stack((LatLon_Data,np.array([float(latlon_['latitude']),float(latlon_['longitude'])])))
    LatLon_Data = np.delete(LatLon_Data, 0, 0)
    return LatLon_Data

def Cluster_Lats_Lons(Lats_Lons):
    global epochs_
    global minimumS
    model = DBSCAN(eps=epochs_, min_samples=minimumS, algorithm='auto').fit(Lats_Lons)
    Clu_labels = model.labels_
    Un_Clu_labels, Un_counts = np.unique(Clu_labels[Clu_labels >= 0], return_counts=True)
    core_samples_mask = np.zeros_like(Clu_labels, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True   
    n_clusters_ = len(Un_Clu_labels)
    n_noise_ = list(Clu_labels).count(-1)
    print("n_clusters: ",n_clusters_," & n_noise: ",n_noise_)

    return Clu_labels, Un_Clu_labels, core_samples_mask

def Get_First_Last_Day_OfMonth(year,month):
    # Takes input year & month (number) and returns datetime first and last day of that month
    first_date = datetime(year, month, 1)
    if month == 12:
        last_date = datetime(year, month, 31)
    else:
        last_date = datetime(year, month + 1, 1) + timedelta(days=-1) # Add a month then remove 1 day and datetime will give last day
    first_date_string = "'"+first_date.strftime("%Y-%m-%dT%H:%M:%S.000")+"'"
    last_date_string = "'"+last_date.strftime("%Y-%m-%dT%H:%M:%S.000")+"'"
    return first_date_string, last_date_string

def calc_centeroid(arr):
    # Calculates centroids from a given [lat, lon] array. Returns lon centroid, lat centroid.
    length = arr.shape[0]
    sum_0 = np.sum(arr[:, 0])
    sum_1 = np.sum(arr[:, 1])
    return sum_0/length, sum_1/length

def centeroidnp(arr):
    # Calculates centroids from a given [lat, lon] array. Returns lon centroid, lat centroid.
    length = arr.shape[0]
    sum_lat = np.sum(arr[:, 0])
    sum_lon = np.sum(arr[:, 1])
    return sum_lon/length, sum_lat/length

print("Beginning script")

# --- Socrata Access Info
Token = getKeys('SocrataToken.txt')
Usr = getKeys('SocrataUsr.txt')
Pss = getKeys('SocrataPass.txt')
GKey = getKeys('gkey.txt')
# Permit dataset from the LA Database
LA_DB = "data.lacity.org"
Permit_DS = "nbyu-2ha9"
pull_limit = 500

# --- Plot Single Zip, All Time, 8000 Limit --- #
#permit_type = "'Bldg-New'"
permit_type = "'Bldg-Alter/Repair'"
zipcode_str = "'91335'"
print("Zipcode:",zipcode_str)
yearlist = []
for y in range(2013,2023):
    yearlist.append(y)
#print("yearlist:",yearlist)
#yearlist = [2022]
monthlist = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthcount = 0
permit_count = np.zeros((1,2))
for year in yearlist:
    print("Year:",year)
    for month in monthlist:
        i = monthlist.index(month)+1
        if (year > 2021) and (i > 11):
            break
        first_day, last_day = Get_First_Last_Day_OfMonth(year,i) # Get date strings for first & last day of each month
        # Make the request to Socrata with a query string filter
        query_str = "zip_code="+zipcode_str+"AND permit_type="+permit_type+"AND issue_date BETWEEN "+ first_day +" AND "+ last_day
        results_df = Return_Results_Df(query_str)
        month_permit_count = results_df.shape[0]
        print("Number of Permits for",month,"-",results_df.shape[0])
        datestr = str(year)+"-"+str(month)
        permit_count = np.row_stack((permit_count,np.array([monthcount,month_permit_count],dtype=int)))
        monthcount+=1
permit_count = np.delete(permit_count,0,0)

# --- Create Data Set --- #
Xzeros = np.zeros((len(permit_count),1))
X = permit_count[:,0] # Months since Jan 1, 2013
# X = np.column_stack((Xzeros,X)) # Months since Jan 1, 2013 with a zero column for constants
Y = permit_count[:,1]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Prepare Features --- #
# Creates new features from the data set according to the degree: x1, x1^2, etc.
X_poly = X.reshape(-1,1)
poly_features = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly_features.fit_transform(X_poly)
print("X",X_poly)
# X_poly_test = poly_features.transform(X_test)

# --- Standardize Inputs --- #
scaler = StandardScaler()
X_std = scaler.fit_transform(X_poly)
# X_std_test = scaler.fit_transform(X)

# --- Train Model w Lasso w Auto GridsearchCV --- #
lasso_model_GCV = LassoCV(alphas=np.arange(0, 1, 0.01), cv=10)
lasso_model_GCV.fit(X_std, Y)
co_lasso_GCV = lasso_model_GCV.coef_
inter_lasso_GCV = lasso_model_GCV.intercept_
co = np.reshape(co_lasso_GCV,(len(co_lasso_GCV),1))
print("co:",co)
print("X_std Shape",X_std.shape)
Y_pred = np.dot(X_std,co) + inter_lasso_GCV
#print("Y_pred",Y_pred)

# --- Simple polyfit --- #
# t=4
# Coef_List = list(range(0,t))
# RMSE_List = list(range(0,t))
# Y_pred_List = list(range(0,t))
#
# for p in range(0,t):
#     Fullcoef, Fullextra = P.polyfit(X,Y,(p+1),full=True) # Calc coefficients from the data for power n, returns list of residuals as well
#     Coef_List[p] = Fullcoef
#     RMSE_List[p] = math.sqrt(Fullextra[0])
#     Y_pred_List[p] = Pval(X,Coef_List[p]) # Get calculated values of y for each new fit coefficients.

# --- Graph --- #
figure, axis = plt.subplots(1,1,figsize=(11,7),dpi=80)
# leg1 = ['Permit Data']
# axis[0].scatter(permit_count[:,0],permit_count[:,1])
# for s in range(0,t):
#     axis[0].plot(X,Y_pred_List[s])
#     leg1.append("Fit: "+str(Coef_List[s]))
# # every_nth = 12
# # for n, label in enumerate(axis.xaxis.get_ticklabels()):
# #     if n % every_nth != 0:
# #         label.set_visible(False)
# ptitle = zipcode_str+" "+permit_type+" Permit Counts Over Time"
# figure.suptitle(ptitle)
# #axis[0].set_xlabel("Months Since Jan 2013")
# axis[0].set_ylabel("Number Of Permits")
# axis[0].legend(leg1)
#plt.xticks(rotation = 90)
leg2 = ['Permit Data','GCV Model - Feature Eng n=3']
axis.scatter(permit_count[:,0],permit_count[:,1])
axis.plot(X,Y_pred,color='r')
axis.set_xlabel("Months Since Jan 2013")
axis.set_ylabel("Number Of Permits")
axis.legend(leg2)
axis.set_title(zipcode_str+" "+permit_type+" Permits Over Time")
ptext = "Fit Form: "+str(round(float(inter_lasso_GCV),2))+"+"+str(round(float(co[1]),2))+"*x+"+str(round(float(co[2]),2))+"*x^2"+str(round(float(co[3]),2))+"*x^3"
at = AnchoredText(ptext,loc='center left', prop=dict(size=8), frameon=True)
axis.add_artist(at)
#text(0, 70, ptext, fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))
plt.show()
