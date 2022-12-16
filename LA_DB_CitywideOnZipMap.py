from sodapy import Socrata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import googlemaps

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
pull_limit = 5000
# --- DBSCAN Params
# epochs_ is the size of core circle sample
# minimumS is the number of points allowed inside circle sample before considered core

# --- Plot City Wide Clusters, 1 Month, 5000 Limit --- #
# Make the request to Socrata with a query string filter
daterangestart = "'2019-01-01T00:00:00.000'"
daterangeend = "'2019-12-31T00:00:00.000'"
permit_type = "'Bldg-Alter/Repair'"
#permit_type = "'Bldg-New'"
query_str = "permit_type="+permit_type+" AND issue_date BETWEEN "+daterangestart+" AND "+daterangeend
results_df = Return_Results_Df(query_str)
# Reformat the pandas data into numpy arrays
Lats_Lons = Get_Lat_Lon_Info(results_df.location_1)
print("LatLon_: ",Lats_Lons.shape)
# DBSCAN Parameters
epochs_ = 0.005
minimumS = 25
# Perform the DBSCAN clustering and return information needed to plot.
Clu_labels, Un_Clu_labels, core_samples_mask = Cluster_Lats_Lons(Lats_Lons)
unique_labels = set(Clu_labels)


# --- Creating City Wide Map Plot with Geographic Background --- #
figure, axis = plt.subplots(1,1,figsize=(11,7),dpi=80)
gmaps = googlemaps.Client(key=GKey)
# Graph the map with the clusters in varying color and the remaining data in black
axis.set_aspect('equal')
map_df = gpd.read_file("LA_County_ZIP_Codes.shx")
map_df.plot(ax=axis, color='white', edgecolor='black')
# Graph the points on top of it.
colors = []
Legend_List = []
for c in np.linspace(0, 1, len(unique_labels)): # Creates list of numbers spaced equally between 1st & 2nd argument, number of samples is 3rd arg.
    colors.append(plt.cm.Spectral(c))
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1] # Black used for noise.
    class_member_mask = Clu_labels == k
    data = Lats_Lons[class_member_mask & ~core_samples_mask]
    data_df = pd.DataFrame(data=data, columns=['Lat','Lon'])
    data_gdf = gpd.GeoDataFrame(data_df,crs='epsg:4269',geometry=gpd.points_from_xy(data_df.Lon, data_df.Lat))
    data_gdf = data_gdf.to_crs(epsg=3857)
    data_gdf.plot(ax=axis,marker="v",color=tuple(col),markersize=6,alpha=0.3)

    clusterdata = Lats_Lons[class_member_mask & core_samples_mask]
    # Use reverse geocoding to get the neighborhood name for the cluster centroids.
    if len(clusterdata)>0:
        centroid = calc_centeroid(clusterdata)
        rev_geocode_result = gmaps.reverse_geocode((centroid))
        print("rev_geocode_result",rev_geocode_result)
        Neighborhoods = "LAX"
        for t in range(0,len(rev_geocode_result[0]['address_components'])):
            types = rev_geocode_result[0]['address_components'][t]['types']
            for ty in types:
                if ty == 'neighborhood':
                    Neighborhoods = rev_geocode_result[0]['address_components'][t]['long_name']
        
        Zipcodes = "N/A"
        for z in range(0,len(rev_geocode_result[0]['address_components'])):
            types = rev_geocode_result[0]['address_components'][z]['types']
            for ty in types:
                if ty == 'postal_code':
                    Zipcodes = rev_geocode_result[0]['address_components'][z]['long_name']
        # if len(Zipcodes)!=5:
        #     Zipcodes = rev_geocode_result[0]['address_components'][-2]['long_name']
        Legend_List.append(Neighborhoods+", "+Zipcodes)
    clusterdata_df = pd.DataFrame(data=clusterdata, columns=['Lat','Lon'])
    clusterdata_gdf = gpd.GeoDataFrame(clusterdata_df,crs='epsg:4269',geometry=gpd.points_from_xy(clusterdata_df.Lon, clusterdata_df.Lat))
    clusterdata_gdf = clusterdata_gdf.to_crs(epsg=3857)
    clusterdata_gdf.plot(ax=axis,marker="o",color=tuple(col),markersize=14,alpha=1.0,label=(Neighborhoods+", "+Zipcodes))
axis.set_xbound(-1.31497e07, -1.32130e07)
axis.set_ybound(3.9866e06, 4.0763e06)
axis.set_title( "2019 Top "+str(len(Un_Clu_labels))+" Building "+permit_type+" Permit Hotspots")
# ptext = "# of Clusters: " +str(len(Un_Clu_labels))
# axis.text(-1.32074e07, 4.0056e06, ptext, fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))
axis.axis('off')
axis.legend(loc='best',bbox_to_anchor=(0.5, 0.4),borderpad=0.2,handletextpad=0.4,framealpha=1.0)
plt.show()