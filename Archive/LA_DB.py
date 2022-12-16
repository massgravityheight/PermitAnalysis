from sodapy import Socrata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd
import contextily as cx
from datetime import date, datetime, timedelta
#import dateutil.parser

def getKeys(path):
    with open(path, 'r', encoding='latin-1') as f:
        keys = f.readline()
    return keys

def Cluster_Lons_Lats(LatLon_Data):
    # Performs clustering and splitting of data into lat lon lists.
    global epochs_
    global minimumS
    Clu_lats = []
    Clu_lons = []

    model = DBSCAN(eps=epochs_, min_samples=minimumS, algorithm='auto').fit(LatLon_Data) # eps = typical lot size 80', min_samples = how many in area to be core
    #score_a = np.array(score[0] * np.std(LatLon_Data, 0)) # Reverting the standardization that was done. LA Transformation!
    Clu_labels = model.labels_
    Un_Clu_labels, Un_counts = np.unique(Clu_labels[Clu_labels >= 0], return_counts=True)
    #Clu_indices = model.core_sample_indices_
    components = model.components_
    core_samples_mask = np.zeros_like(Clu_labels, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True
    # Number of clusters in labels, ignoring noise if present.
    
    n_clusters_ = len(Un_Clu_labels)
    n_noise_ = list(Clu_labels).count(-1)
    print("n_clusters: ",n_clusters_," & n_noise: ",n_noise_)
    #print("Clu_labels: ",Clu_labels.shape)
    #print("Unique_Clu_labels: ",Un_Clu_labels)
    #print("Unique_counts: ",Un_counts)
    #print("components: ",components.shape)
    for i in range(0,components.shape[0]):

        Clu_lats.append([components[i][1]])
        Clu_lons.append([components[i][0]])
    
    return Clu_lons, Clu_lats, Clu_labels, components, Un_Clu_labels, Un_counts, core_samples_mask

def Get_Lats_Lons(LocSeries): 
    #Given a dataframe series, pull the lat & lon data into a 1 row by 2 col np array
    LatLon_ = LocSeries.array
    LatLon_Data = np.zeros((1,2))
    lats = []
    lons = []
    for la in range(0,len(LocSeries)):
        latlon_ = LatLon_[la]
        if latlon_==latlon_:
            lats.append(float(latlon_['latitude']))
            lons.append(float(latlon_['longitude']))
            LatLon_Data = np.row_stack((LatLon_Data,np.array([float(latlon_['latitude']),float(latlon_['longitude'])])))
    LatLon_Data = np.delete(LatLon_Data, 0, 0)
    return LatLon_Data, lats, lons

def Return_Clu_Results_By_Date(db_pointer, ds_code, token, usr, pss, daterangestart, daterangeend): 
    # Takes a str database, str dateranges, and int pull_limit and returns a dataframe result
    global pull_limit
    query_str = "issue_date BETWEEN "+ daterangestart +" AND "+ daterangeend #zip_code='90024' AND 
    client = Socrata(db_pointer, token, username=usr, password=pss)
    results = client.get(ds_code, where=query_str, limit=pull_limit)
    results_df = pd.DataFrame.from_records(results)
    
    print("Total Pulled Data Size: ",results_df.shape)
    LatLon_, lats, lons = Get_Lats_Lons(results_df.location_1)
    print("LatLon_: ",LatLon_.shape)
    #whitened = whiten(LatLon_)
    Clu_lons, Clu_lats, Clu_labels, components, Un_Clu_labels, Un_counts, core_samples_mask = Cluster_Lons_Lats(LatLon_)
    print("")
    return LatLon_, Clu_lons, Clu_lats, Clu_labels, lons, lats, components, Un_Clu_labels, Un_counts, core_samples_mask

def Return_Zip_Results_By_Date(db_pointer, ds_code, token, usr, pss, daterangestart, daterangeend): 
    # Takes a str database, str dateranges, and int pull_limit and returns a dataframe result
    global pull_limit
    query_str = "issue_date BETWEEN "+ daterangestart +" AND "+ daterangeend #zip_code='90024' AND 
    client = Socrata(db_pointer, token, username=usr, password=pss)
    results = client.get(ds_code, where=query_str, limit=pull_limit)
    results_df = pd.DataFrame.from_records(results)

    return results_df

def Return_Top_Zip_Results_By_Date(db_pointer, ds_code, token, usr, pss, daterangestart, daterangeend, zipcode): 
    # Takes a str database, str dateranges, and int pull_limit and returns a dataframe result
    global pull_limit
    query_str = "zip_code=" +str(zipcode) + " AND issue_date BETWEEN "+ daterangestart +" AND "+ daterangeend #zip_code='90024' AND 
    client = Socrata(db_pointer, token, username=usr, password=pss)
    results = client.get(ds_code, where=query_str, limit=pull_limit)
    results_df = pd.DataFrame.from_records(results)

    return results_df
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

Token = getKeys('SocrataToken.txt')
Usr = getKeys('SocrataUsr.txt')
Pss = getKeys('SocrataPass.txt')

LA_DB = "data.lacity.org"
Permit_DS = "nbyu-2ha9"
daterangestart = "'2022-09-01T00:00:00.000'"
daterangeend = "'2022-09-08T00:00:00.000'"
pull_limit = 5000

# --- DBSCAN Params --- #
# City Wide, 1 Week, 2000 Limit
# epochs_ = 0.01 # Size of core circle sample
# minimumS = 10 # Number of points allowed inside circle sample before considered core

# --- City Wide, 1 Month, 5000 Limit --- #
epochs_ = 0.01
minimumS = 15

# --- Vary City Wide, 1 Month, 5000 Limit --- #
# epoch_ListN = [0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005]
# epoch_ListStr = []
# n_cluster_Array = np.zeros((24,1))
# for e in epoch_ListN:
#     epochs_ = e
#     ne_cluster_list = np.zeros((1,1))
#     minimumS_List = []
#     for i in range(1,24):
#         minimumS = i
#         print("Epochs: ",epochs_," & MinS: ", minimumS)
#         LatLon_, Clu_lons, Clu_lats, Clu_labels, lons, lats, components, Un_Clu_labels, Un_counts, core_samples_mask = Return_Clu_Results_By_Date(LA_DB,Permit_DS,Token,Usr,Pss,daterangestart,daterangeend)
#         minimumS_List.append(minimumS)
#         ne_cluster_list = np.append(ne_cluster_list,len(Un_Clu_labels))
#     ne_cluster_list = np.reshape(ne_cluster_list,(len(ne_cluster_list),1))
#     print("ne_cluster_list: ",ne_cluster_list)
#     n_cluster_Array = np.concatenate((n_cluster_Array,ne_cluster_list), axis=1)
#     print("n_cluster_Array",n_cluster_Array)
#     epoch_ListStr.append(str(epochs_))
# n_cluster_Array = np.delete(n_cluster_Array, 0, 0)
# n_cluster_Array = np.delete(n_cluster_Array, 0, 1)
# print("Epoch_ListN Len: ",len(epoch_ListN))
# print("epoch_ListStr Len: ",len(epoch_ListStr))
# print("n_cluster_Array Shape: ", n_cluster_Array.shape)

# --- 1 Zip Code, 3 Month, 2000 Limit --- #
# epochs_ = 0.001 
# minimumS = 10 

# --- Plotting Lon/Lat Data & Cluster Function --- #
# LatLon_, Clu_lons, Clu_lats, Clu_labels, lons, lats, components, Un_Clu_labels, Un_counts, core_samples_mask = Return_Clu_Results_By_Date(LA_DB,Permit_DS,Token,Usr,Pss,daterangestart,daterangeend)

# --- Plotting Zip Data --- #
# Get top 3 permitting zip codes for each month
    # Get date strings for first & last day of each month
yearlist = [2018, 2019, 2020, 2021, 2022]
monthlist = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# ziplist = []
# for year in yearlist:
#     for month in monthlist:
#         i = monthlist.index(month)+1
#         if (year > 2021) and (i > 10):
#             break
#         first_day, last_day = Get_First_Last_Day_OfMonth(year,i)
#         results_df = Return_Zip_Results_By_Date(LA_DB,Permit_DS,Token,Usr,Pss,first_day,last_day)
#         # Get top 3 zips and their counts from each month's data
#         Count_Zips = results_df.zip_code.value_counts(dropna=False)
#         Top3Zips = Count_Zips[:3]
#         Zips_Only = Top3Zips.index.values.tolist()
#         print("In year ",year,"the month of ",month," had these 3 top permitting zip codes: ",Zips_Only," with these respective occurrences: ",Top3Zips.values.tolist())
#         for z in Zips_Only:
#             if z in ziplist:
#                 pass
#             else:
#                 ziplist.append(z)
# print("ziplist: ",ziplist)

    # Pull data again but only ask for the top zip codes from the previous block of code date ranges.
ziplist = ['90045', '90049', '90019', '91331', '91326', '91342', '91401', '90272', '90066', '91304', '90039', '91367', '90064', '90025', '90042', '90018', '90016', '90034', '90024', '90026', '91606', '90036', '91344', '90033', '91402', '90027', '90028']
Top_Zip__Array = np.zeros((len(monthlist)*len(yearlist),1))
c = 1
for z in ziplist:
    specific_zip_list = np.zeros((1,1))
    datelist = []
    for year in yearlist:
        for month in monthlist:
            datelist.append(month + " " + str(year))
            i = monthlist.index(month)+1
            if (year > 2021) and (i > 10): # If loop heads into current month, stop and populate remaining array with zeros.
                specific_zip_list = np.append(specific_zip_list,0)
                continue
            first_day, last_day = Get_First_Last_Day_OfMonth(year,i)
            top_results_df = Return_Top_Zip_Results_By_Date(LA_DB,Permit_DS,Token,Usr,Pss,first_day,last_day,z)
            specific_zip_list = np.append(specific_zip_list,len(top_results_df))
    specific_zip_list = np.reshape(specific_zip_list,(len(specific_zip_list),1))
    specific_zip_list = np.delete(specific_zip_list, 0, 0)
    print("specific_zip_list: ",specific_zip_list)
    Top_Zip__Array = np.concatenate((Top_Zip__Array,specific_zip_list), axis=1)
    print(c,"of",len(ziplist),"complete.")
    c=+1
Top_Zip__Array = np.delete(Top_Zip__Array, 0, 1)
print("Top_Zip__Array: ",Top_Zip__Array)
# Convert Issue Date Floating Timestamps to datetime format
# for row in range(0,len(results_df.issue_date)):
#     results_df.issue_date[row] = dateutil.parser.isoparse(results_df.issue_date[row])

# Month_df = results_df.groupby(pd.Grouper(key='issue_date', axis=0, freq='M'))
# print("Month_df: ",Month_df)
#results_df.plot(x ='issue_date', y='zip_code', kind='line')

# --- Graphs --- #
figure, axis = plt.subplots(1,1,figsize=(11,7),dpi=80)

# --- Parameter Selection Graph --- #
# axis.plot(minimumS_List,n_cluster_Array)
# axis.set_title("Clusters Found with varying MinS")
# axis.set_xlabel("MinimumS")
# axis.set_ylabel("Number of Clusters")
# axis.legend(epoch_ListStr)

# --- Creating a Map Plot with Geographic Background --- #
# map_df = gpd.read_file("LA_County_ZIP_Codes.shx")
# LatLon_df = pd.DataFrame(data=LatLon_, columns=['Lat','Lon'])
# newindex = ['Lon','Lat']
# LonLat_df = LatLon_df.reindex(columns=newindex)
# points_gdf = gpd.GeoDataFrame(LonLat_df,crs='epsg:4269',geometry=gpd.points_from_xy(LonLat_df.Lon, LonLat_df.Lat))
# points_gdf = points_gdf.to_crs(epsg=3857)
# axis[0].set_aspect('equal')
# map_df.plot(ax=axis[0], color='white', edgecolor='black')
# points_gdf.plot(ax=axis[0], alpha=0.3, marker='v')
# axis[0].set_xbound(-1.31497e07, -1.32130e07)
# axis[0].set_ybound(3.9866e06, 4.0763e06)
# axis[0].set_title("All Permits - Sept. 2022")
# axis[0].axis('off')

# --- Creating the Cluster Display --- #
# unique_labels = set(Clu_labels)
# colors = []
# for c in np.linspace(0, 1, len(unique_labels)): # Creates list of numbers spaced equally between 1st & 2nd argument, number of samples is 3rd arg.
#     colors.append(plt.cm.Spectral(c))
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#     class_member_mask = Clu_labels == k
#     xy = LatLon_[class_member_mask & core_samples_mask]
#     axis.plot(xy[:, 1],xy[:, 0],"o",markerfacecolor=tuple(col),markeredgecolor="k",markersize=14,alpha=0.4)
#     xy = LatLon_[class_member_mask & ~core_samples_mask]
#     axis.plot(xy[:, 1],xy[:, 0],"v",markerfacecolor=tuple(col),markeredgecolor="k",markersize=6,alpha=0.1) # Black removed and is used for noise instead.
# ptitle = "Clustering - Epochs: "+str(epochs_)+", MinimumS: "+str(minimumS)
# axis.set_title(ptitle)
# ptext = "# of Clusters: " +str(len(Un_Clu_labels))
# axis.text(-118.68, 33.727, ptext, fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5)) #
# axis.axis('off')

# --- Graphing Top Zips Over Time --- #
axis.plot(datelist,Top_Zip__Array)
axis.set_title("Top Permitting Zip Code Performance")
axis.set_xlabel("Time")
axis.set_ylabel("Number of Permits")
axis.legend(ziplist)

plt.show()
