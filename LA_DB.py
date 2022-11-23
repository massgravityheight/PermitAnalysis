from sodapy import Socrata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd
import contextily as cx

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
    Clu_indices = model.core_sample_indices_
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

def Return_Clu_Results_By_Date(db_pointer, ds_code, token, usr, pss, daterangestart, daterangeend, pull_limit): 
    # Takes a str database, str dateranges, and int pull_limit and returns a dataframe result
    query_str = "issue_date BETWEEN "+ daterangestart +" AND "+ daterangeend #zip_code='90024' AND 
    client = Socrata(db_pointer, token, username=usr, password=pss)
    results = client.get(ds_code, where=query_str, limit=pull_limit)
    results_df = pd.DataFrame.from_records(results)
    print("Total Received Data Size Week1: ",results_df.shape)
    LatLon_, lats, lons = Get_Lats_Lons(results_df.location_1)
    print("LatLon_: ",LatLon_.shape)
    #whitened = whiten(LatLon_)
    Clu_lons, Clu_lats, Clu_labels, components, Un_Clu_labels, Un_counts, core_samples_mask = Cluster_Lons_Lats(LatLon_)
    print("")
    return LatLon_, Clu_lons, Clu_lats, Clu_labels, lons, lats, components, Un_Clu_labels, Un_counts, core_samples_mask

Token = getKeys('SocrataToken.txt')
Usr = getKeys('SocrataUsr.txt')
Pss = getKeys('SocrataPass.txt')

LA_DB = "data.lacity.org"
Permit_DS = "nbyu-2ha9"
daterangestart = "'2022-10-01T00:00:00.000'"
daterangeend = "'2022-10-31T00:00:00.000'"
pull_limit = 2000

# --- DBSCAN Params --- #
# City Wide, 1 Week, 2000 Limit
# epochs_ = 0.01 # Size of core circle sample
# minimumS = 10 # Number of points allowed inside circle sample before considered core
epochs_ = 0.008
minimumS = 10
# 1 Zip Code, 3 Month, 2000 Limit
# epochs_ = 0.001 
# minimumS = 10 

LatLon_, Clu_lons, Clu_lats, Clu_labels, lons, lats, components, Un_Clu_labels, Un_counts, core_samples_mask = Return_Clu_Results_By_Date(LA_DB,Permit_DS,Token,Usr,Pss,daterangestart,daterangeend,pull_limit)

# # Get Zip Code Counts
# # Count_Zips = results_df.zip_code.value_counts(dropna=False)
# # Zips_Only = Count_Zips.index.values.tolist()
# # print("Top 3: ",Zips_Only[:3])
#
# --- Graphs --- #
figure, axis = plt.subplots(1,2,figsize=(11,7),dpi=80)

# Black removed and is used for noise instead.
unique_labels = set(Clu_labels)
colors = []
for c in np.linspace(0, 1, len(unique_labels)): # Creates list of numbers spaced equally between 1st & 2nd argument, number of samples is 3rd arg.
    colors.append(plt.cm.Spectral(c))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = Clu_labels == k
    xy = LatLon_[class_member_mask & core_samples_mask]
    axis[1].plot(xy[:, 1],xy[:, 0],"o",markerfacecolor=tuple(col),markeredgecolor="k",markersize=14,alpha=0.4)
    xy = LatLon_[class_member_mask & ~core_samples_mask]
    axis[1].plot(xy[:, 1],xy[:, 0],"v",markerfacecolor=tuple(col),markeredgecolor="k",markersize=6,alpha=0.1)
ptitle = "Clustering - Epochs: "+str(epochs_)+", MinimumS: "+str(minimumS)
axis[1].set_title(ptitle)
ptext = "# of Clusters: " +str(len(Un_Clu_labels))
axis[1].text(-118.68, 33.727, ptext, fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5)) #
axis[1].axis('off')

# --- Creating a Geographic data frame --- #
# Coordinate reference system : WGS84
map_df = gpd.read_file("LA_County_ZIP_Codes.shx")
# print("Map Geo: ",map_df.geometry)
# print("Map CRS: ",map_df.crs)
LatLon_df = pd.DataFrame(data=LatLon_, columns=['Lat','Lon'])
newindex = ['Lon','Lat']
LonLat_df = LatLon_df.reindex(columns=newindex)
points_gdf = gpd.GeoDataFrame(LonLat_df,crs='epsg:4269',geometry=gpd.points_from_xy(LonLat_df.Lon, LonLat_df.Lat))
points_gdf = points_gdf.to_crs(epsg=3857)
# print("Points Head: ",points_gdf.head())
# print("Points CRS: ",points_gdf.crs)

axis[0].set_aspect('equal')
map_df.plot(ax=axis[0], color='white', edgecolor='black')
points_gdf.plot(ax=axis[0], alpha=0.3, marker='v')
axis[0].set_xbound(-1.31497e07, -1.32130e07)
axis[0].set_ybound(3.9866e06, 4.0763e06)

# --- Other Graphs --- #
# # axis[0,0].bar(Zips_Only,Count_Zips)
# # axis[0,0].set_xticklabels(Zips_Only, rotation='vertical')
# axis[0].plot(lons, lats, alpha=0.1, linestyle='none', marker='o')
axis[0].set_title("All Permits - Oct. 2022")
axis[0].axis('off')
# #axis[1,0].plot(lons2,lats2, alpha=0.1, linestyle='none', marker='o')
# #axis[1,1].plot(Clu_lons,Clu_lats, alpha=0.5, linestyle='none', marker='o')
# #axis[1,0].plot(Clu_lonsW2,Clu_latsW2, alpha=0.5, linestyle='none', marker='o')
# axis[1,0].legend(['Data','Clusters'])
plt.show()