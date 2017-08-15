#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:47:38 2017

@author: kushaankumar
"""
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn import metrics


directory = "Dataset"

print("test")
subfolders = [os.path.relpath(f, directory) for f in os.scandir(directory) if f.is_dir() ]    
pre = dict.fromkeys(subfolders)
post = dict.fromkeys(subfolders)

for data in subfolders:
    for root,dirs,files in os.walk(directory+"/"+data):
        for file in files:
            subdirectory = os.path.relpath(root,directory)
            
            if pre[subdirectory] is None:
                pre[subdirectory] = pd.read_csv(os.path.abspath(directory+"/"+subdirectory+"/"+file))
            else:
                post[subdirectory] = pd.read_csv(os.path.abspath(directory+"/"+subdirectory+"/"+file))
                

pre_intensities = {}
post_intensities = {}
for key in pre:
    pre_df = pre[key]
    pre_df.drop(pre_df[pre_df.override != 7].index, inplace=True)
    pre_intensities[key] = pre[key][['X','Y','Z']]

for key in post:  
    post_df = post[key]
    post_df.drop(post_df[post_df.override != 7].index, inplace=True)
    post_intensities[key] = post[key][['X','Y','Z']]

#==============================================================================
# Rot_l1: 2016-01-16_A
# Rot_l2: 2017-03-16_A
# Rot_l3: 2017-03-17_B
# Rot_nl1: 2017-02-02_A
# Rot_nl2: 2017-03-29_C
# Rot_nl3: 2017-04-12_A
#==============================================================================


Rot_l1 = np.array([[0.997996, -0.031025, 0.055144, -12.004886],
                 [.030149, 0.999407, 0.016648, -66.594376],
                 [0.055628, -0.014953, 0.998340, 57.223022],
                 [0.000000, 0.000000, 0.000000, 1.000000]])

Rot_l2 =np.array([[0.995406, -0.017769, 0.094081, 17.602526], 
                  [0.017286, 0.999833, 0.005950, -6.332537], 
                  [-0.094171, -0.004297, 0.995547, 62.415340], 
                  [0.000000, 0.000000, 0.000000, 1.000000]])

Rot_l3 = np.array([[0.991936, -0.126055, -0.013188, 85.725159], 
                   [0.126061, 0.992022, -0.000387, -70.314095], 
                   [0.013132, -0.001279, 0.999913, 4.820217], 
                   [0.000000, 0.000000, 0.000000, 1.000000]])


Rot_nl1 = np.array([[0.998969, -0.032752, -0.031452, 32.868774], 
                    [0.032451, 0.999423, -0.010008, -17.975409], 
                    [0.031761, 0.008977, 0.999455, -23.138821], 
                    [0.000000, 0.000000, 0.000000, 1.000000]])

Rot_nl2 = np.array([[0.995969, -0.089516, -0.005645, 71.939552], 
                    [0.089322, 0.995598, -0.028402, -44.920918], 
                    [0.008162, 0.027783, 0.999581, -13.377362], 
                    [0.000000, 0.000000, 0.000000, 1.000000]])

Rot_nl3 = np.array([[0.995535, 0.094183, 0.006272, -60.612293], 
                    [-0.094357, 0.994777, 0.038937, 43.265373], 
                    [-0.002572, -0.039355, 0.999222, 17.586245], 
                    [0.000000, 0.000000, 0.000000, 1.000000]])
    
rotation_matrices = {"2016-01-16_A": Rot_l1, "2017-03-16_A": Rot_l2, "2017-03-17_B": Rot_l3, "2017-02-02_A": Rot_nl1, "2017-03-29_C": Rot_nl2, "2017-04-12_A": Rot_nl3} 
print("checkpoint")
#==============================================================================
# for key,value in post_intensities.items():
#     rotation_matrix = rotation_matrices[key]
#     #print(value.iloc[0])    
#     for i in value.index:
#         temp = np.append(np.asarray(value.ix[i]).tolist(), [1])
#         temp = [int(i) for i in temp]
#         to_rotate = np.dot(rotation_matrix, temp)
#         value.ix[i] = to_rotate[:-1]
#==============================================================================

print("checkpoint1")    

    #print(value.iloc[0])
#==============================================================================
# print("checkpoint")
# kmeans = KMeans(n_clusters=5, random_state=0).fit(pre_intensities["2016-01-16_A"].as_matrix())
# print(kmeans.labels_)
# centroids = kmeans.cluster_centers_
# print(centroids)
# print('Silhouette Coefficient: %0.3f'
#       % metrics.silhouette_score(pre_intensities["2016-01-16_A"].as_matrix(), kmeans.labels_))
# #==============================================================================
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# #==============================================================================
# x = np.array(pre_intensities["2016-01-16_A"]['X'])
# x = [int(i) for i in x]
# y = np.array(pre_intensities["2016-01-16_A"]['Y'])
# y = [int(i) for i in y]
# z = np.array(pre_intensities["2016-01-16_A"]['Z'])
# z = [int(i) for i in z]
# 
# #==============================================================================
# # ax.scatter(x,y,z, marker="s", c=kmeans.labels_, s=40, cmap="RdBu")
# #==============================================================================
# 
# fig=p.figure()
# ax = p3.Axes3D(fig)
# # scatter3D requires a 1D array for x, y, and z
# # ravel() converts the 100x100 array into a 1x10000 array
# ax.scatter3D(x,y,z, c=kmeans.labels_)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# p.show()
# 
# kmeans = KMeans(n_clusters=5, random_state=0).fit(post_intensities["2016-01-16_A"].as_matrix())
# print(kmeans.labels_)
# centroids = kmeans.cluster_centers_
# print(centroids)
# 
# print('Silhouette Coefficient: %0.3f'
#       % metrics.silhouette_score(post_intensities["2016-01-16_A"].as_matrix(), kmeans.labels_))
#==============================================================================
'''
    plots the clusters given the dataframe and the labels obtained from running k means
'''
def plot_clusters(dataframe, labels, month, state):
    x = np.array(dataframe['X'], dtype=np.float32)
    y = np.array(dataframe['Y'], dtype=np.float32)
    z = np.array(dataframe['Z'],dtype=np.float32)
   
    fig=p.figure()
    ax = p3.Axes3D(fig)
    # scatter3D requires a 1D array for x, y, and z
    # ravel() converts the 100x100 array into a 1x10000 array
    ax.scatter3D(x,y,z, c=labels)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(state+" "+month)
    p.show(state+" "+month)
    
'''
    runs k means on a given dataframe from the biology data set
    returns the labels and the centroids 
'''    
def run_kmeans(dataframe, num_clusters, month, state):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(dataframe.as_matrix())
    #print(kmeans.labels_)
    centroids = kmeans.cluster_centers_
    #print(centroids)
    print(state+ " " + month+":"+'\n')
    print('Silhouette Coefficient: %0.3f'
          % metrics.silhouette_score(dataframe.as_matrix(), kmeans.labels_))
    return kmeans.labels_, centroids


#run pre clusters
for month, df in pre_intensities.items():
    labels, centroids = run_kmeans(df, 5, month, "pre")
    plot_clusters(df, labels, month, "pre")
    print("checkpoint2")    


#run post clusters
for month, df in post_intensities.items():
    labels, centroids = run_kmeans(df, 5, month, "post")
    plot_clusters(df, labels, month, "post") 
    print("checkpoint3")    




        