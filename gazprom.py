#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
# для карты банкоматов
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from collections import Counter

from sklearn.cluster import KMeans

from cairo import ImageSurface, FORMAT_ARGB32, Context


file_gazprom_csv = "/home/vitt/Документы/data/gazprom/Moskow.csv"

file_opendata_csv = "/home/vitt/Документы/data/torgovl_stat.csv"

def image_bank(file_gazprom,file_opendata):
   dat = pd.read_csv(file_gazprom, sep='\t')

   dat.head()

   print(dat["Широта"].min())
   print(dat["Широта"].max())
   print(dat["\Долгота"].min())
   print(dat["\Долгота"].max())
   dat.info()
   dat = dat[dat["Область"] == "Москва"]
   dat["lat"] = dat["Широта"]
   dat["long"] = dat["\Долгота"]
   dat = dat.drop(["\Долгота", "Широта"], axis=1)

   dat.head()

   open_data = pd.read_csv(file_opendata)
   types = []
   lats = []
   longs = []

   for line in open_data.geoData:
      typ = line.split(", ")[0].split('=')[1]
      lat = line.split(", ")[1].split('[')[1]
      long = line.split(", ")[2].split(']')[0]
      types.append(typ)
      lats.append(lat)
      longs.append(long)

   open_df = pd.DataFrame({'types':types,
                       'lat':lats,
                       'long':longs})

   open_df = open_df.astype({'lat': 'float64', 'long': 'float64'})

   print(open_df.lat.min())
   print(open_df.lat.max())
   print(open_df.long.min())
   print(open_df.long.max())


   open_df["lat_int"] = np.round(open_df["lat"],2)
   open_df["long_int"] = np.round(open_df["long"],2)


   open_df.head()

# обучение усреднением
   kmeans = KMeans(n_clusters=253)
   kmeans.fit(open_df[["lat", "long"]])

   y_means = kmeans.predict(open_df[["lat", "long"]])

   bankomats = kmeans.cluster_centers_

   plt.scatter(open_df.lat, open_df.long, c='green', s=5)
   plt.scatter(bankomats[:, 0], bankomats[:, 1], c='cyan', s=10)
   plt.scatter(dat.long, dat.lat, c='red', s=10)


def data_process():
  dat = pd.read_csv(file_gazprom_csv, sep='\t')
  dat = dat[dat["Область"] == "Москва"]
  dat["lat"] = dat["Широта"]
  dat["long"] = dat["\Долгота"]
  dat = dat.drop(["\Долгота", "Широта"], axis=1)

  open_data = pd.read_csv(file_opendata_csv)
  types = []
  lats = []
  longs = []

  for line in open_data.geoData:
    typ = line.split(", ")[0].split('=')[1]
    lat = line.split(", ")[1].split('[')[1]
    long = line.split(", ")[2].split(']')[0]
    types.append(typ)
    lats.append(lat)
    longs.append(long)

  open_df = pd.DataFrame({'types': types,
                          'lat': lats,
                          'long': longs})

  open_df = open_df.astype({'lat': 'float64', 'long': 'float64'})
  print(open_df.head())

  x = np.array([open_df["lat"], open_df["long"]])
  x = np.transpose(x)

  centers, labels = find_clusters(x, 100)
  print(centers)


def find_clusters(X, n_clusters, rseed=3, max_iters=50, weight_koef=0.000002):
  rng = np.random.RandomState(rseed)
  i = rng.permutation(X.shape[0])[:n_clusters]
  centers = X[i]
  # print(X[i])

  for iter in range(max_iters):
    # print(centers)
    labels = pairwise_distances_argmin(X, centers, metric='manhattan')
    # weights = pairwise_distances(X, centers, metric='manhattan')
    elems_count = Counter(labels)
    lengths = []
    for x_iter in range(X.shape[0]):
        weights = []
        for center_id in range(len(centers)):
            weight = abs(X[x_iter, 0] - centers[center_id][0]) + abs(X[x_iter, 1] - centers[center_id][1])
            # Поправка на очереди у банкомата
            weight += weight_koef * elems_count[center_id]
            weights.append(weight)
        lengths.append(weights)
    labels_res = []
    for x in lengths:
        labels_res.append(np.argmin(x))
    labels = np.array(labels_res)

    new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
    length = len(new_centers[np.isnan(new_centers)]) // 2
    # lat_rand = np.array([X[:, 0].min()]*length) + (X[:, 0].max() - X[:, 0].min()) * np.random.random(length)
    # long_rand = np.array([X[:, 1].min()]*length) + (X[:, 1].max() - X[:, 1].min()) * np.random.random(length)
    # arr = np.transpose(np.array([lat_rand, long_rand]))
    i = rng.permutation(X.shape[0])[:length]
    new_centers[np.isnan(new_centers[:, 0])] = X[i]

    if np.all(centers == new_centers):
      break

    centers = new_centers
  return centers, labels


# if __name__ == "__main__":
#     image_bank(file_gazprom_csv, file_opendata_csv)

data_process()
