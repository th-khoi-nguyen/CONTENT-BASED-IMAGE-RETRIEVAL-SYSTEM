import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans

# since I did this project in GG Colab, I need to import "cv2_imshow" function. You can skip this step if it's not necessary
from google.colab.patches import cv2_imshow

# take all images and convert them to grayscale. 
# return a dictionary that holds all images category by category. 
def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):                                                                                                                                                     
            img = cv2.imread(path + "/" + cat,0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images

images = load_images_from_folder('path to your dataset')  # take all images category by category 

def sift_features(images):
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in images.items():
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img,None)     
            descriptor_list.extend(des)
            features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]

sifts = sift_features(images) 
# take the descriptor list which is unordered one
descriptor_list = sifts[0] 
# take the sift features that is seperated class by class for train data
all_bovw_feature = sifts[1] 

# k-means clustering takes 2 parameter which are number of cluster(k) and descriptors list(unordered 1d array)
# return an array that holds central points.
def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words
    
# take the central points which is visual words    
visual_words = kmeans(100, descriptor_list)  # higher K normally equals to more computational cost but better score

from scipy.spatial import distance

# find the closest visual words (index) to the feature by calculating euclidean distance
def find_index(feature, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if i == 0:
            count = distance.euclidean(feature, center[i])
        else:
            dist = distance.euclidean(feature, center[i])
            if dist < count:
                ind = i
                count = dist
    return ind
  
def image_class(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature
    
# create histograms for train data    
bovw_train = image_class(all_bovw_feature, visual_words) 

# create the histogram for the querry image
def query_histogram(path, centers): 
  img = cv2.imread(path)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  sift = cv2.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(gray,None) 

  histogram = np.zeros(len(centers))
  for feature in des:
    ind = find_index(feature, centers)
    histogram[ind] +=1
  
  histogram = histogram.tolist()

  return histogram

query = query_histogram('path to your image', visual_words)

# k is the number of targets we want to find
def shortest_distance(query, k): 
  dists = []
  for key, value in bovw_train.items():
    for feature in value:
      feat = feature.tolist()
      dist = distance.euclidean(query, feat)
      dists.append(dist)
  
  k_cbir = np.argsort(dists)[:k] #return the index of shortest distances
  return k_cbir

distance = shortest_distance(query, 10) # here I try to find 10 targets

def image_retrieval(distance, number_of_targets):
  folders = ['path to your dataset'] 
  all = []
  for i in folders: # here I have multiple folders
    for item in os.listdir(i):
      img = cv2.imread(os.path.join(i,item))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      all.append(img)

  targets = []
  for i in range(number_of_targets):
    targets.append(all[distance[i]])

  return targets

target_list = image_retrieval(distance, 10)

# display the targets
def display(target_list, query_path):
  qr = cv2.imread(query_path)
  qr = cv2.cvtColor(qr, cv2.COLOR_BGR2RGB)
  plt.figure(figsize = (10,20))
  plt.subplot(4,4,1)
  plt.imshow(qr)
  plt.title('Query')
  plt.axis('off')


  i = 0
  for image in target_list:
    plt.figure(figsize = (10,20))
    plt.subplot(4,4,i+2)
    plt.imshow(target_list[i])
    plt.title('targets')
    plt.axis('off')
    i = i+1 
