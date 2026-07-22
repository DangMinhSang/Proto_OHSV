#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype Model info.

@author: de Moura, K.
"""
import numpy as np
from sklearn.cluster import KMeans

PROTOTYPE_MODELS = ['kmeans']

class PrototypeModel:
    
    def __init__(self, name, 
                 n_clusters=None, 
                 rng = None):

        self.name = name
        self.n_clusters = n_clusters
        rng = rng or 42
        if name == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=rng)

        else:
            raise(f"Invalid PrototypeModel name: {name}. Valid names: {PROTOTYPE_MODELS}")
    
    def __str__(self):
        return f"PrototypeModel(name={self.name}, k={self.n_clusters}, model={type(self.model).__name__})"
    
    def to_string(self):
        return self.__str__()
    
    def fit(self, data):
        self.data = data
        self.model.fit(data)

    def get_k(self):
        return self.n_clusters
    
    def get_data_labels(self):
        return self.model.labels_
        
    def get_prototypes(self):
        prototypes = self.model.cluster_centers_
        return prototypes
    