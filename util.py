#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions.

@author: de Moura, K.
"""
import subprocess
import sys
import numpy as np
from typing import Tuple, Union 
from scipy.spatial.distance import cdist

def find_closest_samples(samples, sample_A, metric='euclidean'):
    # Calculate distances between sample_A and all samples
    distances = cdist([sample_A], samples, metric=metric).flatten()
    
    # Get indices that would sort the distances
    sorted_indices = np.argsort(distances)
    
    # Sort the samples and distances
    sorted_samples = samples[sorted_indices]
    sorted_distances = distances[sorted_indices]
    
    return sorted_samples, sorted_indices, sorted_distances

def run_script(script_name, args):
    command = [sys.executable, '-m', script_name] + args
    subprocess.run(command)

def get_subset(data: Tuple[np.ndarray, ...],
               subset: Union[list, range],
               y_idx: int = 1,
               filter_gen = False) -> Tuple[np.ndarray, ...]:
    """ Gets a data for a subset of users (the second array in data)

    Parameters
    ----------
    data: Tuple (x, y, ...)
        The dataset
    subset: list
        The list of users to include
    y_idx: int
        The index in data that refers to the users (usually index=1)
    filter_gen : bool, optional (default=False)
        If True, the function filters the dataset to keep only genuine samples
        (i.e., samples where the forgery-label array equals 0). In this case,
        `data` is expected to be a tuple (x, y, yforg).

    Returns
    -------
    Tuple (x, y , ...)
        The dataset containing only data from users in the subset

    """
   
    if filter_gen:
        x, y, yforg = data
        
        x, y, yforg = (x[yforg == 0], y[yforg == 0], yforg[yforg==0])
    
        data = (x, y, yforg)
    
    to_include = np.isin(data[y_idx], subset)

    return tuple(d[to_include] for d in data)
