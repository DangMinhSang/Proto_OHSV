#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for model learning, validation, and test.

@author: de Moura, K.
"""
import argparse
import os

import numpy as np
import pandas as pd
from typing import Tuple, Sequence, Optional, Literal, Union, List

import sklearn.pipeline as pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from shsv.data import (load_extracted_features,
                       _compute_dissimilarity,
                       generate_diss_test_data)

from prototype_model import PrototypeModel, PROTOTYPE_MODELS
from util import find_closest_samples, run_script, get_subset


def generate_diss_training_data(
        data: np.ndarray,
        label: np.ndarray,
        prototypes: Optional[np.ndarray],
        rng: np.random.RandomState,
        dist_type: str = "standard",
        n_gen: int = 12,
        n_writer_centroids: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Generate dissimilarity data for writer-independent learning.

    dist_type:
        - standard: random signatures from other writers
        - poscentroid: prototypes closest to one writer centroid
        - multicentroid: prototypes closest to multiple writer centroids
    """

    feat_size = data.shape[1]
    diss_data = []
    diss_target = []

    if dist_type not in ["standard", "poscentroid", "multicentroid"]:
        raise ValueError(f"Invalid dist_type: {dist_type}")

    if dist_type in ["poscentroid", "multicentroid"] and prototypes is None:
        raise ValueError(f"prototypes cannot be None when dist_type={dist_type}")

    print(f"Computing dissimilarities using {dist_type}")

    for user_id in np.unique(label):
        user_indices = np.where(label == user_id)[0]

        if len(user_indices) < n_gen:
            raise ValueError(
                f"Writer {user_id} has {len(user_indices)} genuine signatures, "
                f"but n_gen={n_gen}"
            )

        gen_idxs = rng.choice(user_indices, size=n_gen, replace=False)
        f_gen = data[gen_idxs]

        # Positive dissimilarities
        pos_diss = np.abs(f_gen[:, None] - f_gen)
        sig_idxs = np.triu_indices(n_gen, k=1)
        ddp = pos_diss[sig_idxs]

        # Same configuration as the original repository
        n_pos_s = n_gen - 1 if n_gen % 2 == 0 else n_gen
        n_neg_s = n_gen // 2

        if dist_type == "standard":
            diff_user_indices = np.where(label != user_id)[0]

            if len(diff_user_indices) < n_neg_s:
                raise ValueError("Not enough signatures from other writers")

            rf_idxs = rng.choice(diff_user_indices, size=n_neg_s, replace=False)
            f_rf = data[rf_idxs]

            neg_diss, _, _ = _compute_dissimilarity(
                f_gen[:n_pos_s],
                f_rf,
                gen_idxs[:n_pos_s],
                rf_idxs
            )

        elif dist_type == "poscentroid":
            n_neg_s = min(n_neg_s, prototypes.shape[0])

            pos_centroid = np.mean(f_gen, axis=0)
            sorted_prot, _, _ = find_closest_samples(prototypes, pos_centroid)
            selected_prototypes = sorted_prot[:n_neg_s]

            neg_diss, _, _ = _compute_dissimilarity(
                f_gen[:n_pos_s],
                selected_prototypes,
                gen_idxs[:n_pos_s],
                np.arange(len(selected_prototypes))
            )

        elif dist_type == "multicentroid":
            n_neg_s = min(n_neg_s, prototypes.shape[0])
            actual_n_centroids = min(n_writer_centroids, n_gen)

            if actual_n_centroids == 1:
                writer_centroids = np.mean(f_gen, axis=0, keepdims=True)
            else:
                writer_kmeans = KMeans(
                    n_clusters=actual_n_centroids,
                    random_state=int(rng.randint(0, 2**31 - 1)),
                    n_init=10
                )
                writer_kmeans.fit(f_gen)
                writer_centroids = writer_kmeans.cluster_centers_

            # Rank global prototypes by distance to each writer centroid
            prototype_rankings = []

            for centroid in writer_centroids:
                _, sorted_indices, _ = find_closest_samples(prototypes, centroid)
                prototype_rankings.append(sorted_indices)

            # Select prototypes in round-robin order to represent every writer mode
            selected_indices = []
            selected_set = set()
            positions = [0] * actual_n_centroids

            while len(selected_indices) < n_neg_s:
                added = False

                for centroid_idx in range(actual_n_centroids):
                    ranking = prototype_rankings[centroid_idx]

                    while positions[centroid_idx] < len(ranking):
                        prototype_idx = int(ranking[positions[centroid_idx]])
                        positions[centroid_idx] += 1

                        if prototype_idx not in selected_set:
                            selected_indices.append(prototype_idx)
                            selected_set.add(prototype_idx)
                            added = True
                            break

                    if len(selected_indices) == n_neg_s:
                        break

                if not added:
                    break

            if len(selected_indices) == 0:
                raise RuntimeError(f"Could not select prototypes for writer {user_id}")

            selected_prototypes = prototypes[selected_indices]

            neg_diss, _, _ = _compute_dissimilarity(
                f_gen[:n_pos_s],
                selected_prototypes,
                gen_idxs[:n_pos_s],
                np.arange(len(selected_prototypes))
            )

            print(
                f"Writer {user_id}: {actual_n_centroids} centroids, "
                f"{len(selected_prototypes)} selected prototypes"
            )

        ddn = neg_diss.reshape(-1, feat_size)
        diss_data.append(np.concatenate([ddp, ddn], axis=0))

        diss_target.extend([1] * ddp.shape[0])
        diss_target.extend([0] * ddn.shape[0])

    diss_x = np.concatenate(diss_data, axis=0)
    diss_y = np.asarray(diss_target, dtype=np.int64)

    print(f"Dissimilarity shape: {diss_x.shape}")
    print(f"Positive pairs: {np.sum(diss_y == 1)}")
    print(f"Negative pairs: {np.sum(diss_y == 0)}")

    return diss_x, diss_y

def train(model_choice: Literal["svm", "sgd"],
            tr_x: np.ndarray,
            tr_y: np.ndarray,
            seed: int = 42,
            perform_training: bool = True,
            svm_cache_size_mb: int = 16384,
        ) -> Union[SVC, SGDClassifier]:
    
    """
    Train a writer-independent classifier on dissimilarity data.

    Parameters
    ----------
    model_choice : {"svm", "sgd"}
        Classification model to train.
    tr_x : np.ndarray of shape (N, F)
        Training dissimilarity vectors.
    tr_y : np.ndarray of shape (N,)
        Binary labels (1 = positive pair, 0 = negative pair).
    seed : int, default=42
        Random seed for model initialization.
    perform_training : bool, default=True
        If False, return an untrained model.
    svm_cache_size_mb : int, default=16384
        Cache size for the SVM model.

    Returns
    -------
    model : sklearn.svm.SVC or sklearn.linear_model.SGDClassifier
        The fitted model (or unfitted model if perform_training=False).
    """
    
    print("--- BATCH TRAINING ---")

    
    n_neg, n_pos = np.unique(tr_y, return_counts=True)[1]
    
    skew = n_neg / float(n_pos)
    
    if 'sgd' in model_choice:
        
         model = SGDClassifier(loss='hinge', 
                               random_state=seed,
                               alpha=0.1,
                               #eta0=1,
                               eta0=0.01,
                               max_iter=2000,
                               tol=0.001)
           
    else: # model_choice == 'svm':
        model = SVC(C=1, gamma=2**-11, 
                    class_weight={1: skew},
                    cache_size=svm_cache_size_mb) 
   
    
        
    if perform_training:
        final_model = model

        final_model.fit(tr_x, tr_y)
        
        
        return final_model

    return model
        
def test(
        model: Union[SVC, SGDClassifier],
        test_x: np.ndarray,
        test_y: np.ndarray,
        test_ds: Sequence[np.ndarray],
        output_path: str,
        filename: str = "",
    ) -> None:
    
    """
    Run batch testing using a trained classifier and save predictions to disk.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier with `predict` and `decision_function`.
    test_x : np.ndarray of shape (N, F)
        Dissimilarity vectors for testing.
    test_y : np.ndarray of shape (N,)
        Ground-truth binary labels.
    test_ds : sequence of arrays
        Auxiliary test data returned by `generate_diss_test_data`,
        containing:
            - reference indices
            - query indices
            - reference user IDs
            - query user IDs
            - query types (genuine/forgery)
    output_path : str
        Directory where prediction CSV will be stored.
    filename : str, optional
        Base name used to generate the prediction file.

    Returns
    -------
    None
    """
    
    
    print("--- BATCH TEST ---")

    batch = 1000
            
    o_pred = []
    o_proba_class0 = []
    o_proba_class1 = []
    o_label = []
           
    for bi in range(0,test_x.shape[0], batch):
        #print(bi)
        ts_x = test_x[bi:bi+batch]
        ts_y = test_y[bi:bi+batch]
        
        pred = model.predict(ts_x)
        o_pred.extend(pred)
        o_proba_class0.extend(np.zeros_like(pred))
        
        decisionf = model.decision_function(ts_x) 
        
        o_proba_class1.extend(decisionf)
        o_label.extend(ts_y)


    #Create result folder
    if not os.path.exists(output_path):
        print(f"Creating folder: {output_path}")
        os.makedirs(output_path)
        
    #o_filename = "pred#"+type(model).__name__+"#"+filename.replace(".npz",".csv")
    model_info = model.named_steps['classifier'] if isinstance(model, pipeline.Pipeline) else model
    o_filename = "pred#"+type(model_info).__name__+"#"+filename.replace(".npz",".csv")
    
    
    pd.DataFrame({'pred': o_pred, 
                  'proba_class0': o_proba_class0 , 
                  'proba_class1': o_proba_class1,
                  'label': o_label,
                  'ref_idxs': test_ds[0],
                  'q_idxs': test_ds[1],
                  'ref_users': test_ds[2],
                  'q_users': test_ds[3],
                  'q_type': test_ds[4]
                  }
                 ).to_csv(os.path.join(output_path, o_filename), sep='\t')

    print(f'Saving predictions in: {o_filename}') 
    print('---------------')

def evaluate(
            f_input_path: str,
            f_metric_path: str,
            folders: Optional[List[str]] = None,
            n_refs: List[str] = ['1', '2', '3', '5', '10', '12'],
            forgeries: List[str] = ['skilled', 'random'],
        ) -> None:
    
    """
    Compute verification metrics (EER, FAR/FRR curves) for a set of prediction folders.

    This function iterates through prediction directories, constructs evaluation
    commands, and delegates metric computation to `shsv.evaluation`.

    Parameters
    ----------
    f_input_path : str
        Path containing prediction folders.
    f_metric_path : str
        Output directory for metric files.
    folders : list of str or None, default=None
        Explicit list of prediction subfolders to evaluate.
        If None, all subfolders in `f_input_path` are used.
    n_refs : list of str, default=['1','2','3','5','10','12']
        Number of reference samples to evaluate.
        Automatically adjusted for MCYT.
    forgeries : list of str, default=['skilled','random']
        Which forgery types to evaluate.

    Returns
    -------
    None
    """
    
    print("--- BATCH EVALUATION ---")
       
    folders = os.listdir(f_input_path) if folders == None else folders
    for f in folders:
        if 'mcyt' in f:
            n_refs = ['1', '2', '3', '5', '10']
        else:
            n_refs = ['1', '2', '3', '5', '10', '12']
        if f.startswith("."):
            continue
        parameters = ['batch', 
                        '--f-input-path', os.path.join(f_input_path,f),
                        '--f-output-path', f_metric_path, 
                        '--n-ref', *n_refs,
                        '--forgery', *forgeries,
                        '--thr-type', 'global', 'user',
                        '--fusions', 'max'
        ]
        run_script('shsv.evaluation', parameters)
    print("-----------------------------------") 
       
def main_validation(args):

    print(args)
    
    # Input configuration
    cluster_algo = args.cluster_algo
    dist_type   = args.dist_type
    k   = int(args.n_clusters)
    model_choice = args.model_choice
    
    f_pred_path = args.f_pred_path
    f_metric_path = args.f_metric_path
    input_features_path = args.input_feat_path

    num_gen_train = int(args.gen_for_train)
    num_gen_test = int(args.gen_for_test)
    num_gen_ref = int(args.gen_for_ref)
    
    dev_users = range(*args.dev_users)

    basename = os.path.basename(input_features_path).replace(".npz","")
    
    # Output pred folder
    pred_folder = f'{basename}_{model_choice}_{cluster_algo}_{dist_type}_g{num_gen_train}_k{k}_r{num_gen_ref}_q{num_gen_test}_val'
    output_path = os.path.join(f_pred_path, pred_folder)
    
    # Preparing data
    features, y, yforg = load_extracted_features(input_features_path)
    data_tuple = (features, y, yforg)
    
    data, label, _ = get_subset(data_tuple, dev_users, filter_gen = True)
    forg_data, forg_label, forg = get_subset(data_tuple, dev_users)
    forg_data, forg_label = forg_data[forg == 1], forg_label[forg == 1]
    
    # Seed
    rng = np.random.RandomState(args.seed)
    
    # Create folds
    n_splits = 10 if 'sgpds' in basename else 5 
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng)
    
    # Iterate through the folds
    user_ids = np.array(dev_users)
    
    for fold, (train_index, val_index) in enumerate(kf.split(user_ids)):
        
        # Split data into training and validation
        v_user_ids = user_ids[val_index]
        mask = np.isin(label,v_user_ids)
        X_train, X_val, y_train, y_val = data[~mask], data[mask], label[~mask], label[mask]
        mask_forg = np.isin(forg_label,v_user_ids)
        v_forg_data, v_forg_label = forg_data[mask_forg], forg_label[mask_forg]
        
        # Compute prototypes
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(X_train)
        prot_model = PrototypeModel(name=cluster_algo, n_clusters=k)
        prot_model.fit(data_scaled)
        prototypes = prot_model.get_prototypes()
        
        # Create diss. training data
        diss_data, diss_target =   generate_diss_training_data(X_train, 
                                           y_train, 
                                           prototypes,
                                           rng, 
                                           dist_type = dist_type, 
                                           n_gen = num_gen_train
                                           )

        # Create feat. validation data
        input_data = (
            np.concatenate([X_val, v_forg_data]), 
            np.concatenate([y_val, v_forg_label]), 
            np.concatenate([[0]*len(y_val), [1]*len(v_forg_label)])
            )
        
        # Create diss. validation data
        diss_val_x, diss_val_y, *diss_val_ds  = next(generate_diss_test_data(
                     input_data, 
                     rng,
                     n_data = 1,
                     n_ref= num_gen_ref,
                     n_query= num_gen_test,
                     include_skilled_forgery=False,
                     return_indices=True
                    ))
 
        # Train classifier
        model = train(model_choice, diss_data,  diss_target)
        
        # Output pred filename
        filename= f'{basename}_ts__n{fold}_r{num_gen_ref}_q{num_gen_test}_sk0__iuVal.csv'
        
        # Validate classifier
        test(model, diss_val_x, diss_val_y, diss_val_ds, output_path,filename=filename)
        
    #Compute EER metric          
    evaluate(f_pred_path, 
                     f_metric_path, 
                     folders = [pred_folder],
                     forgeries = ['random']
                     )
    
    print("END VALIDATION")

def main_test(args):
   
    print(args)
    
    # Input configuration
    cluster_algo = args.cluster_algo
    dist_type   = args.dist_type
    k = int(args.n_clusters)

    model_choice = args.model_choice
    
    f_pred_path = args.f_pred_path
    f_metric_path = args.f_metric_path
    input_features_path = args.input_feat_path
    
    num_gen_train = int(args.gen_for_train)
    num_gen_test = int(args.gen_for_test)
    num_gen_ref = int(args.gen_for_ref)
    
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)
    
    
    basename = os.path.basename(input_features_path).replace(".npz","")

    # Preparing data
    features, y, yforg = load_extracted_features(input_features_path)
    data_tuple = (features, y, yforg)
    exp_set = get_subset(data_tuple, exp_users)
    data, label, _ = get_subset(data_tuple, dev_users, filter_gen = True)
    
    # Initializing prototype model
    prot_model = PrototypeModel(name=cluster_algo, n_clusters=k )
    prototypes = None
        
    # Output pred folder
    pred_folder = f'{basename}_{model_choice}_{cluster_algo}_{dist_type}_g{num_gen_train}_k{k}_r{num_gen_ref}_q{num_gen_test}'
    output_path = os.path.join(f_pred_path, pred_folder)
    

    if dist_type != 'standard':
        if args.saved_prot_filename is not None:
            # Using saved prototypes
            prot_data = np.load(args.saved_prot_filename)
            prototypes = prot_data['prototypes']
            
        else:
            # Compute prototypes
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
                
            prot_model.fit(data_scaled)
            prototypes = prot_model.get_prototypes()

    # Seed            
    rng = np.random.RandomState(args.seed)
    for file_number in range(args.n_folds):
        print(file_number)
        
        # Create diss. training data
        diss_data, diss_target =  generate_diss_training_data(data, 
                                                        label, 
                                                        prototypes,
                                                        rng, 
                                                        dist_type = dist_type, 
                                                        n_gen = num_gen_train
                                                        )
        
        # Train classifier
        model = train(model_choice, diss_data,  diss_target)
        
        # Create diss. test data
        diss_test_x, diss_test_y, *diss_test_ds  = next(generate_diss_test_data(
                     exp_set, 
                     rng,
                     n_data = 1,
                     n_ref= num_gen_ref,
                     n_query= num_gen_test,
                     include_skilled_forgery=True,
                     return_indices=True
                    ))
        
        test_filename = f'{basename}_ts__n{file_number}_r{num_gen_ref}_q{num_gen_test}_sk1_iu{exp_users[0]}-{exp_users[-1]+1}.npz'
       
        # Test classifier
        test(model, diss_test_x, diss_test_y, diss_test_ds, 
                   output_path,filename=test_filename)
        
       
             
    # Compute EER metric          
    evaluate(f_pred_path, f_metric_path, folders = [pred_folder])
     
def main(args):
    
    if args.perform_validation:
        main_validation(args)
    else:
        main_test(args)

def parse_args(args_list=None):
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument('--cluster-algo', type=str, default='kmeans', choices=PROTOTYPE_MODELS)
    main_parser.add_argument('--n-clusters', type=int, default=10)
    main_parser.add_argument('--model-choice', type=str, default='sgd', choices=['svm','sgd'])
    main_parser.add_argument('--dist-type', type=str, default='poscentroid', choices=['standard', 'poscentroid', 'multicentroid'])

    main_parser.add_argument('--f-pred-path', type=str, required=True,  help='Absolute path to a folder where predictions will be saved.')
    main_parser.add_argument('--f-metric-path', type=str, required=True,  help='Absolute path to a folder where computed metrics will be saved.')
    main_parser.add_argument('--input-feat-path', type=str, required=True, help='Path to a npz file containing the fields: features, y (labels), and yforg (forgery flag).')
    main_parser.add_argument("--n-writer-centroids",type=int,default=2)

    main_parser.add_argument('--exp-users', type=int, nargs=2, default=(0, 300))
    main_parser.add_argument('--dev-users', type=int, nargs=2, default=(300, 581))
    main_parser.add_argument('--gen-for-train', type=int, default=12)
    main_parser.add_argument('--gen-for-test', type=int, default=10)
    main_parser.add_argument('--gen-for-ref', type=int, default=12)
    
    main_parser.add_argument('--perform-validation', action='store_true', default=False)

    main_parser.add_argument('--saved-prot-filename', type=str)
    
    main_parser.add_argument('--seed',  type=int,  default=42, help='Seed for reproducibility.')
    main_parser.add_argument('--n-folds', type=int, default=5, help = 'Determine the number of repetition.')
    
    main_parser.set_defaults(func=main)

    return main_parser.parse_args(args_list)


if __name__ == '__main__':
     
    args = parse_args()
    args.func(args)
