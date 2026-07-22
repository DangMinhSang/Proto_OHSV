#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration reproducibility.

@author: de Moura, K.
"""
import os

from main_process import parse_args

def get_param(f_pred_path, 
              f_metric_path,
              config,
              ds,
              model_choice, 
              dist_type, 
              k):
    param = [
        '--cluster-algo', 'kmeans', 
        '--n-clusters',    str(k), 
        '--model-choice', model_choice,
        '--dist-type',    dist_type,
        '--f-pred-path',  f_pred_path,
        '--f-metric-path',f_metric_path,
        '--input-feat-path',config[ds]['input-features-path'],
        
        '--exp-users',    config[ds]['exp-users'][0], config[ds]['exp-users'][1],
        '--dev-users',    config[ds]['dev-users'][0], config[ds]['dev-users'][1],
        '--gen-for-train',config[ds]['gen-for-train'],
        '--gen-for-test', config[ds]['gen-for-test'],
        '--gen-for-ref',  config[ds]['gen-for-ref']
        ]
    return param
    
def reproduce_test(f_pred_path, f_metric_path, config):
    
    for ds in config.keys():
        for model_choice in ['svm', 'sgd']:
            for dist_type in ['poscentroid', 'standard']:
                
                k = 0 if dist_type == 'standard' else config[ds]['best-k'][model_choice]
                
                param = get_param(f_pred_path, 
                              f_metric_path,
                              config,
                              ds,
                              model_choice, 
                              dist_type, 
                              k)
                args = parse_args(param)
                args.func(args)
                

def reproduce_validation(f_pred_path, f_metric_path, config):
    
    dist_type = 'poscentroid'
    for ds in config.keys():    
        for model_choice in ['sgd', 'svm']:#['svm', 'sgd']:
            for k in config[ds]['k-range']:
                print(f"n_clusters = {k}")
                param = get_param(f_pred_path, 
                              f_metric_path,
                              config,
                              ds,
                              model_choice, 
                              dist_type, 
                              k)
                args = parse_args(param + ['--perform-validation'])
                args.func(args)

def create_folders(f_main_path):
    
    f_pred_test_path = os.path.join(f_main_path,'pred_test')
    f_metric_test_path = os.path.join(f_main_path,'metric_test')
    
    f_pred_val_path = os.path.join(f_main_path,'pred_val')
    f_metric_val_path = os.path.join(f_main_path,'metric_val')
    
    if not os.path.exists(f_pred_test_path):
        print(f"Creating prediction folder: {f_pred_test_path}")
        os.makedirs(f_pred_test_path)
    
    if not os.path.exists(f_metric_test_path):
        print(f"Creating metrics folder: {f_metric_test_path}")
        os.makedirs(f_metric_test_path)
        
    if not os.path.exists(f_pred_val_path):
        print(f"Creating prediction folder: {f_pred_val_path}")
        os.makedirs(f_pred_val_path)
    
    if not os.path.exists(f_metric_val_path):
        print(f"Creating metrics folder: {f_metric_val_path}")
        os.makedirs(f_metric_val_path)
    
    return f_pred_test_path, f_metric_test_path, f_pred_val_path, f_metric_val_path

        
if __name__ == '__main__':
    f_main_path = os.path.expanduser("~/proto_hsv_results")
    
    f_pred_test_path, f_metric_test_path, f_pred_val_path, f_metric_val_path = create_folders(f_main_path)

    #GPDS-S
    gpdss_npz_path = "path/to/gpdss_features.npz"
    # MCYT
    mcyt_npz_path  = "path/to/mcyt_features.npz"
    # CEDAR
    cedar_npz_path = "path/to/cedar_features.npz"
    
    config = {
        "sgpds": {
            'exp-users': ['0', '300'],
            'dev-users': ['300', '581'],
            'gen-for-train': '12',
            'gen-for-test':  '10',
            'gen-for-ref': '12',
            'input-features-path': gpdss_npz_path, 
            'best-k': {"svm": '150', "sgd": '100'},
            'k-range': [10] + list(range(50, 601,50)),
            },

        "cedar": {
            'exp-users': ['27', '55'],
            'dev-users': ['0', '27'],
            'gen-for-train': '12',
            'gen-for-test':  '10',
            'gen-for-ref': '12',
            'input-features-path': cedar_npz_path,
            'best-k': {"svm": '10', "sgd": '10'},
            'k-range': [10] + list(range(50, 501,50)),
            },

        "mcyt": {
            'exp-users': ['37', '75'],
            'dev-users': ['0', '37'],
            'gen-for-train': '12',
            'gen-for-test':  '5',
            'gen-for-ref': '10',
            'input-features-path': mcyt_npz_path,
            'best-k': {"svm": '50', "sgd": '100'},
            'k-range': [10] + list(range(50, 401,50)),
            },
        }
    
    reproduce_validation(f_pred_val_path, f_metric_val_path, config)
    
    reproduce_test(f_pred_test_path, f_metric_test_path, config)
    
   