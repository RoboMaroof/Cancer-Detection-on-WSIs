from __future__ import print_function

import argparse
import pdb
import os
import math
import time

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *

##########
from utils.core_utils import train
##########

from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.metrics import balanced_accuracy_score, f1_score, average_precision_score, confusion_matrix


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    #START FOLD
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    #END FOLD
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []

    test_balanced_accuracies = []
    test_f1_scores = []
    test_average_precisions = []

    folds = np.arange(start, end)

    #Repeat for number of folds
    for i in folds:
        seed_torch(args.seed)

        #Dataset splits through datasets/dataset_generic.py 
        #dataset object of datasets/dataset_generic.py --> Generic_MIL_Dataset class
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, true_labels, predicted_labels  = train(datasets, i, args)   #TO utils/core_utils.py --> TRAIN 
        # RETURNS: results, test_auc, val_auc, test_acc, val_acc FOR EACH FOLD
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        test_balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
        test_f1 = f1_score(true_labels, predicted_labels, average='weighted')  # 'weighted' accounts for multi-class
        '''
        ap_scores = []
        predicted_labels = np.array(predicted_labels)
        print(np.unique(predicted_labels))
        true_labels = np.array(true_labels)
        print(np.unique(true_labels))
        for class_idx in range(args.n_classes):
            true_labels_one_class = (true_labels == class_idx)
            predicted_scores_one_class = predicted_labels[:, class_idx]
            ap_score = average_precision_score(true_labels_one_class, predicted_scores_one_class)
            ap_scores.append(ap_score)

        # Calculate mean average precision (mAP)
        test_map = np.mean(ap_scores)
        '''


        average_precisions = {}
        for class_label in range(args.n_classes):
            # Create binary labels for the current class
            true_class_labels = [1 if label == class_label else 0 for label in true_labels]
            predicted_class_scores = [1 if label == class_label else 0 for label in predicted_labels]

            # Calculate average precision for the current class
            ap = average_precision_score(true_class_labels, predicted_class_scores)
            average_precisions[class_label] = ap

        # Print average precision for each class
        for class_label, ap in average_precisions.items():
            print(f'Class {class_label}: Average Precision = {ap:.4f}')

        # Calculate the overall mean average precision
        test_map = np.mean(list(average_precisions.values()))
        print(f'Mean Average Precision = {test_map:.4f}')


        test_balanced_accuracies.append(test_balanced_acc)
        test_f1_scores.append(test_f1)
        test_average_precisions.append(test_map)

        #Confusion Matrix
        confusion = confusion_matrix(true_labels, predicted_labels)
        print(f'Confusion Matrix for Fold {i}:\n{confusion}')


        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    #Dataframe of validation and test scores for all (10) folds --> saved in summary.csv in results dir
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc, 'test_mAP': test_average_precisions, 'test_F1': test_f1_scores, 'test_bACC' : test_balanced_accuracies })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


#START POINT --> GO DOWN
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})


# LOAD DATASET according to CSV PATH
print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/work/scratch/abdul/CLAM/Renal/Trials/Binary_SelectedNormal_250823/tumor_vs_normal_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    ############################
    #args.n_classes=3
    args.n_classes=4
    
    dataset = Generic_MIL_Dataset(csv_path = '/work/scratch/abdul/CLAM/Renal/Trials/08_Triplet_Center_Loss_011223/Normal_Vs_Subtyping_dataset_SELECTED.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            #label_dict = {'KIRC':0, 'KIRP':1, 'KICH':2},
                            label_dict = {'KIRC':0, 'KIRP':1, 'KICH':2, 'Normal':3},
                            patient_strat= False,
                            ignore=[])
    ######################
    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
        
else:
    raise NotImplementedError

#RESULTS DIR    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

#SPLITS DIR
if args.split_dir is None:
    args.split_dir = os.path.join('/work/scratch/abdul/CLAM/Renal/Trials/04_Normal_Vs_Subtyping_051023/splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('/work/scratch/abdul/CLAM/Renal/Trials/04_Normal_Vs_Subtyping_051023/splits', args.split_dir)
    

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    start_time = time.time()
    results = main(args)
    #GO UP to MAIN
    end_time = time.time()
    elapsed_time = (end_time - start_time)/3600
    print("finished!")
    print(f"Training took {elapsed_time:.2f} hours")
    print("end script")


