import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/work/scratch/abdul/CLAM/Renal/Trials/Binary_SelectedNormal_250823/tumor_vs_normal_dataset.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/work/scratch/abdul/CLAM/Renal/Trials/05_Normal_&_Subtyping_171023/Normal_Vs_Subtyping_dataset_SELECTED.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            #label_dict = {'KIRC':0, 'KIRP':1, 'KICH':2},
                            label_dict = {'KIRC':0, 'KIRP':1, 'KICH':2, 'Normal':3},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids]) # empty array for each class [num_classes,]
val_num = np.round(num_slides_cls * args.val_frac).astype(int) # array of number of validation samples per class [num_classes,]
test_num = np.round(num_slides_cls * args.test_frac).astype(int) # array of number of test samples per class [num_classes,]

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = '/work/scratch/abdul/CLAM/Renal/Trials/05_Normal_&_Subtyping_171023/splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)

        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)

        for i in range(args.k):
            dataset.set_splits()

            # Report distribution of samples across training, validation, and testing datasets
            descriptor_df = dataset.test_split_gen(return_descriptor=True)

            # Return data splits for training, validation, and testing, based on pre-defined IDs (self.train_ids, self.val_ids, self.test_ids)
            splits = dataset.return_splits(from_id=True)    # <--- tuple of train_split, val_split, and test_split. Each of these elements an instance of the Generic_Split class 
            
            # save dataset splits to CSV file
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)

            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



