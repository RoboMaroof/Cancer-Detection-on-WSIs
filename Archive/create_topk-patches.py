
'''
def save_topk_attention_patches(wsi_object, k, output_folder, slide_id, patch_args, sample_args, topk_output_folder):
    # Load attention scores
    attention_scores_file = os.path.join(output_folder, '{}_blockmap.h5'.format(slide_id))
    with h5py.File(attention_scores_file, 'r') as file:
        attention_scores = file['attention_scores'][:]
        coords = file['coords'][:]

    # Find the top-K indices
    topk_indices = np.argpartition(attention_scores, -k)[-k:]

    # Create the directory for saving the top-K attention patches
    os.makedirs(topk_output_folder, exist_ok=True)

    # Load the label_dict and model
    # ... (code to load label_dict and model)

    # Save the top-K attention patches
    for idx, topk_idx in enumerate(topk_indices):
        coord = coords[topk_idx]
        x, y = coord[0], coord[1]

        # Read the patch from the WSI object
        patch = wsi_object.wsi.read_region((x, y), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size))

        # Save the patch as an image
        patch.save(os.path.join(topk_output_folder, f'{slide_id}_topk_{idx}.png'))

if __name__ == '__main__':
    # Define parameters
    slide_id = "your_slide_id"
    k = 10  # Number of top-K patches to save
    output_folder = "path_to_output_folder"
    topk_output_folder = "path_to_topk_output_folder"

    # Initialize WSI object, load data, and perform other setup as needed
	wsi_object = initialize_wsi(slide_path, seg_mask_path, seg_params, filter_params)
    
    # Call the function to save top-K attention patches
    save_topk_attention_patches(wsi_object, k, output_folder, slide_id, patch_args, sample_args, topk_output_folder)

'''


##########
from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from PIL import Image


parser = argparse.ArgumentParser(description='Configuration for topk patch generation')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--results_dir', default=None, help='results directory')
parser.add_argument('--k', type=int, default=40, help='topk')
parser.add_argument('--patch_size', type=int, default=256, help='patch size')
parser.add_argument('--overlap', type=float, default=0.9,
                    help='overlap of patches (default: 0.9)')
parser.add_argument('--slide_ext', type=str, default='.svs', 
                    help='slide extension')
parser.add_argument('--drop_out', action='store_true', default=True,
                     help='drop out')
parser.add_argument('--n_classes', type=int, default=4,
                    help='number of classes (default: 4)')
parser.add_argument('--model_size', type=str, default='small', 
                    help='model size')
parser.add_argument('--model_type', type=str, default='clam_sb', 
                    help='model type')

args = parser.parse_args()

args.label_dict = {'KIRC':0, 'KIRP':1, 'KICH':2, 'Normal':3}
args.ckpt_path = '/work/scratch/abdul/CLAM/Renal/Trials/04_Normal_Vs_Subtyping_051023/results/task_2_tumor_subtyping_CLAM_100_s1/s_4_checkpoint.pt'
args.process_list = '/work/scratch/abdul/CLAM/Renal/Trials/00_Dataset_topkpatches/01_topk_50x40/KICH_KIRP_topk_images_SELECTED.csv'

def save_topk_attention_patches(wsi_object, k, output_folder, slide_id, patch_args, sample_args, topk_output_folder):
    # Load attention scores
    attention_scores_file = os.path.join(output_folder, '{}_blockmap.h5'.format(slide_id))
    with h5py.File(attention_scores_file, 'r') as file:
        attention_scores = file['attention_scores'][:]
        coords = file['coords'][:]

    # Find the top-K indices
    topk_indices = np.argpartition(attention_scores, -k)[-k:]

    # Create the directory for saving the top-K attention patches
    os.makedirs(topk_output_folder, exist_ok=True)

    # Load the label_dict and model
    # ... (code to load label_dict and model)

    # Save the top-K attention patches
    for idx, topk_idx in enumerate(topk_indices):
        coord = coords[topk_idx]
        x, y = coord[0], coord[1]

        # Read the patch from the WSI object
        patch = wsi_object.wsi.read_region((x, y), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size))

        # Save the patch as an image
        patch.save(os.path.join(topk_output_folder, f'{slide_id}_topk_{idx}.png'))

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, _ = model(features)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
		
		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict

if __name__ == '__main__':


	patch_size = tuple([args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], args.overlap, step_size[0], step_size[1]))


	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	df = pd.read_csv(args.process_list)
	df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	ckpt_path = args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	

	model =  initiate_model(args, ckpt_path)

	feature_extractor = resnet50_baseline(pretrained=True)
	feature_extractor.eval()
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Done!')

	label_dict =  args.label_dict
	class_labels = list(label_dict.keys())
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

	if torch.cuda.device_count() > 1:
		device_ids = list(range(torch.cuda.device_count()))
		feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
	else:
		feature_extractor = feature_extractor.to(device)

	#os.makedirs(exp_args.production_save_dir, exist_ok=True)
	#os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
	'custom_downsample':args.custom_downsample, 'level': args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}


	# MAIN LOOP STARTS HERE
	for i in range(len(process_stack)):
		slide_name = process_stack.loc[i, 'slide_id']
		if args.slide_ext not in slide_name:
			slide_name+=args.slide_ext
		print('\nprocessing: ', slide_name)	

		try:
			label = process_stack.loc[i, 'label']
		except KeyError:
			label = 'Unspecified'

		slide_id = slide_name.replace(args.slide_ext, '')

		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label

		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)

		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		if isinstance(args.data_dir, str):
			slide_path = os.path.join(args.data_dir, slide_name)
		elif isinstance(args.data_dir, dict):
			data_dir_key = process_stack.loc[i, args.data_dir_key]
			slide_path = os.path.join(args.data_dir[data_dir_key], slide_name)
		else:
			raise NotImplementedError

		mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
		
		# Load segmentation and filter parameters
		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()

		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		vis_params = load_params(process_stack.loc[i], vis_params)

		keep_ids = str(seg_params['keep_ids'])
		if len(keep_ids) > 0 and keep_ids != 'none':
			seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		else:
			seg_params['keep_ids'] = []

		exclude_ids = str(seg_params['exclude_ids'])
		if len(exclude_ids) > 0 and exclude_ids != 'none':
			seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		else:
			seg_params['exclude_ids'] = []

		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))
		
		
		print('Initializing WSI object')
		wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
		print('Done!')

		wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

		block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))	#constructs the full file path for saving the blockmap (attention scores) as an HDF5 file
		mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
		if vis_params['vis_level'] < 0:
			############################
			best_level = wsi_object.wsi.get_best_level_for_downsample(32)
			############################
			vis_params['vis_level'] = best_level
		mask = wsi_object.visWSI(**vis_params, number_contours=True)
		mask.save(mask_path)
		
		features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
		h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
	

		##### check if h5_features_file exists ######
		if not os.path.isfile(h5_path) :
			_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
											model=model, 
											feature_extractor=feature_extractor, 
											batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
											attn_save_path=None, feat_save_path=h5_path, 
											ref_scores=None)				
		
		##### check if pt_features_file exists ######
		if not os.path.isfile(features_path):
			file = h5py.File(h5_path, "r")
			features = torch.tensor(file['features'][:])
			torch.save(features, features_path)
			file.close()

		# load features 
		features = torch.load(features_path)
		process_stack.loc[i, 'bag_size'] = len(features)
		
		wsi_object.saveSegmentation(mask_file)
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
		del features
		
		if not os.path.isfile(block_map_save_path): 
			file = h5py.File(h5_path, "r")
			coords = file['coords'][:]
			file.close()
			asset_dict = {'attention_scores': A, 'coords': coords}
			block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
		# save top 3 predictions
		for c in range(exp_args.n_classes):
			process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
			process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

		os.makedirs('heatmaps/results/', exist_ok=True)
		if args.process_list is not None:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(args.process_list.replace('.csv', '')), index=False)
		else:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
		
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		samples = sample_args.samples
		for sample in samples:
			if sample['sample']:
				tag = "label_{}_pred_{}".format(label, Y_hats[0])
				sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
				os.makedirs(sample_save_dir, exist_ok=True)
				print('sampling {}'.format(sample['name']))
				sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
					score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
				for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
					print('coord: {} score: {:.3f}'.format(s_coord, s_score))
					patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
					patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

		wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
		'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}



	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)
