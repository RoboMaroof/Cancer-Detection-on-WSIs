from __future__ import print_function
import numpy as np
import argparse
import torch
import os
import pandas as pd
from utils.utils import *
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from utils.eval_utils import initiate_model as initiate_model

parser = argparse.ArgumentParser(description='Top-k Patch Saving Script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()

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
    return params

def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    return config_dict

if __name__ == '__main__':
    config_path = os.path.join('heatmaps/configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 50.0, 'a_h': 8.0, 'max_n_holes': 10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
    else:
        df = pd.read_csv(data_args.process_list)
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    if model_args.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError

    for i in range(len(df)):

        slide_name = df.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        slide_id = slide_name.replace(data_args.slide_ext, '')
        print('\nProcessing slide: ', slide_name)
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(df.loc[i], seg_params)
        filter_params = load_params(df.loc[i], filter_params)
        vis_params = load_params(df.loc[i], vis_params)

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

        print('Initializing WSI object')
        slide_path = os.path.join(data_args.data_dir, slide_name)
        # mask_file = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(slide_id), '{}_mask.pkl'.format(slide_id))

        try:
            label = df.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)
        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')


        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)

        label = df.loc[i, 'label']
        label_dict = data_args.label_dict
        class_labels = list(label_dict.keys())
        class_encodings = list(label_dict.values())
        reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

        patch_size = tuple([args['patching_arguments']['patch_size'] for i in range(2)])
        step_size = tuple((np.array(patch_size) * (1 - args['patching_arguments']['overlap'])).astype(int))

        blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
                             'custom_downsample': args['patching_arguments']['custom_downsample'], 
                             'level': args['patching_arguments']['patch_level'], 'use_center_shift': args['heatmap_arguments']['use_center_shift']}

        feature_extractor = resnet50_baseline(pretrained=True)
        feature_extractor.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
        else:
            feature_extractor = feature_extractor.to(device)

        features_path = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(label), slide_id, '{}.pt'.format(slide_id))
        h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')

        # Load features

        ##### Check if h5_features_file exists ######
        if not os.path.isfile(h5_path):
            _, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
                                        model=model, 
                                        feature_extractor=feature_extractor, 
                                        batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
                                        attn_save_path=None, feat_save_path=h5_path, 
                                        ref_scores=None)

        if not os.path.isfile(features_path):
            h5_path = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(label), slide_id, '{}.h5'.format(slide_id))
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            file.close()
        else:
            features = torch.load(features_path)

        wsi_object.saveSegmentation(mask_file)
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
        del features


        if not os.path.isfile(block_map_save_path): 
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

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
                sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'], slide_id)
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                    score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))



print("Top-k patches saved successfully.")
