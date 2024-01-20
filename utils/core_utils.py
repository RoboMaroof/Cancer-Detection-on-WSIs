import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from torch.optim.lr_scheduler import StepLR
import gc
import itertools



################
from models.model_clam import CLAM_SB
from sklearn.manifold import TSNE
################

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, average_precision_score
import topk

import sys
import logging




# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    filename='logfile.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Redirect print statements to the same log file
class PrintLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        pass

#sys.stdout = PrintLogger('/home/students/abdul/ResearchProject/Cancer-Detection-on-Whole-Slide-Images/Debugging/logfile.log')



class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.current_epoch = 0

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        self.current_epoch = epoch
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, patch_datasets, cur, args):
    """   
        train for a single fold
    """
    # FROM MAIN --> START POINT
    # print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=300)

    else:
        writer = None

    # print('\nInit train/val/test splits...', end=' ')

    train_split, val_split, test_split = datasets
    patch_train_split, _, _ = patch_datasets


    
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur))) # creates splits_cur.csv file in results folder
    save_splits([patch_train_split], ['train'], os.path.join(args.results_dir, 'patch_splits_{}.csv'.format(cur)))
    
    '''
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('Patch level data')
    print("Training on {} samples".format(len(patch_train_split)))

    # LOSS FUNCTION
    print('\nInit loss function...', end=' ')
    '''

    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    

    # model ARGS definition
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            # model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()    #MOVE the model's parameters and computation to GPU
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    #scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')


    print('\nPatch level Init Loaders...', end=' ')
    patch_train_loader = get_split_loader(patch_train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    #LOOP IN EPOCHS
    for epoch in range(args.max_epochs):  
        train_loop_clam(epoch, model, train_loader, patch_train_loader, train_split, patch_train_split, optimizer, args.n_classes, args.bag_weight, args.centerloss_weight, args.patchloss_weight, writer, loss_fn, early_stopping, args.max_epochs)    #TO TRAIN_LOOP_CLAM
        
        stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

        #scheduler.step()

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, _, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger, true_labels, pred_labels = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and epoch % 5 == 0:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer and epoch % 5 == 0:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, true_labels, pred_labels


def train_loop_clam(epoch, model, loader, patch_loader, train_split, patch_train_split, optimizer, n_classes, bag_weight, centerloss_weight, patchloss_weight, writer = None, loss_fn = None, early_stopping=None, max_epochs=None):

    #logging.debug('Entering in train loop clam epoch: {}'.format(epoch))
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    train_center_loss = 0.
    inst_count = 0
    center_count = 0

    slide_embeddings = []
    patch_embeddings = []
    labels = []
    patch_labels = []

    update_frequency = 20  # Update class centers every 20th batch
    accumulated_gradients = torch.zeros_like(model.centers)

    fine_tune_start_epoch = len(loader) - len(patch_loader)
    patch_loader_iter = iter(patch_loader)

    loader_iter = iter(loader)

    print('\n')

    for batch_idx in range(len(loader)):
        try:
            #logging.debug('Entered try')

            data, label = next(loader_iter)
            slide_id = str(train_split.slide_data['slide_id'].iloc[batch_idx])

            #logging.debug('Entering in train loop clam batch: {} with slide_id {}'.format(batch_idx, slide_id))

            data, label = data.to(device), label.to(device)

            # Ensure the model is on the correct device
            model.relocate()

            # DATA --> all patches of a single WSL image
            # data.shape --> K x D  K: No of patches in the WSL image   D:1024  Label: 0/1/2

            # Determine if the current epoch is is the final epoch
            is_final_epochs = epoch >= max_epochs - 1 or (early_stopping and (early_stopping.patience - early_stopping.counter) < 2)
            # FORWARD PASS
            logits, Y_prob, Y_hat, _, instance_dict, patch_feature_embeddings, slide_feature_embeddings = model(data, label = label, is_final_epochs = is_final_epochs, instance_eval=True)   #TO CLAM_SB (forward)

            #logging.debug('Forward pass completed train loop clam batch: {} with slide_id {}'.format(batch_idx, slide_id))
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()

            instance_loss = instance_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value

            center_loss = instance_dict['center_loss']
            center_count+=1
            center_loss_value = center_loss.item()
            train_center_loss += center_loss_value

            # Slide LOSS COMPUTATION
            slide_loss = bag_weight * loss + (1 - bag_weight) * instance_loss   # without center loss
            slide_total_loss = (1 - centerloss_weight) * slide_loss + centerloss_weight * center_loss
            
            
            # PATCH level data set
            if batch_idx >= fine_tune_start_epoch:
                try:
                    patch_id = str(patch_train_split.slide_data['slide_id'].iloc[batch_idx-fine_tune_start_epoch])
                    #logging.debug('Entering in train loop clam batch: {} with patch_id {}'.format(batch_idx, patch_id))
                    
                    patch_data, patch_label = next(patch_loader_iter)
                    patch_data, patch_label = patch_data.to(device), patch_label.to(device)
                    patch_logits, patch_Y_prob, patch_Y_hat, _, patch_instance_dict, _, _ = model(patch_data, label=patch_label, instance_eval=False)

                    patch_loss = loss_fn(patch_logits, patch_label)
                    patch_center_loss = patch_instance_dict['center_loss']
                    patch_total_loss = (1 - centerloss_weight) * patch_loss + centerloss_weight * patch_center_loss

                    total_loss = (1 - patchloss_weight) * slide_total_loss + patchloss_weight * patch_total_loss
                except:
                    total_loss = slide_total_loss
                    #logging.exception('Exception occurred in train loop clam at patch_id {}'.format(patch_id)) 
                    #print('Patch level sample skipped')
                    pass
            else:
                total_loss = slide_total_loss

            # BACKWARD PASS
            total_loss.backward()
            
            model.batch_counter += 1
            if model.batch_counter % update_frequency == 0:
                # Update class centers based on accumulated gradients
                optimizer_centers = torch.optim.SGD([model.centers], lr=2e-4)
                optimizer_centers.zero_grad()

                # Sum the accumulated gradients over the previous 20 batches
                model.centers.grad = accumulated_gradients

                # Average the gradients
                model.centers.grad /= update_frequency

                # Update the class centers
                optimizer_centers.step()

                # Reset the accumulated gradients
                accumulated_gradients.zero_()
            else:
                # Accumulate gradients for the current batch
                accumulated_gradients += model.centers.grad.clone()

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            train_loss += loss_value
            #if (batch_idx + 1) % 20 == 0:
            #    print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, center_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, center_loss.item(), total_loss.item()) + 
            #        'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

            error = calculate_error(Y_hat, label)
            train_error += error
            

            # GRADIEND DESCENT OPTIMIZATION
            optimizer.step()
        
            # Clear the gradients
            optimizer.zero_grad()

            if is_final_epochs:
            # Feature embeddings
                #print('Entering final epochs --> Embedding logging started')
                patch_embeddings.append(patch_feature_embeddings.detach().cpu().numpy())
                num_embeddings = patch_feature_embeddings.size(0)  # Get the number of patch embeddings
                repeated_labels = [label.detach().cpu().numpy()] * num_embeddings  # Repeat the label for each embedding
                patch_labels.append(repeated_labels)

                slide_embeddings.append(slide_feature_embeddings.detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())      

            #logging.debug('Exiting in train loop clam batch: {} with slide_id {}'.format(batch_idx, slide_id))
            #log_gpu_usage()
            #log_cpu_usage()
            #log_system_metrics()

        except:
            #print('Slide level sample skipped')
            #logging.exception('Exception occurred in train loop clam at slide_id {}'.format(slide_id)) 
            pass

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None and epoch % 5 == 0:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if center_count > 0:
        train_center_loss /= center_count
    
    if writer and epoch % 5 == 0:     # Log every 5 epochs
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        writer.add_scalar('train/center_loss', train_center_loss, epoch)

    if is_final_epochs:
        # Concatenate embeddings and labels
        patch_embeddings = np.concatenate(patch_embeddings, axis=0)
        patch_embeddings = patch_embeddings.reshape(-1, patch_embeddings.shape[-1])
        patch_labels = np.concatenate(patch_labels, axis=0)

        slide_embeddings = np.concatenate(slide_embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
    
        # Log embeddings to TensorBoard
        writer.add_embedding(
            mat=patch_embeddings,
            metadata=patch_labels,
            global_step=(epoch * len(loader) + batch_idx),
            tag='patch_embeddings'  
        )

        writer.add_embedding(
            mat=slide_embeddings,
            metadata=labels,
            global_step=(epoch * len(loader) + batch_idx),
            tag='slide_embeddings'  
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Delete large temporary variables
    del slide_embeddings, patch_embeddings, labels, patch_labels, accumulated_gradients, patch_loader_iter


def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            try:
                data, label = data.to(device), label.to(device)      
                logits, Y_prob, Y_hat, _, instance_dict, _, _= model(data, label=label, instance_eval=True)
                acc_logger.log(Y_hat, label)
                
                loss = loss_fn(logits, label)

                val_loss += loss.item()

                instance_loss = instance_dict['instance_loss']
                
                inst_count+=1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value

                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)

                prob[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()
                
                error = calculate_error(Y_hat, label)
                val_error += error
            except:
                print('Slide skipped in Validation')
                pass


    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer and epoch % 5 == 0:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None and epoch % 5 == 0:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping in Validate_clam")
            return True

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    true_labels = []  # To store true class labels
    predicted_labels = []  # To store predicted class labels

    for batch_idx, (data, label) in enumerate(loader):
        try:
            data, label = data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                logits, Y_prob, Y_hat, _, _, _, _ = model(data)

            #############################################
            _, predicted = torch.max(logits, 1)
            true_labels.extend(label.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            #############################################

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error
        except:
            print('Slide skipped in Testing')
            pass

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger, true_labels, predicted_labels
