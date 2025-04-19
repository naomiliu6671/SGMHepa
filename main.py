import random
import numpy as np
import pandas as pd
import torch
import sys
import torch.nn.functional as F
import time
from joblib import dump
import pickle
from sklearn.utils import compute_class_weight
from tqdm import tqdm
from model_set.model import CombinedModel
from parsers import get_args, write_record
from utils.evaluate import evaluation, mean_std
from utils.data_process import load_smiles_data
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device_id = torch.cuda.current_device()
            print('using device', device_id, torch.cuda.get_device_name(device_id))
    args.device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    print(f'current running on the device: cuda_{args.gpu} with loading type: {args.load_long}')
    write_record(args, f'current running on the device: {args.device} with loading type: {args.load_long}',
                 'train')
    print(
        f'data: {args.property_name} seq: {args.sequence} graph: {args.graph} mito: {args.mito} graph_conv: {args.graph_conv1}, {args.graph_conv2}')
    write_record(args,
                 f'data: {args.property_name} seq: {args.sequence} graph: {args.graph} mito: {args.mito} graph_conv: {args.graph_conv1}, {args.graph_conv2}',
                 'train')
    sys.stdout.flush()
    if os.path.isfile(args.dataset_pre):
        with open(args.dataset_pre, "rb") as f:
            feature_dicts = pickle.load(f)
        print('load preprocessed data')
        train_data, train_labels, train_mitos, test_data, test_labels, test_mitos, node2index, node_types, graph_mask_id = (
            feature_dicts['train_data'],
            feature_dicts['train_labels'], feature_dicts['train_mitos'], feature_dicts['test_data'],
            feature_dicts['test_labels'],
            feature_dicts['test_mitos'], feature_dicts['node2index'], feature_dicts['node_types'],
            feature_dicts['graph_mask_id'])
    else:
        train_data, train_labels, train_mitos, test_data, test_labels, test_mitos, node2index, node_types, graph_mask_id = load_smiles_data(
            args)
        feature_dicts = {'train_data': train_data, 'train_labels': train_labels, 'train_mitos': train_mitos,
                         'test_data': test_data, 'test_labels': test_labels, 'test_mitos': test_mitos,
                         'node2index': node2index, 'node_types': node_types, 'graph_mask_id': graph_mask_id}
        pickle.dump(feature_dicts, open(args.dataset_pre, "wb"))
        print('preprocessed data saved')
    args.smiles_input_size_graph = len(node_types)
    args.graph_mask_id = graph_mask_id

    total_metrics = pd.DataFrame(columns=['auc_score', 'pre', 'rec', 'f1', 'acc', 'spe', 'mcc'])
    # print(args)
    for atime in range(args.mean_times):
        args.patience = 50
        write_record(args, f'\n{str(args)}', 'train')
        sys.stdout.flush()
        print(f'Running {atime + 1} times ...')
        write_record(args, f'\nRunning {atime + 1} times ...\n', 'train')
        multiclass_metrics = []
        bad_cases = 0
        save_path = f'{args.save_pt}_5fold_{atime + args.add_num}.joblib'
        model = CombinedModel(args).to(args.device)

        for epoch in range(args.epochs):
            start_time = time.time()
            print("Training, epoch %d ..." % epoch)
            sys.stdout.flush()
            train_graph = np.arange(len(train_data['sequence']))
            np.random.shuffle(train_graph)
            epoch_len = len(train_graph)
            sample = []
            total_loss = 0
            model.train()
            for idx, s in enumerate(train_graph):
                sample.append(s)
                if (idx + 1) == epoch_len and len(sample) < args.batch_size:
                    index_range = len(sample)
                    for i in range(args.batch_size - index_range):
                        idx_add = random.randint(0, epoch_len - index_range - 1)
                        sample.append(train_graph[idx_add])
                if (idx + 1) % args.batch_size == 0 or (idx + 1) == epoch_len:
                    logits, loss = model(sample, train_data, train_labels, train_mitos, node2index)
                    total_loss += loss.item()
                    sample.clear()
            print(f'epoch: {epoch} total_loss: {total_loss:.4f}')

            test_graph = np.arange(len(test_data['sequence']))
            np.random.shuffle(test_graph)
            print(f"Testing, epoch {epoch} ...")
            sys.stdout.flush()
            outs = torch.zeros(test_labels.shape[0], 2)
            model.eval()
            with torch.no_grad():
                for graph_index in tqdm(test_graph, desc=f'Epoch {epoch}', ascii=True, leave=False):
                    out = model.pred(graph_index, test_data, test_mitos)
                    outs[graph_index, :] = out
                test_pred = outs
                test_pred_ = test_pred.cpu().detach().numpy()
                test_pred_label = torch.max(test_pred, 1)[1]
                test_pred_label = test_pred_label.cpu().detach().numpy()
                pre, rec, f1, acc, auc_score, spe, mcc = evaluation(test_pred_label, test_pred_, test_labels)
                multiclass_metrics.append([auc_score, pre, rec, f1, acc, spe, mcc])
                best_auc = sorted(multiclass_metrics, key=lambda x: x[0], reverse=True)[0][0]

                if auc_score >= best_auc:
                    bad_cases = 0
                    dump(model, save_path)
                    if auc_score > args.bvalue and args.patience > 50:
                        args.patience = 50
                        print(f'patience: {args.patience}')
                else:
                    bad_cases += 1

                print(
                    f'epoch: {epoch} avg_loss: {total_loss:.4f} pre.:{pre:.4f} rec.:{rec:.4f} f1:{f1:.4f} acc:{acc:.4f} spe:{spe:.4f} mcc:{mcc:.4f} auc:{auc_score:.4f} ## best_auc:{best_auc:.4f} times:{bad_cases}')
                write_record(args,
                             f'epoch: {epoch} avg_loss: {total_loss:.4f} pre.:{pre:.4f} rec.:{rec:.4f} f1:{f1:.4f} acc:{acc:.4f} mcc:{mcc:.4f} spe:{spe:.4f} auc:{auc_score:.4f} ## best_auc:{best_auc:.4f}',
                             'train')
                print(f'Done, epoch {epoch}, took {float(time.time() - start_time) / 60.:.1f} minutes ...')
                print('-------------' * 6)

                if bad_cases > args.patience:
                    if best_auc > args.bvalue:
                        best_re = sorted(multiclass_metrics, key=lambda x: x[0], reverse=True)[0]
                        auc_score, pre, rec, f1, acc, spe, mcc = best_re[0], best_re[1], best_re[2], best_re[3], best_re[4],best_re[5], best_re[6]
                        print(f'Early stop at epoch {epoch} ... {bad_cases} times')
                        print(
                            f'best pre.:{pre:.4f} rec.:{rec:.4f} f1:{f1:.4f} acc:{acc:.4f} spe:{spe:.4f} mcc:{mcc:.4f} auc:{auc_score:.4f}')
                        total_metrics.loc[atime, :] = [auc_score, pre, rec, f1, acc, spe, mcc]
                        break
                    else:
                        if args.patience < 100:
                            args.patience = 100
                            print(f'patience: {args.patience}')
                        else:
                            best_re = sorted(multiclass_metrics, key=lambda x: x[0], reverse=True)[0]
                            auc_score, pre, rec, f1, acc, spe, mcc = best_re[0], best_re[1], best_re[2], best_re[3], best_re[4],best_re[5], best_re[6]
                            print(f'Early stop at epoch {epoch} ... {bad_cases} times')
                            print(
                                f'best pre.:{pre:.4f} rec.:{rec:.4f} f1:{f1:.4f} acc:{acc:.4f} spe:{spe:.4f} mcc:{mcc:.4f} auc:{auc_score:.4f}')
                            total_metrics.loc[atime, :] = [auc_score, pre, rec, f1, acc, spe, mcc]
                            break
                sys.stdout.flush()
    # print(total_metrics.columns)
    for i in range(len(total_metrics)):
        write_record(args,
                     f'time: {i + 1} pre.:{total_metrics.at[i, "pre"]:.4f} rec.:{total_metrics.at[i, "rec"]:.4f} f1:{total_metrics.at[i, "f1"]:.4f} acc:{total_metrics.at[i, "acc"]:.4f} spe:{total_metrics.at[i, "spe"]:.4f} mcc:{total_metrics.at[i, "mcc"]:.4f} auc:{total_metrics.at[i, "auc_score"]:.4f}',
                     'train')
    for i in range(len(total_metrics.columns)):
        means = mean_std(total_metrics.iloc[:, i])
        write_record(args, f'{total_metrics.columns[i]}: {means}', 'train')
        print(f'{total_metrics.columns[i]}: {means}')
        sys.stdout.flush()
    write_record(args, f'End\n', 'train')


if __name__ == '__main__':
    args_ = get_args()
    main(args_)

