import os
import sys
import argparse
import time

localtime = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))


def get_args(root_dir=''):    # hepa_mito_model

    project_dir = os.path.dirname(os.path.abspath(sys.executable))

    print(project_dir)
    parsers = argparse.ArgumentParser(description='SGMHepa')
    
    # dataset
    parsers.add_argument('--root_dir', type=str, default=root_dir)
    parsers.add_argument('--dataset', type=str, default=root_dir+'/dataset/Hepatotoxicity/Hepatotoxicity_mito.csv')
    parsers.add_argument('--dataset_pre', type=str, default=root_dir+'/dataset/Hepatotoxicity/Hepatotoxicity_mito_dataset3_pre3.pkl')
    parsers.add_argument('--property_name', type=str, default='Hepatotoxicity')
    parsers.add_argument('--property_add', type=str, default='mito')
    parsers.add_argument('--dataset_run', type=str, default=root_dir+'/dataset/Hepatotoxicity/Hepatotoxicity_mito_test.csv')
    parsers.add_argument('--load_long', type=bool, default=False)
    
    # molecular encoder
    parsers.add_argument('--agg_func', type=str, default='MEAN')
    parsers.add_argument('--epochs', type=int, default=5000)
    parsers.add_argument('--patience', type=int, default=50)
    parsers.add_argument('--bvalue', type=int, default=0.72)
    parsers.add_argument('--seed', type=int, default=72)
    parsers.add_argument('--cuda', type=bool, default=True, help='use CUDA')
    parsers.add_argument('--gpu', type=int, default=2, help='GPU id to use.')
    parsers.add_argument('--name', type=str, default='debug')
    
    # model set
    parsers.add_argument('--smiles_num_layers', type=int, default=2)
    parsers.add_argument('--smiles_input_dim', type=int, default=64)
    parsers.add_argument('--smiles_hidden_size', type=int, default=64)
    parsers.add_argument('--smiles_latent_size', type=int, default=64)
    parsers.add_argument('--graph_input_size', type=int, default=7)
    parsers.add_argument('--graph_hidden_size', type=int, default=64)
    parsers.add_argument('--graph_latent_size', type=int, default=64)
    parsers.add_argument('--mito_size', type=int, default=1)
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--mean_times', type=int, default=5)
    parsers.add_argument('--lr', type=float, default=0.001)
    parsers.add_argument('--lr_smiles', type=float, default=0.001)
    parsers.add_argument('--lr_graph', type=float, default=0.0005)
    parsers.add_argument('--dropout', type=float, default=0.1)
    parsers.add_argument('--weight_decay', type=float, default=5e-4)
    
    # options
    parsers.add_argument('--mask_ratio', type=float, default=0)
    parsers.add_argument('--node_mask_ratio', type=float, default=0)
    parsers.add_argument('--edge_mask_ratio', type=float, default=0)
    parsers.add_argument('--sequence', type=bool, default=True)
    parsers.add_argument('--graph', type=bool, default=True)
    parsers.add_argument('--graph_conv1', type=str, default='GAT')
    parsers.add_argument('--graph_conv2', type=str, default='GAT')
    parsers.add_argument('--mito', type=str, default=True)

    parsers.add_argument('--save_pt', type=str, default=root_dir+'/outputs/model/Hepa_sgm')
    parsers.add_argument('--load_pt', type=str, default=root_dir+'/outputs/model/Hepa_1.joblib')

    args = parsers.parse_args()
    return args


def write_record(args, message, addr):

    fw = open(f'{args.root_dir}/outputs/{addr}/seq{args.sequence}_graph{args.graph}_mito_{args.mito}_{localtime}.txt', 'a')
    fw.write(f'{message}\n')
    fw.close()


def write_result(args, message, data):

    fw = f'{args.root_dir}/results/seq{args.sequence}_graph{args.graph}_mito_{args.mito}_{localtime}.csv'
    data.to_csv(fw, index=False)
    write_record(args, message, 'use')
