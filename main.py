import argparse
from data_loader import load_data
from train import train
import os
from utils import tab_printer, get_logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, help='use gpu', action='store_true')

    # settings for datase
    parser.add_argument('--dataset', type=str, default='DDB14', choices=['FB15k','FB15k-237','wn18','wn18rr','NELL995','DDB14'], help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', choices=['id', 'bow', 'bert'], help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, choices=['True', 'False'], help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=1, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, choices=['True', 'False'], help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=1, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')


    args = parser.parse_args()
    logger = get_logger(args.dataset, "./log/", './config/')
    logger.info(args)
    tab_printer(args)
    data = load_data(args)
    train(args, data)

if __name__ == '__main__':
    main()
