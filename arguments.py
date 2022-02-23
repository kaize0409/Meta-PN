import argparse


# Training settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora_ml',
                        help='Choose from {cora_ml, citeseer, pubmed, ms_academic}')
    parser.add_argument('--K', type=int, default=10,
                        help='the propaggation steps')
    parser.add_argument('--init_alpha', type=float, default=0.1,
                        help='the initial teleport probability')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of samples in a batch.')
    parser.add_argument('--lr_meta', type=float, default=5e-4,
                        help='learning rate.')
    parser.add_argument('--lr_main', type=float, default=1e-1,
                        help='learning rate.')
    parser.add_argument('--shot', type=int, default=5,
                        help='Labeled nodes per class')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='Set weight decay.')
    parser.add_argument('--loss_decay', type=float, default=0.05,
                        help='Set loss_decay.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--seed', type=int, default=3258769933, help='Random seed for split data.')
    parser.add_argument('--ini_seed', type=int, default=3258769933, help='Random seed to initialize parameters.')
    return parser.parse_args()
