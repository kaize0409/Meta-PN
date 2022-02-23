import arguments
from utils import *
from models import MetaLabelPropagation, MLP
import random
from early_stop import EarlyStopping, Stop_args
from L2P import *
import numpy as np

args = arguments.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.ini_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.ini_seed)


def run():

    # Load data and pre_process data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(graph_name=args.dataset, shot=args.shot, seed = args.seed)
    y0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for i in idx_train:
        y0[i][labels[i]] = 1.0

    idx_all = list(range(len(labels)))
    y_soft_train = label_propagation(adj, labels, idx_train, args.K, args.init_alpha)
    y_soft_val = label_propagation(adj, labels, idx_val, args.K, args.init_alpha)
    y_soft_test = label_propagation(adj, labels, idx_test, args.K, args.init_alpha)

    features = features.to(device)
    adj = adj.to(device)
    y_soft_train = y_soft_train.to(device)
    y_soft_val = y_soft_val.to(device)
    y_soft_test = y_soft_test.to(device)
    labels = labels.to(device)
    y0 = y0.to(device)
    # Model and optimizer
    main_net = MLP(nfeat=features.shape[1],
                   nhid=args.hidden,
                   nclass=labels.max().item() + 1,
                   K=args.K,
                   alpha=args.init_alpha,
                   dropout=args.dropout).to(device)

    main_net_optimizer = torch.optim.Adam(main_net.parameters(), lr=args.lr_main)

    meta_net = MetaLabelPropagation(K=args.K,
                                    y0=y0,
                                    adj=adj,
                                    features=features).to(device)

    meta_net_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.lr_meta)

    def train(epoch):
        # train model
        meta_net.train()
        main_net.train()

        # batchify data
        batchified_idx = data_idx_batchify(idx_all, args.batch_size, shuffle=True)
        # print(meta_net.weights)
        for step, batch_idx in enumerate(batchified_idx):
            # bi-level optimization
            # update meta_net

            idx_gold = np.setdiff1d(idx_train, batch_idx)
            meta_net_optimizer.zero_grad()
            meta_loss_unrolled_backward(main_net, main_net_optimizer, meta_net,
                                        features[batch_idx], batch_idx,
                                        features[idx_gold], y_soft_train[idx_gold], args.lr_main)
            meta_net_optimizer.step()

            # update main_net
            y_pseudo = meta_net(batch_idx).detach()
            main_net_optimizer.zero_grad()
            logits = main_net(features[batch_idx])
            loss = args.loss_decay * main_net.soft_cross_entropy(logits, y_pseudo) + args.weight_decay * torch.sum(main_net.Linear1.weight ** 2) / 2
            loss.backward()
            main_net_optimizer.step()

        # Evaluate validation set performance separately
        main_net.eval()
        output = main_net(features)
        loss_val = args.loss_decay * main_net.soft_cross_entropy(output, y_soft_val)
        output = main_net.inference(output, adj)
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print('epoch: {:04d}'.format(epoch + 1),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))

        return loss_val.item(), acc_val.item()

    def test():
        main_net.eval()
        output = main_net(features)
        loss_test = args.loss_decay * main_net.soft_cross_entropy(y_hat=output, y_soft=y_soft_test)

        output = main_net.inference(output, adj)
        acc_test = accuracy(output[idx_test], labels[idx_test])

        return loss_test.item(), acc_test.item()

    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(main_net, **stopping_args)
    # train the model
    for epoch in range(args.epochs):
        loss_val, acc_val = train(epoch)
        if early_stopping.check([acc_val, loss_val], epoch + 1):
            break
        if (epoch + 1) % 10 == 0:
            test()

    # restore best model
    main_net.load_state_dict(early_stopping.best_state)

    return test()


if __name__ == '__main__':

    loss, acc = run()
    print("Test set results:",
          "loss= {:.4f}".format(loss),
          "accuracy= {:.4f}".format(acc))