import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.6, help='the fraction of clients: C')

    parser.add_argument('--dataset', type=str, default='resnet', help="name of dataset")
    parser.add_argument('--setting', type=int, default=0, help= "Set to 0 for IID, 1 for non-iid, 2 for mislabeled, 3 for noisy.")
    parser.add_argument('--retrain', type=int, default=0, help= "Retrain the global model w/ reweighting (default: 0)")
    parser.add_argument('--badclient_prop', type=float, default=0.2, help= "Proportion of either non-iid or mislabeled or noisy clients.")
    parser.add_argument('--num_categories_per_client', type=int, default=4, help= "Number of categories per client in non-iid setting.")
    parser.add_argument('--badsample_prop', type=float, default=0.6, help= "Proportion of mislabeled/noisy samples per client in mislabeled/noisy setting.")
    parser.add_argument('--noise_std', type=float, default=0.0, help= "Standard deviation of Gaussian noise to add to the dataset.")

    # training arguments
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--optimizer', type=str, default='adam', help="type of optimizer")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--acc_stopping', type=int, default=1, help='Stopping criterion based on accuracy (default: 1)')

    # simulation arguments
    parser.add_argument('--processes', type=int, default=8, help="number of processes")
    parser.add_argument('--shapley_processes', type=int, default=6, help="number of processes for shapley values")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')

    args = parser.parse_args()
    return args
