import argparse
import os

import torch
from datasets import Cifar10
from losses import cross_entropy
from models import Net
from process import evaluate, fit, validate
from torch import optim

optims_dict = {
    'adam': {
        'method': optim.Adam,
        'params': {}
    },
    'sgd': {
        'method': optim.SGD,
        'params': {
            'momentum': 0.9
        }
    }  # same as original paper
}


def execute(args):

    filepath = None
    last_epoch = 0
    history = None

    device = torch.device(
        "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)

    criterion = cross_entropy.to(device)

    optimizer = optims_dict[args.optim]['method'](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        **optims_dict[args.optim]['params'])

    dataset = Cifar10(args.data_dir, args.batch_size, args.val_split,
                      args.seed)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay)

    if args.save:
        filepath = os.path.join(args.model_dir, 'checkpoint.pth')
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

    if args.resume:
        # only resumes from the last not from the best
        filepath = os.path.join(args.model_dir, 'checkpoint{}.pth')
        checkpoint = torch.load(filepath.format('-last'), map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        history = checkpoint['history']

        filepath = filepath.format('')

        msg = ' '.join([
            '{} {:.3f}'.format(name, val[-1])
            for name, val in history['val'].items()
        ])
        print(' * Val @ Epoch {} :: {}'.format(last_epoch, msg))

    if args.train:
        fit(dataset.train_loader,
            dataset.valid_loader,
            model,
            criterion,
            args.epochs,
            device,
            optimizer,
            savepath=filepath,
            start_epoch=last_epoch + 1,
            history=history)
        print('Finished Training')

    if args.test:
        if not args.train and not args.resume:
            raise ValueError(
                'Testing requires a previously trained model (load it or train it)'
            )
        validate(dataset.test_loader, model, criterion, device)

    if args.eval:
        if not args.train and not args.resume:
            raise ValueError(
                'Testing requires a previously trained model (load it or train it)'
            )
        evaluate(dataset.test_loader,
                 model,
                 device,
                 savepath=filepath,
                 idx_2_class=dataset.class_names)
        print('Finished Testing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch training w/ subband separation')
    parser.add_argument('--model-dir',
                        default='./models/',
                        help='path to save/load model')
    parser.add_argument('--data-dir', default='./data', help='dir to dataset')

    parser.add_argument('--optim',
                        default='adam',
                        type=str,
                        choices=optims_dict.keys(),
                        help='optimizer')
    parser.add_argument('--wd',
                        default=5e-4,
                        type=float,
                        help='weight decay for optimizer')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--lr-decay',
                        default=15,
                        type=int,
                        help='learning rate decay interval')
    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='nb of epochs to train')
    parser.add_argument('--batch-size',
                        default=32,
                        type=int,
                        help='size of mini-batch')
    parser.add_argument('--val-split',
                        default=0.1,
                        type=float,
                        help='val:train set size ratio')

    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='value for random seed')

    parser.add_argument('--resume',
                        '-r',
                        action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save',
                        '-s',
                        action='store_true',
                        help='save checkpoint')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='use gpu acceleration')

    parser.add_argument('--train',
                        action='store_true',
                        help='train on training set (and use validation set)')
    parser.add_argument('--test',
                        action='store_true',
                        help='either to test or not')
    parser.add_argument('--eval',
                        action='store_true',
                        help='compute predictions on test set')

    args = parser.parse_args()
    execute(args)
