import argparse
import os

import torch
import datasets
from losses import cross_entropy
from models import SRCNN
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

model_dir_prefix = './models'

def execute(args):

    filepath = None
    last_epoch = -1
    history = None

    device = torch.device(
        "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    dataset = datasets.dataset[args.dataset](args.data_dir, args.batch_size, args.val_split,
                                             args.seed, extra_transforms=args.transforms)

    model = SRCNN(args.wavelet, args.level, 'grayscale' in args.transforms)
    model = model.to(device)
    criterion = cross_entropy.to(device)

    optimizer = optims_dict[args.optim]['method'](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        **optims_dict[args.optim]['params'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

    if args.save:
        model_dir = os.path.join(model_dir_prefix, args.dataset, args.model)
        filepath = os.path.join(model_dir, 'checkpoint.pth')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(vars(args), os.path.join(model_dir, 'config.pth'))

    if args.resume:
        # only resumes from the last not from the best
        filepath = os.path.join(model_dir_prefix, args.dataset, args.model, 'checkpoint{}.pth')
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
            scheduler,
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
                 savepath=filepath)
        print('Finished Testing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch training w/ subband separation')
    parser.add_argument('--model',
                        default='trial',
                        help='model name (creates dir to save/load the model)')
    parser.add_argument('--data-dir', default='./data', help='dir to dataset')
    parser.add_argument('--dataset', type=str, choices=datasets.dataset.keys(), help='dataset name')

    parser.add_argument('--optim',
                        default='adam',
                        type=str,
                        choices=optims_dict.keys(),
                        help='optimizer')
    parser.add_argument('--wd',
                        default=1e-3,
                        type=float,
                        help='weight decay for optimizer')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--lr-decay',
                        default=0.2,
                        type=float,
                        help='learning rate decay factor: new = old * factor')
    parser.add_argument('--lr-step',
                        default=10,
                        type=int,
                        help='interval in nb of epochs to decay the learning rate')
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
    parser.add_argument('--wavelet',
                        default='db2',
                        type=str,
                        help='wavelet')
    parser.add_argument('--level',
                        default=0,
                        type=int,
                        help='nb of levels for wavelet analysis')
    parser.add_argument('--transforms',
                        default=[],
                        nargs='*',
                        type=str,
                        help='transformations on the input img')
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
    print(args)
    execute(args)
