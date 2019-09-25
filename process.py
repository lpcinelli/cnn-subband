import os
import time

import numpy as np
import torch
from utils import AverageMeter, accuracy


def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, device):

    print_freq = 100

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, labels) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute output
        output = model(inputs)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))

    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}


def validate(val_loader, model, criterion, device=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):

            inputs = inputs.to(device)
            targets = targets.to(device)

            # compute output
            output = model(inputs)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(
            top1=top1, top5=top5, loss=losses))

    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}


def fit(train_loader,
        val_loader,
        model,
        criterion,
        epochs,
        device,
        optimizer,
        scheduler,
        savepath=None,
        start_epoch=0,
        history=None):
    if savepath is not None:
        path, ext = os.path.splitext(savepath)
        savepath = path + '-{}' + ext

    criterion = criterion.to(device)
    best_prec1 = 0

    if history is None:
        history = {
            'train': {
                'loss': [],
                'acc1': [],
                'acc5': []
            },
            'val': {
                'loss': [],
                'acc1': [],
                'acc5': []
            }
        }

    end = time.time()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # loop over the dataset multiple times
    for epoch in range(start_epoch, epochs):

        # train for one epoch
        print("\nBegin Training Epoch {}".format(epoch + 1))
        stats = train(train_loader, model, criterion, optimizer, epoch, device)
        for name, val in stats.items():
            history['train'][name].append(val)

        # evaluate on validation set
        print("Begin Validation @ Epoch {}".format(epoch + 1))
        stats = validate(val_loader, model, criterion, device)
        for name, val in stats.items():
            history['val'][name].append(val)

        prec1 = stats['acc1']

        # Adjust learning rate according to schedule
        scheduler.step()

        if savepath is not None:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                }, savepath.format('last'))

            if prec1 > best_prec1:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'history': history,
                    }, savepath.format('best'))

        # remember best prec@1 and save checkpoint if desired
        # is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print("Epoch Summary: ")
        print("\tEpoch Accuracy: {}".format(prec1))
        print("\tBest Accuracy: {}".format(best_prec1))
    print("Total training time: {:.1f}".format(time.time() - end))


def evaluate(test_loader, model, device, savepath):

    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    top5 = []

    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(test_loader):

            inputs = inputs.to(device)
            target = target.to(device)

            # compute output
            output = model(inputs)

            # the top5 class predictions in descending order
            _, pred = output.topk(5, 1, True, True)
            top5.append(pred.cpu().numpy().astype(int))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.
                      format(i, len(test_loader), batch_time=batch_time))

    # pdb.set_trace()
    top5 = np.concatenate(top5, 0)
    top5 = [','.join([test_loder.dataset.classes[pred] for pred in preds]) for preds in top5]
    if savepath is not None:
        savepath = os.path.join(os.path.dirname(savepath), 'predictions.csv')
    else:
        savepath = '/dev/stdout'
    np.savetxt(savepath, top5, fmt='%s')
