import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from sklearn.svm import SVC

import imp
import logging
from path import Path
import numpy as np
import time
import os
import sys
import importlib
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
from datasets.modelnet import ModelNet
from datasets.fewmodelnet import fewModelNet

def main(args):
    # load network
    print("loading module")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("models."+args.model)
    model = module.VoxNet(num_classes=args.num_classes, input_shape=(32,32,32))
    model.to(device)

    # backup files
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    os.system('cp {} {}'.format(os.path.join(ROOT_DIR, 'models', args.model+'.py'), args.log_dir))
    os.system('cp {} {}'.format(__file__, args.log_dir))
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    #logging.info('logs will be saved to {}'.format(args.log_fname))
    #logger = Logger(args.log_fname, reinitialize=True)
    print("loading dataset")
    dset_train = fewModelNet(os.path.join(ROOT_DIR, "data"), args.training_fname)
    dset_test = ModelNet(os.path.join(ROOT_DIR, "data"), args.testing_fname)
    
    # set shuffle to false since the label is not shuffle
    # drop_last = False so every instance will be loaded
    # batch_size = 1
    print(args.batch_size)
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, num_workers=4, drop_last=False)
    test_loader = DataLoader(dset_test, batch_size=args.batch_size, num_workers=4, drop_last=False)
    
    global LOG_FOUT
    LOG_FOUT = open(os.path.join(args.log_dir, 'log.txt'), 'w')
    log_string(args)

    start_epoch = 0
    best_acc = 0.
    if args.cont:
        start_epoch, best_acc = load_checkpoint(args, model)
    
    train_feature = np.zeros((len(dset_train), 128))
    train_label = np.zeros((len(dset_train),), dtype=np.int)
    test_feature = np.zeros((len(dset_test), 128))
    test_label = np.zeros((len(dset_test),), dtype=np.int)
    
    model.eval()
    # make sure batch size = 1
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # compute output
            outputs = model(inputs)
            train_feature[i] = outputs
            train_label[i] = targets

        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # compute output
            outputs = model(inputs)
            test_feature[i] = outputs
            test_label[i] = targets


    #print(train_feature[-1])
    #print(train_label[-1])
    #print(test_feature[-1])
    #print(test_label[-1])

    np.save(os.path.join(args.log_dir,"train_feature"), train_feature)
    np.save(os.path.join(args.log_dir,"train_label"), train_label)
    np.save(os.path.join(args.log_dir,"test_feature"), test_feature)
    np.save(os.path.join(args.log_dir,"test_label"), test_label)
    
    svm = SVC(kernel='linear')
    svm.fit(train_feature, train_label)
    train_pred = svm.predict(train_feature)
    test_pred = svm.predict(test_feature)
    train_acc = np.sum(train_pred == train_label) * 1.0 / train_feature.shape[0]
    test_acc = np.sum(test_pred == test_label) * 1.0 / test_feature.shape[0]
    log_string(str(train_acc))
    log_string(str(test_acc))

    with open("log_acc.txt", "a") as f:
        f.write("{}\n".format(test_acc))

    #print("set optimizer")
    # set optimization methods
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, args.decay_rate)

    #for epoch in range(start_epoch, args.max_epoch):
    #    scheduler.step()
    #    log_string('\n-----------------------------------')
    #    log_string('Epoch: [%d/%d]' % (epoch+1, args.max_epoch))
    #    start = time.time()

    #    model.train()
    #    train(train_loader, model, criterion, optimizer, device)
    #    log_string('Time taken: %.2f sec.' % (time.time() - start))

    #    model.eval()
    #    avg_test_acc, avg_loss = test(test_loader, model, criterion, optimizer, device)

    #    log_string('\nEvaluation:')
    #    log_string('\tVal Acc: %.2f - Loss: %.4f' % (avg_test_acc, avg_loss))
    #    log_string('\tCurrent best val acc: %.2f' % best_acc)

    #    # Log epoch to tensorboard
    #    # See log using: tensorboard --logdir='logs' --port=6006
    #    #util.logEpoch(logger, resnet, epoch + 1, avg_loss, avg_test_acc)

    #    # Save model
    #    if avg_test_acc > best_acc:
    #        log_string('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
    #        best_acc = avg_test_acc
    #        best_loss = avg_loss
    #        torch.save({
    #            'epoch': epoch + 1,
    #            #'state_dict': resnet.state_dict(),
    #            'body': model.body.state_dict(),
    #            'feat': model.head.state_dict(),
    #            'acc': avg_test_acc,
    #            'best_acc': best_acc,
    #            'optimizer': optimizer.state_dict()
    #        }, os.path.join(args.log_dir, args.saved_fname+".pth.tar"))

    LOG_FOUT.close()
    return

    
def log_string(out_str):
    LOG_FOUT.write(str(out_str)+'\n')
    LOG_FOUT.flush()
    print(out_str)


def load_checkpoint(args, model):
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    fname = os.path.join(args.ckpt_dir, args.ckpt_fname + '.pth.tar')
    assert os.path.isfile(fname), 'Error: no checkpoint file found!'

    checkpoint = torch.load(fname)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.body.load_state_dict(checkpoint['body'])
    # not loading all the weight for head
    new_head_dict = model.head.state_dict()

    original_head_dict = checkpoint['feat']
    for k in original_head_dict:
        if k in new_head_dict:
            print("same weight:", k)
        else:
            print("discarded weight:", k)

    original_head_dict = {k: v for k, v in original_head_dict.items() if k in new_head_dict}
    new_head_dict.update(original_head_dict)
    model.head.load_state_dict(new_head_dict)

    return start_epoch, best_acc


def train(loader, model, criterion, optimizer, device):
    num_batch = len(loader)
    batch_size = loader.batch_size
    total = torch.LongTensor([0])
    correct = torch.LongTensor([0])
    total_loss = 0.

    for i, (inputs, targets) in enumerate(loader):
        #inputs = torch.from_numpy(inputs)
        inputs, targets = inputs.to(device), targets.to(device)
        # in 0.4.0 variable and tensor are merged
        #inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = torch.max(outputs.detach(), 1)
        total += batch_size
        correct += (predicted == targets).cpu().sum()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_iter = 1000
        if (i + 1) % log_iter == 0:
            log_string("\tIter [%d/%d] Loss: %.4f" % (i + 1, num_batch, total_loss/log_iter))
            total_loss = 0.

    log_string("Train Accuracy %.2f" % (100.0 * correct.item() / total.item()))
    return


def test(loader, model, criterion, optimizer, device):
    # Eval
    total = torch.LongTensor([0])
    correct = torch.LongTensor([0])

    total_loss = 0.0
    n = 0

    for i, (inputs, targets) in enumerate(loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            #inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n += 1

            _, predicted = torch.max(outputs.detach(), 1)
            total += targets.size(0)
            correct += (predicted == targets).cpu().sum()

    avg_test_acc = 100. * correct.item() / total.item()
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_fname', type=Path, help='training .tar file')
    parser.add_argument('--testing_fname', type=Path, help='testing .tar file')
    parser.add_argument('--model', default='voxnet', help='Model name: [default:voxnet]')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--num_classes', type=int, default=40, help='Category Number [10/30/40] [default: 40]')
    parser.add_argument('--max_epoch', type=int, default=256, help='Epoch to run [default: 256]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=16, help='Decay step for lr decay [default: 16]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--saved_fname', type=Path, default=None, help='name of weight to be saved')
    parser.add_argument('--cont', action='store_true', default=False)
    parser.add_argument('--ckpt_dir', default='log', help='check point dir [default: log]')
    parser.add_argument('--ckpt_fname', default='model', help='check point name [default: model]')
    args = parser.parse_args()

    cudnn.benchmark = True
    main(args)
