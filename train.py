import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_loaders.ShapeNet import PartNormalDataset
from data_loaders import data_augmentation
from modules.utils import onehot, inplace_relu, log_string, weights_init, update_lr_bn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # root dir
sys.path.append(os.path.join(BASE_DIR, 'models'))

# python train.py  --use_normal --log_dir PointNetPP --device "cpu"

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='PointNetPP', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    # modified option for non-gpu devices
    parser.add_argument('--device', type=str, default='mps', help='specify device: cpu, mps, cuda:0')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--n_point', type=int, default=2048, help='point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def train(model, dataloader, optim, loss_func, num_classes, num_part, logger, mean_correct):
    for i, (points, cls, target) in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
        # tuple (idx, (points, cls, target))
        optim.zero_grad()

        # Data augmentation ------
        # TODO : torchify the following operations performed in Numpy
        points = points.data.numpy()
        points[:, :, 0:3] = data_augmentation.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = data_augmentation.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)

        # batch * { n_points * (3 + 3 * use_normals), 1, n_points * 1}
        points, cls, target = points.float().to(args.device), cls.long().to(args.device), target.long().to(
            args.device)
        points = points.transpose(2, 1)
        seg_pred = model(points, onehot(cls, num_classes))  # , trans_feat
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        # tensor.view() requires data to be contiguous, i.e. be consecutive in a memory block
        # more on torch.contiguous : https://zhuanlan.zhihu.com/p/64551412

        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct.append(correct.item() / (args.batch_size * args.n_point))
        loss = loss_func(seg_pred, target)  # trans_feat
        loss.backward()
        optim.step()
    train_instance_acc = np.mean(mean_correct)
    log_string(logger, 'Train accuracy is: %.5f' % train_instance_acc)
    return train_instance_acc


def evaluation(model, dataloader, num_classes, num_part, logger):

    test_metrics = {}
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(num_part)]
    total_correct_class = [0 for _ in range(num_part)]
    shape_ious = {ct: [] for ct in seg_classes.keys()}

    for batch_id, (points, cls, target) in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
        cur_batch_size, n_points, _ = points.size()
        points, cls, target = points.float().to(args.device), cls.long().to(args.device), target.long().to(
            args.device)
        points = points.transpose(2, 1)
        seg_pred = model(points, onehot(cls, num_classes))
        cur_pred_val = seg_pred.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, n_points)).astype(np.int32)
        target = target.cpu().data.numpy()

        for i in range(cur_batch_size):
            category = seg_label_to_cat[target[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[category]], 1) + seg_classes[category][0]

        correct = np.sum(cur_pred_val == target)
        total_correct += correct
        total_seen += (cur_batch_size * n_points)

        for l in range(num_part):
            total_seen_class[l] += np.sum(target == l)
            total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i, :]
            segl = target[i, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (
                        np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            log_string(logger, 'eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)
        return test_metrics


def main(args):

    # CREATE LOG DIR
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(time_str)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # LOG
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)

    # statics vars
    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    num_classes = 16
    num_part = 50

    # load data
    train_dataset = PartNormalDataset(seg_classes=seg_classes, root=root, n_points=args.n_point, split='trainval', use_normals=args.use_normals)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_dataset = PartNormalDataset(seg_classes=seg_classes, root=root, n_points=args.n_point, split='test', use_normals=args.use_normals)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    log_string(logger, "The number of training data is: %d" % len(train_dataset))
    log_string(logger, "The number of test data is: %d" % len(test_dataset))

    # load model
    model_obj = importlib.import_module(args.model)
    segmentor = model_obj.PointNetPP(num_part, use_normals=args.use_normals).to(args.device)
    criterion = model_obj.get_loss().to(args.device)
    segmentor.apply(inplace_relu)

    # load weight or random init
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        segmentor.load_state_dict(checkpoint['model_state_dict'])
        log_string(logger, 'Use pretrained model')

    except:
        log_string(logger, "No weights provided of found, start training from scratch")
        start_epoch = 0
        segmentor = segmentor.apply(weights_init)

    # set optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            segmentor.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(segmentor.parameters(), lr=args.learning_rate, momentum=0.9)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_instance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):

        mean_correct = []
        log_string(logger, 'Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        # Adjust learning rate and BN momentum
        # segmentor = update_lr_bn(optimizer, segmentor, logger, args.learning_rate, args.lr_decay, epoch)

        # training ----------------------------------------------------------------------------------------------------
        segmentor = segmentor.train()
        train_instance_acc = train(segmentor, train_dataloader, optimizer, criterion, num_classes, num_part, logger, mean_correct)

        # evaluation ------------------------------------------------------------------------------------------------
        with torch.no_grad():
            segmentor = segmentor.eval()
            test_metrics = evaluation(model=segmentor, dataloader=test_dataloader, num_classes=num_classes,
                                      num_part=num_part, logger=logger)

        # Write results to log ---------------------------------------------------------------------------
        log_string(logger, 'Epoch %d test Accuracy: %f  Class avg mIOU: %f  Instance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['instance_avg_iou']))

        # save best model weights
        if test_metrics['instance_avg_iou'] >= best_instance_avg_iou:
            logger.info('Save model...')
            save_path = str(checkpoints_dir) + '/best_model.pth'
            log_string(logger, 'Saving at %s' % save_path)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'model_state_dict': segmentor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)
            log_string(logger, 'Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['instance_avg_iou'] > best_instance_avg_iou:
            best_instance_avg_iou = test_metrics['instance_avg_iou']
        log_string(logger, 'Best accuracy is: %.5f' % best_acc)
        log_string(logger, 'Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string(logger, 'Best instance avg mIOU is: %.5f' % best_instance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
