import argparse
import os
import random
import time
from os.path import isfile, join, split

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import yaml

from src.dataloader_new import get_loader
from src.model.network import Net
from src.schedulers import get_scheduler
from src.optimizers import get_optimizer
from src.logger import Logger
from src.utils import reverse_mapping, caculate_precision, caculate_recall
from src.utils import save_checkpoint, get_lr, batch_grid, DayHourMinute


from skimage.measure import label, regionprops
from tensorboardX import SummaryWriter


WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ModuleNotFoundError:
    print('wandb is not installed')


def train(train_loader, model, optimizer, epoch, writer):
    # switch to train mode
    model.train()
    # torch.cuda.empty_cache()
    bar = tqdm(train_loader)
    iter_num = len(train_loader.dataset) // CONFIGS["DATA"]["BATCH_SIZE"]

    images_wdb = []
    total_loss_hough = 0
    for i, data in enumerate(bar):
        images, hough_space_label8, names = data

        if WANDB_AVAILABLE:
            if epoch==0 and i<3:
                image_grid = batch_grid(images)
                save_path = os.path.join(CONFIGS["MISC"]['WORK_DIR'], f'train_batch_{i}.png')
                torchvision.utils.save_image(image_grid, save_path)
                images_wdb.append(wandb.Image(save_path, caption=f'train_batch_{i}'))

        if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
            images = images.cuda()
            hough_space_label8 = hough_space_label8.cuda()
        else:
            images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            hough_space_label8 = hough_space_label8.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            
        keypoint_map = model(images)
        hough_space_loss = torch.nn.functional.binary_cross_entropy_with_logits(keypoint_map, hough_space_label8)

        writer.add_scalar('train/hough_space_loss', hough_space_loss.item(), epoch * iter_num + i)

        loss = hough_space_loss

        if not torch.isnan(hough_space_loss):
            total_loss_hough += hough_space_loss.item()
        else:
            logger.info("Warnning: loss is Nan.")

        #record loss
        bar.set_description('Epoch [{:3}/{}]'.format(epoch, CONFIGS["TRAIN"]['EPOCHS']))
        bar.set_postfix({'loss:' : round(loss.item(), 8)})
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % CONFIGS["TRAIN"]["PRINT_FREQ"] == 0:
            visualize_save_path = os.path.join(CONFIGS["MISC"]['WORK_DIR'], 'visualize', str(epoch))
            os.makedirs(visualize_save_path, exist_ok=True)
            
            # Do visualization.
            # torchvision.utils.save_image(torch.sigmoid(keypoint_map), join(visualize_save_path, 'rodon_'+names[0]), normalize=True)
            # torchvision.utils.save_image(torch.sum(vis, dim=1, keepdim=True), join(visualize_save_path, 'vis_'+names[0]), normalize=True)

    total_loss_hough = total_loss_hough / iter_num
    writer.add_scalar('train/total_loss_hough', total_loss_hough, epoch)
    return total_loss_hough, images_wdb
 
    
def validate(val_loader, model, epoch, writer):
    # switch to evaluate mode
    model.eval()
    total_acc = 0.0
    total_loss_hough = 0

    total_precision = np.zeros(99)
    total_recall = np.zeros(99)
    nums_precision = 0
    nums_recall = 0
    with torch.no_grad():
        images_wdb = []
        bar = tqdm(val_loader)
        iter_num = len(val_loader.dataset) // 1
        for i, data in enumerate(bar):

            images, hough_space_label8, gt_coords, names = data

            if WANDB_AVAILABLE:
                if epoch==0 and i<3:
                    image_grid = batch_grid(images)
                    save_path = os.path.join(CONFIGS["MISC"]['WORK_DIR'], f'valid_batch_{i}.png')
                    torchvision.utils.save_image(image_grid, save_path)
                    images_wdb.append(wandb.Image(save_path, caption=f'valid_batch_{i}'))

            if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
                images = images.cuda()
                hough_space_label8 = hough_space_label8.cuda()
            else:
                images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
                hough_space_label8 = hough_space_label8.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
                
            keypoint_map = model(images)

            hough_space_loss = torch.nn.functional.binary_cross_entropy_with_logits(keypoint_map, hough_space_label8)
            writer.add_scalar('val/hough_space_loss', hough_space_loss.item(), epoch * iter_num + i)

            acc = 0
            total_acc += acc

            loss = hough_space_loss
            if not torch.isnan(loss):
                total_loss_hough += loss.item()
            else:
                logger.info("Warnning: val loss is Nan.")

            key_points = torch.sigmoid(keypoint_map)
            binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = []
            for prop in props:
                plist.append(prop.centroid)
            b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], 
                                              numRho=CONFIGS["MODEL"]["NUMRHO"], 
                                              size=(400, 400))
            # [[y1, x1, y2, x2], [] ...]
            gt_coords = gt_coords[0].numpy().tolist()
            for i in range(1, 100):
                p, num_p = caculate_precision(b_points, gt_coords, thresh=i*0.01)
                r, num_r = caculate_recall(b_points, gt_coords, thresh=i*0.01)
                total_precision[i-1] += p
                total_recall[i-1] += r

            nums_precision += num_p
            nums_recall += num_r
            
        total_loss_hough = total_loss_hough / iter_num
        if nums_precision == 0:
            nums_precision = 0
        else:
            total_precision = total_precision / nums_precision
        if nums_recall == 0:
            total_recall = 0
        else:
            total_recall /= nums_recall

        writer.add_scalar('val/total_loss_hough', total_loss_hough, epoch)
        writer.add_scalar('val/total_precison', total_precision.mean(), epoch)
        writer.add_scalar('val/total_recall', total_recall.mean(), epoch)
        logger.info('Validation result: ==== Precision: %.5f, Recall: %.5f' % (total_precision.mean(), total_recall.mean()))
        acc = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-6)
        logger.info('Validation result: ==== F-score: %.5f' % acc.mean())
        writer.add_scalar('val/f-score', acc.mean(), epoch)
    return total_loss_hough, acc.mean(), total_precision.mean(), total_recall.mean(), images_wdb


def main():
    logger.info(args)

    if isinstance(CONFIGS["DATA"]["DIR"], list):
        assert all([os.path.isdir(dr) for dr in CONFIGS["DATA"]["DIR"]]) 
    else:
        assert os.path.isdir(CONFIGS["DATA"]["DIR"]) 

    wandb_run = None
    best_acc1 = 0

    if WANDB_AVAILABLE:
        wandb_run = wandb.init(project=f'RETECHLABS DHT shelves detection',
                                name= CONFIGS["MISC"]['RUN_NAME'],
                                reinit=True)

    if CONFIGS['TRAIN']['SEED'] is not None:
        random.seed(CONFIGS['TRAIN']['SEED'])
        torch.manual_seed(CONFIGS['TRAIN']['SEED'])
        cudnn.deterministic = True

    model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], 
                numRho=CONFIGS["MODEL"]["NUMRHO"], 
                backbone=CONFIGS["MODEL"]["BACKBONE"])
    
    if CONFIGS["TRAIN"]["FREEZE_BACKBONE"]:
        for param in model.backbone.parameters():
            param.requires_grad = False

    if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
        logger.info("Model Data Parallel")
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    # optimizer
    optimizer = get_optimizer(model, CONFIGS["OPTIMIZER"])

    # learning rate scheduler
    scheduler = get_scheduler(optimizer, CONFIGS["OPTIMIZER"])

    if CONFIGS["TRAIN"]["LOAD_WEIGHTS"]:
        if isfile(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]):
            logger.info("=> loading checkpoint '{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
            checkpoint = torch.load(CONFIGS["TRAIN"]["LOAD_WEIGHTS"])
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> start from checkpoint '{}'"
                  .format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))
        else:
            logger.info("=> no checkpoint found at '{}'".format(CONFIGS["TRAIN"]["LOAD_WEIGHTS"]))

    if args.resume:
        if isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    if isinstance(CONFIGS["DATA"]["SPLIT_FILE"], list):
        df_train = []
        df_valid = []
        for sf in CONFIGS["DATA"]["SPLIT_FILE"]:
            df_folds = pd.read_csv(sf)
            df_train.append(df_folds[df_folds.fold != CONFIGS["DATA"]["FOLD"]])
            df_valid.append(df_folds[df_folds.fold == CONFIGS["DATA"]["FOLD"]])
    else:
        df_folds = pd.read_csv(CONFIGS["DATA"]["SPLIT_FILE"])
        df_train = df_folds[df_folds.fold != CONFIGS["DATA"]["FOLD"]]
        df_valid = df_folds[df_folds.fold == CONFIGS["DATA"]["FOLD"]]

    # dataloader
    train_loader = get_loader(CONFIGS["DATA"]["DIR"], 
                              df_train, 
                              batch_size=CONFIGS["DATA"]["BATCH_SIZE"], 
                              num_thread=CONFIGS["DATA"]["WORKERS"], 
                              split='train', 
                              transform_name=CONFIGS['TRAIN']['AUG_TYPE'])
    val_loader = get_loader(CONFIGS["DATA"]["DIR"], 
                            df_valid, 
                            batch_size=1, 
                            num_thread=CONFIGS["DATA"]["WORKERS"], 
                            split='val')

    logger.info("Data loading done.")

    # Tensorboard summary

    writer = SummaryWriter(log_dir=os.path.join(CONFIGS["MISC"]['WORK_DIR']))

    start_epoch = 0
    best_acc = best_acc1
    is_best = False
    start_time = time.time()

    if CONFIGS["TRAIN"]["RESUME"] is not None:
        raise(NotImplementedError)
    
    if CONFIGS["TRAIN"]["TEST"]:
        validate(val_loader, model, 0, writer, args)
        return

    logger.info("Start training.")

    for epoch in range(start_epoch, CONFIGS["TRAIN"]["EPOCHS"]):
        
        train_loss, images_wdb_train = train(train_loader, model, optimizer, epoch, writer)
        valid_loss, valid_f1, valid_precision, valid_recall, images_wdb_valid = validate(val_loader, model, epoch, writer)

        metrics = {
            'train_loss' : train_loss,
            'valid_loss' : valid_loss,
            'valid_acc_f1' : valid_f1,
            'valid_precision' : valid_precision,
            'valid_recall' : valid_recall,
            'lr' : get_lr(optimizer)
        }

        if images_wdb_train and images_wdb_valid:
            metrics["training batch"] = images_wdb_train
            metrics["validation batch"] = images_wdb_valid

        scheduler.step()

        if WANDB_AVAILABLE:
            wandb.log(metrics, step=epoch)

        is_best = best_acc < valid_f1
        best_acc = valid_f1 if is_best else best_acc

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer' : optimizer.state_dict()
            }, is_best, path=CONFIGS["MISC"]['WORK_DIR'])

        t = time.time() - start_time       
        elapsed = DayHourMinute(t)
        t /= (epoch + 1) - start_epoch    # seconds per epoch
        t = (CONFIGS["TRAIN"]["EPOCHS"] - epoch - 1) * t
        remaining = DayHourMinute(t)
        
        logger.info("Epoch {0}/{1} finishied, auxiliaries saved to {2} .\t"
                    "Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes.\t"
                    "Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.".format(
                    epoch, CONFIGS["TRAIN"]["EPOCHS"], CONFIGS["MISC"]['WORK_DIR'], elapsed=elapsed, remaining=remaining))

    logger.info("Optimization done, ALL results saved to %s." % CONFIGS["MISC"]['WORK_DIR'])

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
    # arguments from command line
    parser.add_argument('--config', default="./configs/default_config.yml", help="path to config file")
    parser.add_argument('--resume', default="", help='path to config file')
    parser.add_argument('--tmp', default="", help='tmp')

    args = parser.parse_args()

    assert os.path.isfile(args.config)
    CONFIGS = yaml.safe_load(open(args.config))

    # merge configs
    if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
        CONFIGS["MISC"]["TMP"] = args.tmp

    CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"] = float(CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"])
    CONFIGS["OPTIMIZER"]["LR"] = float(CONFIGS["OPTIMIZER"]["LR"])

    CONFIGS["MISC"]['RUN_NAME'] = '{}_{}_fold{}'.format(CONFIGS['MISC']['RETAILER'],
                                                    CONFIGS['MODEL']['BACKBONE'],
                                                    CONFIGS["DATA"]["FOLD"])
    CONFIGS["MISC"]['WORK_DIR'] = os.path.join(CONFIGS["MISC"]["TMP"], CONFIGS["MISC"]['RUN_NAME'])
    os.makedirs(CONFIGS["MISC"]['WORK_DIR'], exist_ok=True)
    logger = Logger(os.path.join(CONFIGS["MISC"]['WORK_DIR'], "log.txt"))



    logger.info(CONFIGS)
    
    main()
