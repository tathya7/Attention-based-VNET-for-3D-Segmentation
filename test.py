import os
import sys
import time
import torch
import logging
import argparse
from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from medpy import metric
from tqdm import tqdm

from networks.vnet_AMC import VNet_AMC
from dataloader.AortaDissection import AortaDissection
from utils.train_util import set_random_seed, load_net_opt

def get_arguments():
 parser = argparse.ArgumentParser(description='Convert H5 to NPY and Generate Predictions for Specified Test Set')

 # Model
 parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')
 parser.add_argument('--resume', action='store_true')
 parser.add_argument('--load_path', type=str, default='results/TBAD/checkpoints')

 # dataset
 parser.add_argument("--data_dir", type=str, default='/home/amishr17/aryan/new_attempt/preprocess/our_TBAD/ImageTBAD', help="Path to the dataset.")
 parser.add_argument("--list_dir", type=str, default='datalist/AD/AD_1', help="Directory containing list of test files.")
 parser.add_argument("--list_file", type=str, default='test.txt', help="Name of the test list file.")
 parser.add_argument("--save_path", type=str, default=os.path.expanduser('~/results/test/'), help="Path to save predictions.")

 # Optimization options
 parser.add_argument('--lr', type=float, default=0.001, help='maximum epoch number to train')
 parser.add_argument('--beta1', type=float, default=0.5, help='params of optimizer Adam')
 parser.add_argument('--beta2', type=float, default=0.999, help='params of optimizer Adam')
 
 # Miscs
 parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
 parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')
 
 return parser.parse_args()

def create_model(args, ema=False):
 net = nn.DataParallel(VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4))
 model = net
 if ema:
    for param in model.parameters():
        param.detach_()
 return model

@torch.no_grad()
def test_AD(model, data_loader, args, print_result=False, maxdice=0):
    dc_list = []
    jc_list = []
    hd95_list = []
    asd_list = []

    time_start = time.time()
    total_num = len(data_loader)
    for i_batch, sampled_batch in tqdm(enumerate(data_loader)):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.type(torch.FloatTensor), label_batch.type(torch.LongTensor).cpu().numpy()

        out = model(volume_batch)
        outputs = (out[0] + out[1] + out[2] + out[3]) / 4
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1).cpu().numpy()

        for c in range(1, args.num_classes):
            pred_test_data_tr = outputs.copy()
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label_batch.copy()
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            score = cal_metric(pred_gt_data_tr, pred_test_data_tr)
            dc_list.append(score[0])
            jc_list.append(score[1])
            hd95_list.append(score[2])
            asd_list.append(score[3])

            if print_result:
                logging.info('\n[{}/{}] {}%:\t{}'.format(i_batch, total_num, round(i_batch/total_num*100), sampled_batch['name']))
                logging.info('Class: {}: dice {:.1f}% | jaccard {:.1f}% | hd95 {:.1f} | asd {:.1f}'
                .format(c, score[0]*100, score[1]*100, score[2], score[3]))

    time_end = time.time() 
    dc_arr = 100 * np.reshape(dc_list, [-1, 2]).transpose()
    jc_arr = 100 * np.reshape(jc_list, [-1, 2]).transpose()
    hd95_arr = np.reshape(hd95_list, [-1, 2]).transpose()
    asd_arr = np.reshape(asd_list, [-1, 2]).transpose()

    dice_mean = np.mean(dc_arr, axis=1)
    dice_std = np.std(dc_arr, axis=1)
    jc_mean = np.mean(jc_arr, axis=1)
    jc_std = np.std(jc_arr, axis=1)
    hd95_mean = np.mean(hd95_arr, axis=1)
    hd95_std = np.std(hd95_arr, axis=1)
    assd_mean = np.mean(asd_arr, axis=1)
    assd_std = np.std(asd_arr, axis=1)

    logging.info('Dice Mean: {}, Jaccard Mean: {}, Hd95 Mean: {}, Assd Mean: {}'
    .format(np.mean(dice_mean), np.mean(jc_mean), np.mean(hd95_mean), np.mean(assd_mean)))
    logging.info('Dice:')
    logging.info('FL :%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
    logging.info('TL :%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
    logging.info('Mean :%.1f' % np.mean(dice_mean))

    logging.info('\nJaccard:')
    logging.info('FL :%.1f(%.1f)' % (jc_mean[0], jc_std[0]))
    logging.info('TL :%.1f(%.1f)' % (jc_mean[1], jc_std[1]))
    logging.info('Mask :%.1f' % np.mean(jc_mean))

    logging.info('\nHD95:')
    logging.info('FL :%.1f(%.1f)' % (hd95_mean[0], hd95_std[0]))
    logging.info('TL :%.1f(%.1f)' % (hd95_mean[1], hd95_std[1]))
    logging.info('Mask :%.1f' % np.mean(hd95_mean))

    logging.info('\nASSD:')
    logging.info('FL :%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
    logging.info('TL :%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
    logging.info('Mask :%.1f' % np.mean(assd_mean))

    logging.info('Inference time: %.1f' % ((time_end-time_start)/total_num))

    val_dice = np.mean(dc_arr)
    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False
        logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f\n' % (val_dice, maxdice))

    return val_dice, maxdice, max_flag

# def cal_metric(gt, pred):
#     if pred.sum() > 0 and gt.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         jc = metric.binary.jc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         asd = metric.binary.asd(pred, gt)
#         return np.array([dice, jc, hd95, asd])
#     else:
#         return np.zeros(4)

def cal_metric(gt, pred):
    logging.info(f"GT sum: {gt.sum()}, Pred sum: {pred.sum()}")
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        logging.info(f"Metrics calculated: dice={dice}, jc={jc}, hd95={hd95}, asd={asd}")
        return np.array([dice, jc, hd95, asd])
    else:
        logging.info("Returning zeros due to empty prediction or ground truth")
        return np.zeros(4)

# def convert_h5_to_npy(h5_path, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     with h5py.File(h5_path, 'r') as h5_file:
#         image_data = h5_file['image'][()]
#         label_data = h5_file['label'][()]
 
#         base_name = os.path.splitext(os.path.basename(h5_path))[0]
#         image_save_path = os.path.join(output_dir, f'{base_name}-Image.npy')
#         label_save_path = os.path.join(output_dir, f'{base_name}-Label.npy')

#         np.save(image_save_path, image_data)
#         np.save(label_save_path, label_data)
#         logging.info(f"Saved {base_name}-Image.npy and {base_name}-Label.npy to {output_dir}")

def convert_h5_to_npy(h5_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with h5py.File(h5_path, 'r') as h5_file:
        image_data = h5_file['image'][()]
        label_data = h5_file['label'][()]
        
        # Zero out the label data
        label_data[:] = 0

        base_name = os.path.splitext(os.path.basename(h5_path))[0]
        image_save_path = os.path.join(output_dir, f'{base_name}-Image.npy')
        label_save_path = os.path.join(output_dir, f'{base_name}-Label.npy')

        np.save(image_save_path, image_data)
        np.save(label_save_path, label_data)
        logging.info(f"Saved {base_name}-Image.npy and {base_name}-Label.npy to {output_dir}")

def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as h5_file:
        image_data = h5_file['image'][()]
    return image_data

def save_predictions(model, image_data, save_path):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        model.eval()
        with torch.no_grad():
            image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            predictions = model(image_tensor)
 
            if isinstance(predictions, list):
                predictions = predictions[0]
 
            predictions = predictions.argmax(dim=1).squeeze().cpu().numpy()
        np.save(save_path, predictions)
        logging.info(f"Predictions saved to {save_path}")


def main():
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Create logger
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Save at: {}'.format(save_path))

    set_random_seed(args.seed)

    net = create_model(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    load_net_opt(net, optimizer, Path(args.load_path) / 'best_ema.pth')

    # Read the list of H5 files from the list_dir directory
    list_file_path = os.path.join(args.list_dir, args.list_file)
    with open(list_file_path, 'r') as file:
        h5_files = [line.strip() for line in file.readlines()]

    # Convert specified H5 files to NPY and generate predictions
    dataset_dir = args.data_dir
    output_dir = 'results/vis/ilnpy/'
    save_dir = 'results/vis/upcol/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for h5_file in h5_files:
        h5_path = os.path.join(dataset_dir, h5_file)
        if os.path.isfile(h5_path) and h5_path.endswith('.h5'):
            logging.info(f"Processing file: {h5_path}")
            convert_h5_to_npy(h5_path, output_dir)
            image_data = load_h5_data(h5_path)
            save_path = os.path.join(save_dir, f'{os.path.splitext(h5_file)[0]}_UPCoL.npy')
            save_predictions(net, image_data, save_path)
        else:
            logging.warning(f"File {h5_path} does not exist or is not a valid .h5 file")

    # Testing the converted and processed files
    testset = AortaDissection(args.data_dir, args.list_dir, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    test_AD(net, test_loader, args, print_result=True)

if __name__ == '__main__':
    main()
