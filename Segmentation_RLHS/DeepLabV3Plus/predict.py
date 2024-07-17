from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import numpy
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics
import cv2
import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

from PIL import Image
import numpy as np
from area import *

def overlay_segmentation_mask(img, mask, image_path,alpha=0.35):
    
    # Resize the mask to match the size of the image
    mask = mask.resize(img.size, resample=Image.NEAREST)

    # Convert images to NumPy arrays
    img_array = np.array(img)
    mask_array = np.array(mask)

    # Blend the image and the overlay
    blended_array = cv2.addWeighted(img_array, 1 - alpha, mask_array, alpha, 0)

    # Convert back to Image
    blended_image = Image.fromarray(blended_array)
    blended_image.save(image_path)

    return blended_image




    
    
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str,default ="D:\\RLHS-MITACS\\Data\\final_data",
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default='D:/RLHS-MITACS/Segmentation_RLHS/output_test',
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default='D:\\RLHS-MITACS\\Segmentation_RLHS\\DeepLabV3Plus\\checkpoints\\best_deeplabv3plus_mobilenet_cityscapes_os16.pth',
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def load_model():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)
        
    return model



model = load_model()

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(opts.save_val_results_to, img_name+'.png'))
            
            img1 = transform(img).unsqueeze(0) # To tensor of NCHW
            img1 = img1.to(device)
            
            pred = model(img1).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            
            colorized_preds = Image.fromarray(colorized_preds)
            
            calc_area_n(colorized_preds, os.path.join(opts.save_val_results_to, img_name+'_mask.png'))
            # colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'_mask.png'))
            
            # overlay_segmentation_mask(img,colorized_preds,os.path.join(opts.save_val_results_to, img_name+'_masked.png'))
            
            
            
def segmentation(img_path, model = model):
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # # Setup dataloader
    # # image_files = []
    # # if os.path.isdir(opts.input):
    # #     for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
    # #         files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
    # #         if len(files)>0:
    # #             image_files.extend(files)
    # # elif os.path.isfile(opts.input):
    # #     image_files.append(opts.input)
    
    # # Set up model (all models are 'constructed at network.modeling)
    # model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    #     # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
    #     checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint["model_state"])
    #     model = nn.DataParallel(model)
    #     model.to(device)
    #     print("Resume model from %s" % opts.ckpt)
    #     del checkpoint
    # else:
    #     print("[!] Retrain")
    #     model = nn.DataParallel(model)
    #     model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        
        ext = os.path.basename(img_path).split('.')[-1]
        img_name = os.path.basename(img_path)[:-len(ext)-1]
        img = Image.open(img_path).convert('RGB')
        # img.save(os.path.join(opts.save_val_results_to, img_name+'.png'))
            
        img1 = transform(img).unsqueeze(0) # To tensor of NCHW
        img1 = img1.to(device)
            
        pred = model(img1).max(1)[1].cpu().numpy()[0] # HW
        colorized_preds = decode_fn(pred).astype('uint8')
            
        colorized_preds = Image.fromarray(colorized_preds)
        veg_area, ter_area = calc_area_n(colorized_preds, os.path.join(opts.save_val_results_to, img_name+'_mask.png'), prt=False)
        return veg_area, ter_area         
    
    
            

if __name__ == '__main__':
    # veg_area, ter_area = segmentation("D:\\RLHS-MITACS\\Segmentation_RLHS\\dummy\\4.png",model)
    # print(veg_area, " ", ter_area)
    main()
