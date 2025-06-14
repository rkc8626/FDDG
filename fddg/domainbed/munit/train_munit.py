import tensorboardX
import argparse

from core.utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
from core.trainer import *
from datasets import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.backends.cudnn as cudnn
import torch
import os
import sys
import shutil

import time

import torch.distributed as dist

startTime = time.time()

parser = argparse.ArgumentParser(description='PyTorch training')
parser.add_argument('--config', type=str, default='core/tiny_munit.yaml',
                        help='Path to the MUNIT config file.')
parser.add_argument('--output_path', type=str, default='/home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new',
                        help="Path where images/checkpoints will be saved")
parser.add_argument("--resume", action="store_true",
                        help='Resumes training from last avaiable checkpoint')
parser.add_argument('--dataset', type=str, default='CCMNIST1',
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--batch', type=int, default=1,
                        help="Batch size")
parser.add_argument('--env', type=int, default=0,
                        help="env not including")
parser.add_argument('--step', type=int, default=12,
                        help="training step 1 or 2 or cotraining:12")
parser.add_argument('--input_path', type=str, default='/home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/PDDPerson/pretrain_env0_step1/outputs/tiny_munit/checkpoints',
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--input_path1', type=str, default='/home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/PDDPerson/pretrain_env0_step1/outputs/tiny_munit/checkpoints',
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--input_path2', type=str, default='/home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/PDDPerson/pretrain_env0_step2/outputs/tiny_munit/checkpoints',
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--device', type=str, default='0,1,2,3',
                        help="CUDA DEVICE")

args = parser.parse_args()

# Set CUDA device before any CUDA operations
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.cuda.init()  # Initialize CUDA explicitly

# Verify CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU installation.")

def print_gpu_memory():
    print("\n=== GPU Memory Usage ===")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    print("======================\n")

# Print GPU information
print(f"\n=== GPU Information ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print(f"CUDA visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
print_gpu_memory()
print("======================\n")

cudnn.benchmark = True

# Load experiment setting
config = get_config(args.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = args.output_path
config['gen']['dataset'] = args.dataset
config['dis']['dataset'] = args.dataset

# Setup model and data loader
device = torch.device('cuda')
os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

# Initialize models with DataParallel
if args.step == 1:
    trainer1 = MUNIT_Trainer1(config)
    trainer1 = torch.nn.DataParallel(trainer1, device_ids=list(range(torch.cuda.device_count())))
elif args.step == 2:
    trainer1 = MUNIT_Trainer1(config)
    trainer2 = MUNIT_Trainer2(config)
    trainer1 = torch.nn.DataParallel(trainer1, device_ids=list(range(torch.cuda.device_count())))
    trainer2 = torch.nn.DataParallel(trainer2, device_ids=list(range(torch.cuda.device_count())))
    trainer2.to(device)
elif args.step == 12:
    trainer1 = MUNIT_Trainer(config)
    trainer1 = torch.nn.DataParallel(trainer1, device_ids=list(range(torch.cuda.device_count())))
trainer1.to(device)


if args.dataset == "CCMNIST1":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_CCMNIST1_loaders(args.env, args.batch)
elif args.dataset == "FairFace":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_FairFace_loaders(args.env, args.batch)
elif args.dataset == "YFCC":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_YFCC_loaders(args.env, args.batch)
elif args.dataset == "NYPD":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_NYPD_loaders(args.env, args.batch)
elif args.dataset == "BDDPerson":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_BDDPersons_loaders(args.env, args.batch, config)


train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[-1-i][0] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in range(display_size)]).cuda()




# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(args.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(args.output_path + "/logs", model_name))
output_directory = os.path.join(args.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(args.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
if args.step == 1:
    if args.resume:
        try:
            iterations = trainer1.module.resume(checkpoint_directory, hyperparameters=config)
        except IndexError:
            print("No checkpoint found. Starting training from scratch.")
            iterations = 0
    else:
        iterations = 0
    trainer = trainer1
elif args.step == 12:
    if args.resume:
        try:
            iterations = trainer1.module.resume(checkpoint_directory, hyperparameters=config)
        except IndexError:
            print("No checkpoint found. Starting training from scratch.")
            iterations = 0
    else:
        iterations = 0
    trainer = trainer1
elif args.step == 2:
    try:
        _ = trainer1.module.resume(args.input_path, hyperparameters=config)
    except IndexError:
        print("No checkpoint found in input_path. Please ensure checkpoint exists.")
        sys.exit(1)
    if args.resume:
        try:
            iterations = trainer2.module.resume(checkpoint_directory, hyperparameters=config)
        except IndexError:
            print("No checkpoint found. Starting training from scratch.")
            iterations = 0
    else:
        iterations = 0
    trainer = trainer2
while True:
    for it, (a, b) in enumerate(zip(train_loader_a, train_loader_b)):
        images_a, y_a, z_a = a
        images_b, y_b, z_b = b
        trainer.module.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        y_a, y_b = y_a.cuda(), y_b.cuda()
        z_a, z_b = z_a.cuda(), z_b.cuda()

        # with Timer("Elapsed time in update: %f"):
            # Main training code
        if args.step == 1:
            trainer.module.dis_update(images_a, images_b, config)
            trainer.module.gen_update(images_a, images_b, config)
        elif args.step == 12:
            trainer.module.dis_update(images_a, images_b, config)
            trainer.module.gen_update(images_a, z_a, images_b, z_b, config)
        elif args.step == 2:
            ca_a1, s_a1 = trainer1.module.gen_a.encode(images_a)
            ca_b1, s_b1 = trainer1.module.gen_b.encode(images_b)
            ca_a2, s_a2 = trainer1.module.gen_a.encode(images_a)
            ca_b2, s_b2 = trainer1.module.gen_b.encode(images_b)
            trainer.module.dis_update(ca_a1, ca_b1, config)
            trainer.module.gen_update(ca_a2, z_a, ca_b2, z_b, config)
        torch.cuda.synchronize()
        write_loss(iterations, trainer, train_writer)


        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            train_writer.flush()

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0 and args.dataset != 'NYPD':
            with torch.no_grad():
                if args.step == 1 or args.step == 12:
                    test_image_outputs = trainer.module.sample(test_display_images_a, test_display_images_b)
                elif args.step == 2:
                    test_image_outputs = trainer1.module.sample2(test_display_images_a, test_display_images_b, trainer)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            image_save_path = os.path.join(image_directory, 'test_%08d' % (iterations + 1))
            print(f"Saved images to: {image_save_path}")
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
            print_gpu_memory()

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.module.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

train_writer.close()
endTime = time.time()

print("Running Time:", (endTime-startTime)/3600, "hours")

print(torch.cuda.is_available())