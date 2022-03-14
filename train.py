import sys
sys.path.append('../')
import argparse

from ssd_data import datasets
from ssd_data import transforms, target_transforms, augmentations, utils

from ssd.models.ssd300 import SSD300
from ssd.train import *
from ssd_data._utils import DATA_ROOT

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

import tensorboardX as tbx

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/shuffle_custom',type=str)
parser.add_argument('--focus', type= str)
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()

augmentation = augmentations.AugmentationOriginal()

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
)
target_transform = target_transforms.Compose(
    [target_transforms.Corners2Centroids(),
     target_transforms.OneHot(class_nums=datasets.VOC_class_nums, add_background=True),
     target_transforms.ToTensor()]
)
# train_dataset = datasets.VOC2007_TrainValDataset(ignore=target_transforms.Ignore(difficult=True), transform=transform, target_transform=target_transform, augmentation=augmentation)
# train_dataset = datasets.Custom_TrainDataset(ignore=target_transforms.Ignore(difficult=True), transform=transform, target_transform=target_transform, augmentation=augmentation)
train_dataset = datasets.Custom_TrainDataset(ignore=target_transforms.Ignore(difficult=True), transform=transform, target_transform=target_transform, augmentation=augmentation,
                                             voc_dir=DATA_ROOT + '/shuffle_custom', focus=args.focus
                                             )

train_loader = DataLoader(train_dataset,
                          batch_size=16,
                          shuffle=True,
                          collate_fn=utils.batch_ind_fn,
                          num_workers=4,
                          pin_memory=True)
print('{} training images'.format(len(train_dataset)))


model = SSD300(class_labels=train_dataset.class_labels, batch_norm=False).cuda()
# model.load_vgg_weights()
print(model)
# model.load_weights('')



optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4) # slower
#optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4) # faster
iter_sheduler = SSDIterMultiStepLR(optimizer, milestones=(40000, 50000), gamma=0.1, verbose=True)

#save_manager = SaveManager(modelname='ssd300-voc2007++', interval=10, max_checkpoints=15, plot_yrange=(0, 8))#5000
# save_manager = SaveManager(modelname='ssd300-custom612', interval=4000, max_checkpoints=15, plot_yrange=(0, 8), savedir='./weights/res50pre1')
save_manager = SaveManager(modelname='ssd300-custom', interval=4000, max_checkpoints=15, plot_yrange=(0, 8), savedir=args.save_dir)
log_manager = LogManager(interval=10, save_manager=save_manager, loss_interval=10, live_graph=LiveGraph((0, 8)))
trainer = TrainLogger(model, loss_func=SSDLoss(), optimizer=optimizer, scheduler=iter_sheduler, log_manager=log_manager)

#trainer.train(70, train_loader)
trainer.train(7000, train_loader)
