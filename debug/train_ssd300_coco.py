from ssd_data import transforms, target_transforms, augmentations, utils, datasets

from ssd.models.ssd300 import SSD300
from ssd.train import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
import torch

if __name__ == '__main__':
    """
    augmentation = augmentations.Compose(
        []
    )"""
    augmentation = augmentations.AugmentationOriginal()
    #augmentation = None

    transform = transforms.Compose(
        [transforms.Resize((300, 300)),
         transforms.ToTensor(),
         transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Corners2Centroids(),
         target_transforms.OneHot(class_nums=datasets.COCO_class_nums, add_background=True),
         target_transforms.ToTensor()]
    )

    train_dataset = datasets.COCO2014_TrainValDataset(ignore=target_transforms.Ignore(iscrowd=True), transform=transform, target_transform=target_transform, augmentation=augmentation)

    #train_dataset = datasets.VOC2007Dataset(transform=transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              collate_fn=utils.batch_ind_fn,
                              num_workers=4,
                              pin_memory=True)

    model = SSD300(class_labels=train_dataset.class_labels, batch_norm=False).cuda()
    model.load_vgg_weights()
    print(model)
    """
    imgs, targets = utils.batch_ind_fn((train_dataset[2000],))
    p, d = model(imgs)
    from ssd.core.boxes import matching_strategy
    matching_strategy(targets, d, batch_num=1)
    """
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    #optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    #iter_sheduler = SSDIterMultiStepLR(optimizer, milestones=(10, 20, 30), gamma=0.1, verbose=True)
    iter_sheduler = SSDIterStepLR(optimizer, step_size=60000, gamma=0.1, verbose=True)
    """
    save_manager = SaveManager(modelname='ssd300', interval=10, max_checkpoints=3)
    log_manager = LogManager(interval=10, save_manager=save_manager, loss_interval=10, live_graph=None)
    trainer = TrainLogger(model, loss_func=SSDLoss(), optimizer=optimizer, scheduler=iter_sheduler, log_manager=log_manager, gpu=True)

    trainer.train(30, train_loader)
    """
    save_manager = SaveManager(modelname='ssd300', interval=5000, max_checkpoints=3)
    log_manager = LogManager(interval=10, save_manager=save_manager, loss_interval=10, live_graph=None)
    trainer = TrainLogger(model, loss_func=SSDLoss(), optimizer=optimizer, scheduler=iter_sheduler,
                          log_manager=log_manager)

    trainer.train(80000, train_loader)#, evaluator=VOC2007Evaluator(val_dataset, iteration_interval=10))
