import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from utils.cutout import SLCutoutPIL
# from randaugment import RandAugment
import os.path as osp
from lib.dataset.customDataset_only import CustomDataset_csv_multiLabel


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            # RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]

    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )    
    elif args.dataname =='custom':
        dataset_dir = args.dataset_dir
        print('args.dataset_dir', args.dataset_dir)
        if args.resume: 
            print('resume')
            val_dataset = CustomDataset_csv_multiLabel(
            sourcePath=osp.join(dataset_dir, 'image/test'),
            transform=test_data_transform,
            csv_file=osp.join(dataset_dir, 'annotation/VQIS-TEST.csv')
            )
            print("len(val_dataset):", len(val_dataset))
            return None, val_dataset 
        else:
            train_data_transform_list.insert(1, transforms.RandomHorizontalFlip())
            train_dataset = CustomDataset_csv_multiLabel(
                sourcePath=osp.join(dataset_dir, 'train'),
                transform=train_data_transform,
                # csv_file=osp.join(dataset_dir, 'annotation/VQIS-TRAIN.csv')
            )
            val_dataset = CustomDataset_csv_multiLabel(
                sourcePath=osp.join(dataset_dir, 'val'),
                transform=test_data_transform,
                # csv_file=osp.join(dataset_dir, 'annotation/VQIS-VAL.csv')
            )

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
