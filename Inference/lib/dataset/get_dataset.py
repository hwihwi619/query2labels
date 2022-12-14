import torchvision.transforms as transforms
import os.path as osp
from lib.dataset.customDataset import CustomDataset_csv_multiLabel


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
                                               transforms.ToTensor(),
                                               normalize]

    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname =='custom':
        dataset_dir = args.dataset_dir
        target_map=args.target.split(",")
        print('args.dataset_dir', args.dataset_dir)
    
        val_dataset = CustomDataset_csv_multiLabel(
        # sourcePath=osp.join(dataset_dir, 'image/test'),
        sourcePath=dataset_dir,
        target=target_map,
        transform=test_data_transform,
        )

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(val_dataset):", len(val_dataset))
    return None, val_dataset
