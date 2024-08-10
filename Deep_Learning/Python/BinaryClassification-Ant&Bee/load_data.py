import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")


class MyDataset(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_path = os.path.join(root_dir, image_dir)
        self.label_path = os.path.join(root_dir, label_dir)

        self.image_list = sorted(os.listdir(self.image_path))
        self.label_list = sorted(os.listdir(self.label_path))
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.image_list[index]
        label_name = self.label_list[index]
        img_item_path = os.path.join(self.image_path, img_name)
        label_item_path = os.path.join(self.label_path, label_name)
        img = Image.open(img_item_path)

        with open(label_item_path, 'rb') as f:
            label = f.readline()

        img = self.transform(img)
        return {'img': img, 'label': label}

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()])
    root_dir = ("/Users/zhenxiangjin/Projects/Machine_Learning/Data"
                "/cv_binary_classification/hymenoptera_data")
    image_ants = "train/ants_image"
    label_ants = "train/ants_label"
    ants_dataset = MyDataset(root_dir, image_ants, label_ants, transform)
    image_bees = "train/bees_image"
    label_bees = "train/bees_label"
    bees_dataset = MyDataset(root_dir, image_bees, label_bees, transform)
    train_dataset = ants_dataset + bees_dataset

    # transforms = transforms.Compose([transforms.Resize(256, 256)])
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    writer.add_image('error', train_dataset[119]['img'])
    writer.close()
