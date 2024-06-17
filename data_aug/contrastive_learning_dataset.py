from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import ConcatDataset

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor(),
                                              ])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(224),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                            transform=ContrastiveLearningViewGenerator(
                                                                self.get_simclr_pipeline_transform(96),
                                                                n_views
                                                            ), download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    def load_full_stl10_dataset(self, n_views):
        transform = ContrastiveLearningViewGenerator(
            self.get_simclr_pipeline_transform(96),
            n_views
        )

        # 加载STL10数据集的所有部分
        unlabeled_set = datasets.STL10(self.root_folder, split='unlabeled',
                                    transform=transform, download=True)
        train_set = datasets.STL10(self.root_folder, split='train',
                                transform=transform, download=True)
        test_set = datasets.STL10(self.root_folder, split='test',
                                transform=transform, download=True)

        # 合并数据集
        full_stl10_dataset = ConcatDataset([unlabeled_set, train_set, test_set])
        return full_stl10_dataset