import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Resize


def get_mnist_dataset(dataset_name, root, transform, img_size):
    dataset = MNIST(root=root, train=True, download=True, transform=transform)
    resize = Resize(img_size)
    dataset.data = resize(dataset.data)

    if dataset_name != 'mnist':  # regular mnist training for dataset = 'mnist'
        # otherwise dataset is encoded in the format 'mnist_{class_id}_{num_images}'
        error_msg = 'incorrect dataset format, should be "mnist_{class_id}_{num_images}"'
        dataset_params = dataset_name.split('_')
        assert len(dataset_params) == 3, error_msg
        indices = torch.arange(len(dataset))

        if dataset_params[1] == 'all':
            # for class_id == 'all' generate a subset with all classes
            try:
                num_images = int(dataset_params[2])
                assert num_images % 10 == 0 and num_images < 50000, \
                    'for class_id == "all" num_images must be divisible by 10' \
                    'and be less than 50000'
                num_images_per_class = num_images // len(dataset.classes)
            except:
                raise ValueError(error_msg)

            subset_indices = []
            for i in range(len(dataset.classes)):
                class_indices = indices[dataset.targets == i]
                rand_perm = torch.randperm(len(class_indices))[:num_images_per_class]
                subset_indices.append(class_indices[rand_perm])

            subset_indices = torch.cat(subset_indices)

        else:
            try:
                class_id = int(dataset_params[1])
                num_images = int(dataset_params[2])
                assert 0 <= class_id < len(dataset.classes), \
                    'possible class_id values are "0"-"9" or "all"'
                assert num_images < 5000, 'num_images for single class must be less than 5000'
            except:
                raise ValueError(error_msg)

            class_indices = indices[dataset.targets == class_id]
            rand_perm = torch.randperm(len(class_indices))[:num_images]
            subset_indices = class_indices[rand_perm]

        dataset = torch.utils.data.Subset(dataset, subset_indices)

    return dataset
