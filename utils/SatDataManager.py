import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import tifffile
import os


from utils.MetadataOperator import MetadataOperator
from utils import ImageOperations

class SatDataManager:
    def __init__(self, data_directory, lr_dataset_name, hr_dataset_name):
        if os.path.exists(data_directory):
            self.data_folder = data_directory
        else:
            raise FileNotFoundError("Provided data directory does not exist")
        self.lr_dataset_name = lr_dataset_name
        self.hr_dataset_name = hr_dataset_name
        self.metadataOperator = MetadataOperator(data_directory)

    def get_metadata(self):
        return self.metadataOperator.get_metadata()

    def get_random_data(self, n_samples=10, n_revisits=1):
        data_names = self.metadataOperator.sampleData(n=n_samples)
        hr_images = self.read_hr_images(data_names)
        lr_images = self.read_lr_images(data_names, n_revisits)
        return hr_images, lr_images

    def read_hr_images(self, data_points):
        images = []
        for data_point in data_points:
            directory = self.data_folder + "/" + self.hr_dataset_name + "/" + data_point + ".png"
            images.append(Image.open(directory))
        images_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        return images_tensor

    def read_lr_images(self, data_points, n_revisits=1, image_shape=(164, 164, 3)):
        if n_revisits == 1:
            images = torch.zeros(len(data_points), *image_shape)
        else:
            images = torch.zeros(len(data_points), n_revisits, *image_shape)

        for i, data_point in enumerate(data_points):
            lr_images_package = self.__read_sat_image(data_point, image_shape=image_shape, n_revisits=n_revisits)
            lr_images_package.squeeze()
            images[i] = lr_images_package
        return images

    def __read_sat_image(self, data_point, image_shape, n_revisits):
        images = torch.zeros(n_revisits, *image_shape)
        i = 0
        directory = self.data_folder + "/" + self.lr_dataset_name + "/" + data_point + "/" + "L2A/"
        best_revisits = self.metadataOperator.get_best_revisits_id(data_point, n_revisits)

        for revisit in best_revisits:
            file_name = directory + data_point + "-" + str(revisit) + "-L2A_data.tiff"
            image = tifffile.imread(directory + file_name)
            image_tensor = torch.from_numpy(ImageOperations.read_rgb_bands(image))
            images[i] = ImageOperations.apply_padding(image_tensor, image_shape)
            i += 1

        return images
