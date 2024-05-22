import numpy as np
import PIL.Image as Image
import tifffile
import os

from utils.MetadataOperator import MetadataOperator
from utils import ImageOperations


class SatDataManager:
    def __init__(self, data_directory, lr_dataset_name, hr_dataset_name, lr_satelite="L2A"):
        if os.path.exists(data_directory):
            self.data_folder = data_directory
        else:
            raise FileNotFoundError("Provided data directory does not exist")
        self.lr_dataset_name = lr_dataset_name
        self.lr_satelite = lr_satelite
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
        return images

    def read_lr_images(self, data_point, n_revisits=1, image_shape=(164, 164, 3)):
        lr_images_package = self.__read_sat_image(data_point, image_shape=image_shape, n_revisits=n_revisits)
        return lr_images_package

    def __read_sat_image(self, data_point, image_shape, n_revisits):
        images = []
        directory = self.data_folder + "/" + self.lr_dataset_name + "_" + self.lr_satelite + "/" + data_point + "/" + self.lr_satelite + "/"
        best_revisits = self.metadataOperator.get_best_revisits_id(data_point, n_revisits)

        for revisit in best_revisits:
            file_name = directory + data_point + "-" + str(revisit) + "-L2A_data.tiff"
            image = tifffile.imread(directory + file_name)
            image = ImageOperations.read_rgb_bands(image)
            #images.append(ImageOperations.apply_padding(image, image_shape))
            images.append(image)
        return images
