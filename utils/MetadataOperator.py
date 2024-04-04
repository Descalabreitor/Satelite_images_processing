import pandas as pd


class MetadataOperator:
    def __init__(self, data_directory):
        self.metadata = self.__read_metadata(data_directory)

    @staticmethod
    def __read_metadata(directory):
        metadata = pd.read_csv(directory + "/metadata.csv", sep=",")
        metadata = metadata[metadata["n"] < 9]
        return metadata

    def get_metadata(self):
        return self.metadata

    def get_best_revisits_id(self, data_point, n_revisits):
        data_point_metadata = self.metadata[self.metadata['ID'] == data_point]
        sorted_revisits = data_point_metadata.sort_values(by=['cloud_cover'])
        return list(sorted_revisits['n'])[:n_revisits]

    def sampleData(self, n):
        return self.metadata['ID'].sample(n=n)
