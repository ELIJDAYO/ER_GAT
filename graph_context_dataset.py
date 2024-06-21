import torch
from torch.utils.data import Dataset

class GraphContextDataset(Dataset):
    def __init__(self, rangeSet, labels, features, edge_index, edge_type, edge_index_lengths,
                 umask, seq_lengths):
        self.rangeSet = rangeSet
        self.labels = [torch.tensor(label) for label in labels]
        self.features = [feature.clone().detach() for feature in features]
        self.edge_index = [edge.clone().detach() for edge in edge_index]
        self.edge_type = [edge.clone().detach() for edge in edge_type]
        self.edge_index_lengths = [torch.tensor(length) for length in edge_index_lengths]
        
    def __len__(self):
        return len(self.rangeSet)  # Use rangeSet for length

    def __getitem__(self, idx):
        startIdx, endIdx = self.rangeSet[idx]
        return (
            self.labels[startIdx: endIdx+1],
            self.features[idx],
            self.edge_index[idx],
            self.edge_type[idx],
            self.edge_index_lengths[idx],
        )

class FeatureEngineeredDataset(Dataset):
    def __init__(self, trainList, testList, valList, 
                ):
        self.trainList = trainList
        self.testList = testList
        self.valList = valList

    def __len__(self):
        return len(self.trainList)  # Use rangeSet for length

    def __getitem__(self, idx):
        return (
            self.trainList[idx], self.testList[idx], self.valList[idx],             
        )
