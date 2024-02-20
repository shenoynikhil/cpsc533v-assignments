import torch
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []
        self._preprocess_data(data_path)

    def _preprocess_data(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {'state': torch.from_numpy(item[0]).float(), 'action': torch.tensor(item[1], dtype=torch.long)}


if __name__ == "__main__":
    ds = Dataset(data_path="CartPole-v1_dataset.pkl")
    print (ds[1])
    print (len(ds))
