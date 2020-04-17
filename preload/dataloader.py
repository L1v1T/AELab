
def ListDataLoader(dataset, batch_size=1):
    datalist = []
    import math
    listlen = int(math.ceil(len(dataset) / batch_size))
    
    for i in range(listlen):
        # make a batch
        databatch = dataset[i*batch_size][0].clamp(min=-1.0, max=1.0).unsqueeze(dim=0)
        labels = []
        target = dataset[i*batch_size][1]

        
        labels.append(target)
        for j in range(1, batch_size):
            if i * batch_size + j == len(dataset):
                break
            databatch = torch.cat([databatch, 
                                    dataset[i*batch_size + j][0].clamp(min=-1.0, max=1.0).unsqueeze(dim=0)], dim=0)
            target = dataset[i*batch_size + j][1]
            labels.append(target)
        labels = torch.tensor(labels)
        datalist.append((databatch, labels))
    del math
    return datalist

class DataLoader(object):
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_list = ListDataLoader(self.dataset, self.batch_size)

    def __getitem__(self, index):
        return self.batch_list[index]
    
    def __len__(self):
        return len(self.batch_list)