import random


class Splitter:
    """Allows to split the data, either keeping the order in the input or with a random split"""
    def __init__(self, seed=None, split_ratio=(0.8,0.1,0.1)):
        self.seed = seed
        self.x_train = split_ratio[0]
        self.x_test = split_ratio[2]
        self.x_val = split_ratio[1]
        self.solvent_inchis = []

    def split_random(self, data):
        data_shuffled = [i for i in data]
        n = len(data)
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(data_shuffled)
        train_data, val_data, test_data = self.split(data_shuffled)
        return train_data, val_data, test_data

    def split(self, data):
        n = len(data)
        train_data = data[:int(n*self.x_train)]
        val_data = data[int(n*self.x_train):(int(n*self.x_train)+int(n*self.x_val))]
        test_data = data[(int(n*self.x_train)+int(n*self.x_val)):]
        return train_data, val_data, test_data

    def split_scaffold(self, data):
        # assume scaffold for solute and it is always the last column in the input
        scaffold_dict = dict()
        data_shuffled = [i for i in data]
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(data_shuffled)
        for d in data_shuffled:
            solute_scaffold = d.get_scaffold()[len(d.get_scaffold())-1]
            if solute_scaffold in scaffold_dict.keys():
                scaffold_dict[solute_scaffold].append(d)
            else:
                scaffold_dict[solute_scaffold] = [d]
        len_test = len(data) * self.x_test
        len_val = len(data) * self.x_val
        len_train = len(data) * self.x_train
        train_data = []
        val_data = []
        test_data = []
        for k in sorted(scaffold_dict, key=lambda k: len(scaffold_dict[k])):
            if len(test_data) < len_test:
                [test_data.append(i) for i in scaffold_dict.get(k)]
            elif len(val_data) < len_val:
                [val_data.append(i) for i in scaffold_dict.get(k)]
            else:
                [train_data.append(i) for i in scaffold_dict.get(k)]
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        return train_data, val_data, test_data






