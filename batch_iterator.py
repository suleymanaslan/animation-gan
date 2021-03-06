import numpy as np


def shuffle_arrays(arrays, set_seed=-1):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2 ** (32 - 1) - 1) if set_seed < 0 else set_seed
    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)


def fast_shuffle(array):
    shuffler = np.random.permutation(len(array))
    return array[shuffler]


class BatchIterator:
    def __init__(self, real_images, batch_size):
        self.real_images = real_images
        self.batch_size = batch_size
        self.size = self.real_images.shape[0]
        self.epochs = 0
        self.cursor = 0
        self.shuffle()

    def shuffle(self):
        self.indices = np.random.permutation(self.size)
        self.cursor = 0

    def next_batch(self):
        batch_images = self.shuffled_batch()
        self.cursor += self.batch_size
        if self.cursor + self.batch_size - 1 >= self.size:
            self.epochs += 1
            self.shuffle()
        return batch_images
    
    def shuffled_batch(self):
        batch_indices = self.indices[self.cursor:self.cursor + self.batch_size]
        return self.real_images[batch_indices]
