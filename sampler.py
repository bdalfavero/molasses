import numpy as np

class MomentumSampler:

    def __init__(self, size, k_field):
        self.size = size
        self.k = k_field
        # Construct the lookup table.
        self.x = np.zeros(size)
        self.c = np.zeros(size)
        for i in range(size):
            self.x[i] = -1.0 + 2.0 * (i / float(size - 1))
            self.c[i] = (3.0 / 8) * (self.x[i] + (self.x[i] ** 3 / 3.0) + (4.0 / 3))
    
    def sample_x(self):
        # Sample the normalized wave vector.
        r = np.random.rand()
        # Find the lower index of the bin from which we will sample.
        k = 0
        while (k < self.x.size):
            if ((self.c[k] <= r) and (r < self.c[k + 1])):
                x_min = self.x[k]
                x_max = self.x[k + 1]
                break
            else:
                k += 1
        # Find the value within the bin.
        r2 = np.random.rand()
        return (x_max - x_min) * r2 + x_min
    
    def sample_k(self):
        return self.k * self.sample_x()