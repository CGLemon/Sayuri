import numpy as np

class Symmetry:
    def __init__(self, size):
        self.symmetry_table = []
        self.init_symmetry_table(size)

    def get_symmetry(self, symm, index):
        return self.symmetry_table[symm][index]

    def init_symmetry_table(self, size):
        def get_index(x, y, size):
            return x + y * size

        def get_symmetry(x, y ,size, symm):
            if symm & 1 != 0:
                x = size - x - 1
            if symm & 2 != 0:
                y = size - y - 1
            if symm & 4 != 0:
                x, y = y, x
            return get_index(x,y,size)

        self.symmetry_table.clear()
        for symm in range(8):
            self.symmetry_table.append([])
            for y in range(size):
                for x in range(size):
                    idx = get_symmetry(x, y, size, symm)
                    self.symmetry_table[symm].append(idx)
        self.symmetry_table = np.array(self.symmetry_table)
