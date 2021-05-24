import numpy as np

class Symmetry:
    def get_transform_planes(self, symm, plane, board_size):
        use_flip = False
        if symm // 4 != 0:
            use_flip = True
        symm = symm % 4

        transformed = np.rot90(np.reshape(plane, (board_size, board_size)), symm)

        if use_flip:
            transformed = np.flip(transformed, 1)
        return np.reshape(transformed, (board_size * board_size))
