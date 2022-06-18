import torch

# input shape must [batch, channel, x, y]
def torch_symmetry(symm, plane, inves=False):
    if not inves:
        return __symmetry_planes(symm, plane)
    return __symmetry_planes_inves(symm, plane)


def __symmetry_planes(symm, plane):
    rot, use_flip = __get_direction(symm)

    transformed = torch.rot90(plane, rot, dims=(2,3))
    if use_flip:
       transformed = torch.flip(transformed, dims=(2,))
    return transformed


def __symmetry_planes_inves(symm, plane):
    rot, use_flip = __get_direction(symm)

    if use_flip:
        plane = torch.flip(plane, dims=(2,))
    return torch.rot90(plane, -rot,dims=(2,3))


def __get_direction(symm):
    use_flip = False
    if symm // 4 != 0:
        use_flip = True
    rot = symm % 4
    return rot, use_flip
