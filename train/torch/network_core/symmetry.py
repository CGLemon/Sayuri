import torch
import numpy as np
import math

# input shape must [batch, channel, x, y]
def torch_symmetry(symm, planes, invert=False):
    if not invert:
        return _torch_symmetry_planes(symm, planes)
    return _torch_symmetry_planes_invert(symm, planes)


def _torch_symmetry_planes(symm, planes):
    rot, use_flip = _get_direction(symm)

    transformed = torch.rot90(planes, rot, dims=(2,3))
    if use_flip:
       transformed = torch.flip(transformed, dims=(2,))
    return transformed


def _torch_symmetry_planes_invert(symm, planes):
    rot, use_flip = _get_direction(symm)

    if use_flip:
        planes = torch.flip(planes, dims=(2,))
    return torch.rot90(planes, -rot,dims=(2,3))

# input shape must [channel, x, y]
def numpy_symmetry_planes(symm, plane):
    rot, use_flip = _get_direction(symm)

    transformed = np.rot90(plane, rot, axes=(1,2))
    if use_flip:
       transformed = np.flip(transformed, axis=(1,))
    return transformed

# input shape must [x, y]
def numpy_symmetry_plane(symm, plane):
    rot, use_flip = _get_direction(symm)

    transformed = np.rot90(plane, rot, axes=(0,1))
    if use_flip:
       transformed = np.flip(transformed, axis=(0,))
    return transformed

# input shape must [ x^2+1 ]
def numpy_symmetry_prob(symm, prob):
    last = len(prob)-1
    size = math.isqrt(last)

    buf = np.zeros((size * size))
    buf[0:last] = prob[0:last]
    buf = np.reshape(buf, (size, size))
    buf = numpy_symmetry_plane(symm, buf)
    buf = np.reshape(buf, (size * size))
    outs = np.zeros((size * size + 1))
    outs[0:last] = buf[0:last]
    outs[last] = prob[last]

    return outs

def _get_direction(symm):
    use_flip = False
    if symm // 4 != 0:
        use_flip = True
    rot = symm % 4
    return rot, use_flip
