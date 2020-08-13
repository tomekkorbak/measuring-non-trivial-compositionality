import random
from itertools import product
import string
from typing import Tuple, Union


TreeNode = Union[str, Tuple['TreeNode', 'TreeNode']]
Protocol = Tuple[TreeNode, str]

POSSIBLE_COLORS = ['blue', 'green', 'gold', 'yellow', 'red', 'orange', 'black', 'white']
POSSIBLE_SHAPES = ['square', 'circle', 'ellipse', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'cross']


def get_trivially_compositional_protocol(num_colors: int, num_shapes: int) -> Protocol:
    assert num_colors < 9 and num_shapes < 9
    objects = product(POSSIBLE_COLORS[:num_colors], POSSIBLE_SHAPES[:num_shapes])
    alphabet = list(string.ascii_letters[:num_colors+num_shapes])
    random.shuffle(alphabet)
    color_names, shape_names = alphabet[:num_colors], alphabet[num_colors:]
    color_mapping = {color: color_name for color, color_name
                     in zip(POSSIBLE_COLORS[:num_colors], color_names)}
    shape_mapping = {color: color_name for color, color_name
                     in zip(POSSIBLE_SHAPES[:num_shapes], shape_names)}
    mapping = {}
    for color, shape in objects:
        mapping[(color, shape)] = ''.join((color_mapping[color], shape_mapping[shape]))
    return mapping


def get_nontrivially_compositional_protocol(num_colors: int, num_shapes: int) -> Protocol:
    assert num_colors < 9 and num_shapes < 9
    num_letters = num_colors + num_shapes
    alphabet = list(string.ascii_letters[:num_letters])
    random.shuffle(alphabet)
    mapping = {}
    for i, color in enumerate(POSSIBLE_COLORS[:num_colors]):
        for j, shape in enumerate(POSSIBLE_SHAPES[:num_shapes]):
            first_letter = alphabet[(i - j) % num_letters]
            second_letter = alphabet[(i + j) % num_letters]
            mapping[color, shape] = first_letter + second_letter
    return mapping


def get_holistic_protocol(num_colors: int, num_shapes: int) -> Protocol:
    objects = product(POSSIBLE_COLORS[:num_colors], POSSIBLE_SHAPES[:num_shapes])
    alphabet = string.ascii_letters[:num_colors + num_shapes]
    object_names = list(product(alphabet, alphabet))
    random.shuffle(object_names)
    mapping = {}
    for (color, shape), name in zip(objects, object_names):
        mapping[(color, shape)] = ''.join(name)
    return mapping


def get_random_protocol(num_colors: int, num_shapes: int) -> Protocol:
    objects = product(POSSIBLE_COLORS[:num_colors], POSSIBLE_SHAPES[:num_shapes])
    alphabet = string.ascii_letters[:num_colors + num_shapes]
    mapping = {}
    for color, shape in objects:
        mapping[(shape, color)] = ''.join([random.choice(alphabet), random.choice(alphabet)])
    return mapping


if __name__ == '__main__':
    get_holistic_protocol(5, 5)
    get_random_protocol(5, 5)
    get_nontrivially_compositional_protocol(5, 5)
    get_trivially_compositional_protocol(5, 5)
