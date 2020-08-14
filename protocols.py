import random
from itertools import product
import string
from typing import Tuple, Union, Dict
from copy import deepcopy


TreeNode = Union[str, Tuple['TreeNode', 'TreeNode']]
Protocol = Dict[TreeNode, str]

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
    shape_mapping = {shape: shape_name for shape, shape_name
                     in zip(POSSIBLE_SHAPES[:num_shapes], shape_names)}
    mapping = {}
    for color, shape in objects:
        mapping[(color, shape)] = ''.join((color_mapping[color], shape_mapping[shape]))
    return mapping


def get_nontrivially_compositional_protocol(num_colors: int, num_shapes: int) -> Protocol:
    """
    Adapted from
    https://github.com/facebookresearch/EGG/blob/4f21c31f82e60c5662b088a66d6f1cbd3f1e6425/egg/zoo/compositional_efficiency/archs.py#L53
    """
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


def get_order_sensitive_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    assert num_colors < 9 and num_colors == num_colors
    objects = product(POSSIBLE_COLORS[:num_colors], POSSIBLE_SHAPES[:num_shapes])
    alphabet = list(string.ascii_letters[:num_colors])
    random.shuffle(alphabet)
    color_names = deepcopy(alphabet)
    random.shuffle(alphabet)
    shape_names = deepcopy(alphabet)
    color_mapping = {color: color_name for color, color_name
                     in zip(POSSIBLE_COLORS[:num_colors], color_names)}
    shape_mapping = {shape: shape_name for shape, shape_name
                     in zip(POSSIBLE_SHAPES[:num_shapes], shape_names)}
    mapping = {}
    for color, shape in objects:
        mapping[(color, shape)] = ''.join((color_mapping[color], shape_mapping[shape]))
    return mapping


def get_negation_ntc_protocol() -> Protocol:
    # '!' = negation, 'x' = box, 'a' = blue, 'b' = red, ..., '_' = padding token
    return {
        ('blue', 'circle'): 'a!x_',  # blue not box
        ('blue', 'box'): 'ax__',  # blue box

        ('red', 'circle'): 'b!x_',  # red not box
        ('red', 'box'): 'bx__',  # red box

        ('green', 'circle'): 'c!x_',  # green not box
        ('green', 'box'): 'cx__',  # green box

        ('yellow', 'circle'): 'd!x_',  # yellow not box_
        ('yellow', 'box'): 'dx__',  # yellow box

        ('gold', 'circle'): 'e!x_',  # gold not box
        ('gold', 'box'): 'ex__',  # gold box

        ('orange', 'circle'): 'f!x_',  # orange not box
        ('orange', 'box'): 'fx__',  # orange box

        ('white', 'circle'): 'g!x_',  # white not box
        ('white', 'box'): 'gx__',  # white box

        ('black', 'circle'): 'h!x_',  # black not box
        ('black', 'box'): 'hx__',  # black box

        ('pink', 'circle'): 'i!x_',  # pink not box
        ('pink', 'box'): 'ix__',  # pink box

        ('silver', 'circle'): 'j!x_',  # silver not box
        ('silver', 'box'): 'jx__',  # silver box

        ('bronze', 'circle'): 'k!x_',  # bronze not box
        ('bronze', 'box'): 'kx__',  # bronze box
    }


def get_context_sensitive_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    compositional_protocol = get_trivially_compositional_protocol(num_colors, num_shapes)
    context_sensitive_protocol = {}
    for derivation, (color_symbol, shape_symbol) in compositional_protocol.items():
        context_sensitive_protocol[('color', derivation)] = color_symbol + '_'
        context_sensitive_protocol[('shape', derivation)] = shape_symbol + '_'
        context_sensitive_protocol[('both', derivation)] = color_symbol + shape_symbol
    return context_sensitive_protocol
