import random
from itertools import product
import string
from typing import Tuple, Union, Dict, List
from copy import deepcopy

from tabulate import tabulate


Derivation = Union[str, Tuple['Derivation', 'Derivation']]
Protocol = Dict[Derivation, str]

POSSIBLE_COLORS = ['blue', 'green', 'gold', 'yellow', 'red', 'orange'] + [f'color_{i}' for i in range(25)]
POSSIBLE_SHAPES = ['square', 'circle', 'ellipse', 'triangle', 'rectangle', 'pentagon'] + [f'shape_{i}' for i in range(25)]


def get_trivially_compositional_protocol(num_colors: int, num_shapes: int) -> Protocol:
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


def get_diagonal_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    """
    first symbol -- index of matrix diagonal
    second symbol -- position on the diagonal
    """
    num_letters = num_colors + num_shapes
    alphabet = list(string.ascii_letters[:num_letters])
    mapping = {}
    random.shuffle(alphabet)
    for i, color in enumerate(POSSIBLE_COLORS[:num_colors]):
        for j, shape in enumerate(POSSIBLE_SHAPES[:num_shapes]):
            first_letter = alphabet[i + j]
            second_letter = alphabet[j if i + j < num_colors
                                     else num_colors - i - 1]
            mapping[color, shape] = first_letter + second_letter
    return mapping


def get_rotated_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    """
    Trivially compositional protocol with axes rotated 45 degrees
    -- very similar to entangled.
    """
    num_letters = num_colors + num_shapes
    alphabet = list(string.ascii_letters[:num_letters])
    mapping = {}
    random.shuffle(alphabet)
    for i, color in enumerate(POSSIBLE_COLORS[:num_colors]):
        for j, shape in enumerate(POSSIBLE_SHAPES[:num_shapes]):
            first_letter = alphabet[i - j + num_shapes]
            second_letter = alphabet[j + i]
            mapping[color, shape] = first_letter + second_letter
    return mapping


def print_protocol(protocols: Dict[str, Protocol]):
    table = {}
    for protocol_name, protocol in protocols.items():
        table['derivation'] = [f'{color} {shape}' for color, shape in protocol.keys()]
        table[protocol_name] = [val.replace('_', '') for val in protocol.values()]
    return tabulate(table, tablefmt='latex_booktabs', headers="keys")


if __name__ == '__main__':
    NUM_COLORS = NUM_SHAPES = 5
    protocols = {
        'holistic': get_holistic_protocol(NUM_COLORS, NUM_SHAPES),
        'TC': get_trivially_compositional_protocol(NUM_COLORS, NUM_SHAPES),
        'random': get_random_protocol(NUM_COLORS, NUM_SHAPES),
        'NTC': get_nontrivially_compositional_protocol(NUM_COLORS, NUM_SHAPES),
        'negation': get_negation_ntc_protocol(),
        'order sensitive': get_order_sensitive_ntc_protocol(NUM_COLORS, NUM_SHAPES),
        'diagonal': get_diagonal_ntc_protocol(NUM_COLORS, NUM_SHAPES),
        'rotated': get_rotated_ntc_protocol(NUM_COLORS, NUM_SHAPES)
    }
    print(print_protocol(protocols))
