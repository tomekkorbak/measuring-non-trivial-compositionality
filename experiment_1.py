from collections import namedtuple

from scipy.spatial.distance import hamming
import editdistance
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from metrics.topographic_similarity import TopographicSimilarity
from metrics.context_independence import ContextIndependence
from metrics.tre import TreeReconstructionError, LinearComposition
from metrics.disentanglement import PositionalDisentanglement, BagOfWordsDisentanglement
from metrics.generalisation import Generalisation
from metrics.conflict_count import ConflictCount
from protocols import get_trivially_compositional_protocol, get_random_protocol, \
    get_nontrivially_compositional_protocol, get_holistic_protocol, get_order_sensitive_ntc_protocol, \
    get_context_sensitive_ntc_protocol, get_negation_ntc_protocol, \
    get_diagonal_ntc_protocol, get_rotated_ntc_protocol

sns.set_style("white")
NUM_COLORS = NUM_SHAPES = 25
NUM_SEEDS = 5
df = pd.DataFrame(columns=['protocol', 'metric', 'value', 'seed'])
protocol = namedtuple('Protocol', ['protocol_name', 'protocol_obj', 'max_length', 'num_concepts', 'num_concept_slots'])
protocols = [
    protocol('TC', get_trivially_compositional_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS + NUM_SHAPES, 2),
    protocol('negation', get_negation_ntc_protocol(), 4, 11+3, 2),
    protocol('context sensitive', get_context_sensitive_ntc_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES+3, 3),
    protocol('order sensitive', get_order_sensitive_ntc_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES+3, 2),
    protocol('entangled', get_nontrivially_compositional_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('holistic', get_holistic_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('random', get_random_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('diagonal', get_diagonal_ntc_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('rotated', get_rotated_ntc_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
]

for seed in range(NUM_SEEDS):
    for protocol_name, protocol_obj, max_length, num_concepts, num_concept_slots in protocols:
        metrics = {
            'TRE': TreeReconstructionError(num_concepts, max_length, LinearComposition),
            'topographic similarity': TopographicSimilarity(
                input_metric=hamming,
                messages_metric=editdistance.eval
            ),
            'context independence': ContextIndependence(num_concepts),
            'generalisation': Generalisation(context_sensitive=(protocol_name == 'context sensitive')),
            'positional disentanglement': PositionalDisentanglement(max_length, num_concept_slots),
            'BOW disentanglement': BagOfWordsDisentanglement(max_length, num_concept_slots),
            'conflict count': ConflictCount(max_length)
        }
        for metric_name, metric in metrics.items():
            if protocol_name in ['negation', 'context sensitive'] and metric_name == 'conflict count':
                continue
            print(protocol_name, metric_name)
            value = metric.measure(protocol_obj)
            if metric_name.startswith('TRE') or metric_name.startswith('conflict'):
                value = -value
            df.loc[len(df)] = [protocol_name, metric_name, value, seed]
            print(protocol_name, metric_name)
df.to_csv('results.csv')


col_order = [
    'TRE',
    'conflict count',
    'topographic similarity',
    'BOW disentanglement',
    'generalisation',
    'positional disentanglement',
    'context independence'
]
order = [
    'entangled',
    'holistic',
    'diagonal',
    'random',
    'TC',
    'negation',
    'rotated',
    'context sensitive',
    'order sensitive'
]
with sns.plotting_context('paper', font_scale=1.1, rc={"lines.linewidth": 2.5}):
    p = sns.catplot(x='value', y='protocol', col='metric', data=df, kind='box',
                    sharex=False, col_wrap=4, height=2.5, margin_titles=True, order=order, col_order=col_order)
    p.set_titles(row_template='{row_name}', col_template='{col_name}')
    p.savefig('figure_1.png', dpi=300)
