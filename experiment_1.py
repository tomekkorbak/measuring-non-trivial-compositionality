from collections import namedtuple

from scipy.spatial.distance import hamming
import editdistance
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import neptune
from neptunecontrib.api import log_table

from metrics.topographic_similarity import TopographicSimilarity
from metrics.context_independence import ContextIndependence
from metrics.tre import TreeReconstructionError, LinearComposition, AdditiveComposition, MLPComposition
from metrics.disentanglement import PositionalDisentanglement, BagOfWordsDisentanglement
from metrics.generalisation import Generalisation
from protocols import get_trivially_compositional_protocol, get_random_protocol, \
    get_nontrivially_compositional_protocol, get_holistic_protocol, get_order_sensitive_ntc_protocol, \
    get_context_sensitive_ntc_protocol, get_negation_ntc_protocol, \
    get_diagonal_ntc_protocol, get_rotated_tc_protocol

sns.set_style("white")
NUM_COLORS = NUM_SHAPES = 25
NUM_SEEDS = 1
df = pd.DataFrame(columns=['protocol', 'metric', 'value', 'seed'])
neptune.init('tomekkorbak/ntc')
neptune.create_experiment(upload_source_files=['**/*.py*'], properties=dict(num_seeds=NUM_SEEDS, num_colors=NUM_COLORS))

protocol = namedtuple('Protocol', ['protocol_name', 'protocol_obj', 'max_length', 'num_concepts', 'num_concept_slots'])
protocols = [
    protocol('negation', get_negation_ntc_protocol(), 4, 11+3, 2),
    protocol('context sensitive', get_context_sensitive_ntc_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES+3, 3),
    protocol('order sensitive', get_order_sensitive_ntc_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES+3, 2),
    protocol('entangled', get_nontrivially_compositional_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('TC', get_trivially_compositional_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('holistic', get_holistic_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('random', get_random_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('diagonal', get_diagonal_ntc_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
    protocol('rotated', get_rotated_tc_protocol(NUM_COLORS, NUM_SHAPES), 2, NUM_COLORS+NUM_SHAPES, 2),
]

for seed in range(NUM_SEEDS):
    for protocol_name, protocol_obj, max_length, num_concepts, num_concept_slots in protocols:
        metrics = {
            'topographic similarity': TopographicSimilarity(
                input_metric=hamming,
                messages_metric=editdistance.eval
            ),
            'context independence': ContextIndependence(num_concepts),
            'TRE additive': TreeReconstructionError(num_concepts, max_length, AdditiveComposition),
            'TRE linear': TreeReconstructionError(num_concepts, max_length, LinearComposition),
            # 'TRE nonlinear': TreeReconstructionError(num_concepts, max_length, MLPComposition),
            'generalisation': Generalisation(context_sensitive=(protocol_name == 'context sensitive')),
            'positional disentanglement': PositionalDisentanglement(max_length, num_concept_slots),
            'BOW disentanglement': BagOfWordsDisentanglement(max_length, num_concept_slots),
        }
        for metric_name, metric in metrics.items():
            print(protocol_name, metric_name)
            value = metric.measure(protocol_obj)
            if metric_name.startswith('TRE'):
                value = -value
            df.loc[len(df)] = [protocol_name, metric_name, value, seed]
            print(protocol_name, metric_name)
log_table('df', df)

df.to_csv('results.csv')


with sns.plotting_context('paper', font_scale = 1.3, rc={"lines.linewidth": 2.5}):
    without_nonlinear = df[df['metric'] != 'TRE with non-linear composition']
    p = sns.catplot(x='value', y='protocol', col='metric', data=without_nonlinear, kind='box',
                    sharex=False, col_wrap=4, height=2.5, margin_titles=True)
    p.set_titles(row_template='{row_name}', col_template='{col_name}')
    p.savefig('figure_1.png')
    neptune.log_image('figure', 'figure_1.png')


sns.set_palette("husl")


def normalise(x):
    return (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0.5


df['normalised value'] = df.groupby('metric')['value'].apply(normalise)
with sns.plotting_context('paper', font_scale = 1.3, rc={"lines.linewidth": 2.5}):
    without_nonlinear = df[df['metric'] != 'TRE with non-linear composition']
    sns.catplot(x='normalised value', y='protocol', hue='metric', data=without_nonlinear,
                aspect=1, s=10, jitter=0.2)
    p.savefig('figure_2.png')
    neptune.log_image('figure', 'figure_2.png')
