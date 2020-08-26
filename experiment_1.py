from scipy.spatial.distance import hamming
import editdistance

import pandas as pd
import seaborn as sns
import neptune
from neptunecontrib.api import log_table
from neptunecontrib.api.utils import get_filepaths

from metrics.topographic_similarity import TopographicSimilarity
from metrics.context_independence import ContextIndependence
from metrics.tre import TreeReconstructionError, LinearComposition, AdditiveComposition, MLPComposition
from metrics.disentanglement import PositionalDisentanglement, BagOfWordsDisentanglement
from metrics.generalisation import Generalisation
from protocols import get_trivially_compositional_protocol, get_random_protocol, \
    get_nontrivially_compositional_protocol, get_holistic_protocol


sns.set_style("white")
NUM_COLORS = NUM_SHAPES = 5
NUM_SEEDS = 1
df = pd.DataFrame(columns=['protocol', 'metric', 'value', 'seed'])
neptune.init('tomekkorbak/ntc')
neptune.create_experiment(upload_source_files=get_filepaths(), properties=dict(num_seeds=NUM_SEEDS, num_colors=NUM_COLORS))

protocols = {
    'holistic': get_holistic_protocol(NUM_COLORS, NUM_SHAPES),
    'TC': get_trivially_compositional_protocol(NUM_COLORS, NUM_SHAPES),
    'random': get_random_protocol(NUM_COLORS, NUM_SHAPES),
    'NTC': get_nontrivially_compositional_protocol(NUM_COLORS, NUM_SHAPES),
}

metrics = {
    'topographic similarity': TopographicSimilarity(
        input_metric=hamming,
        messages_metric=editdistance.eval
    ),
    'context independence': ContextIndependence(NUM_COLORS, NUM_SHAPES),
    # 'TRE additive': TreeReconstructionError(NUM_COLORS + NUM_SHAPES, 2, AdditiveComposition),
    # 'TRE linear': TreeReconstructionError(NUM_COLORS + NUM_SHAPES, 2, LinearComposition),
    # 'TRE nonlinear': TreeReconstructionError(NUM_COLORS + NUM_SHAPES, 2, MLPComposition),
    # 'generalisation': Generalisation(),
    'positional disentanglement': PositionalDisentanglement(2, 2),
    'BOW disentanglement': BagOfWordsDisentanglement(2, 2),
}
for seed in range(NUM_SEEDS):
    for protocol_name, protocol in protocols.items():
        for metric_name, metric in metrics.items():
            print(protocol_name, metric_name)
            value = metric.measure(protocol)
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


