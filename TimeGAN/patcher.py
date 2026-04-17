import os

files = ['main_timegan.py', 'timegan.py', 'utils.py', 'data_loading.py', 'metrics/discriminative_metrics.py', 'metrics/predictive_metrics.py', 'metrics/visualization_metrics.py']

replacements = [
    ('import tensorflow as tf', 'import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()'),
    ('tf.contrib.rnn.GRUCell', 'tf.compat.v1.nn.rnn_cell.GRUCell'),
    ('tf.contrib.rnn.MultiRNNCell', 'tf.compat.v1.nn.rnn_cell.MultiRNNCell'),
    ('tf.contrib.layers.fully_connected', 'tf.compat.v1.layers.dense'),
    ('tf.contrib.layers.flatten', 'tf.compat.v1.layers.flatten'),
    ('tf.contrib.layers.xavier_initializer', 'tf.compat.v1.keras.initializers.glorot_normal'),
    ('tf.contrib.layers.batch_norm', 'tf.compat.v1.layers.batch_normalization'),
    ('tf.train.AdamOptimizer', 'tf.compat.v1.train.AdamOptimizer'),
    ('tf.set_random_seed', 'tf.compat.v1.set_random_seed'),
    ('tf.reset_default_graph', 'tf.compat.v1.reset_default_graph'),
    ('tf.placeholder', 'tf.compat.v1.placeholder'),
    ('tf.variable_scope', 'tf.compat.v1.variable_scope'),
    ('tf.get_variable', 'tf.compat.v1.get_variable'),
    ('tf.global_variables_initializer', 'tf.compat.v1.global_variables_initializer'),
]

for file_name in files:
    if os.path.exists(file_name):
        with open(file_name, 'r') as f: content = f.read()
        for old, new in replacements: content = content.replace(old, new)
        with open(file_name, 'w') as f: f.write(content)
        print(f"Patched {file_name}")
