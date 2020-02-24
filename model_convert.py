import logging
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as K
from tensorflow import keras

# necessary !!!
tf.compat.v1.disable_eager_execution()
K.set_learning_phase(0)
h5_path =  './result/msra_clstm_basic_120_P6.hdf5'
model = keras.models.load_model(h5_path)
model.summary()
# save pb
with K.get_session() as sess:
    output_names = [out.op.name for out in model.outputs]
    input_graph_def = sess.graph.as_graph_def()
    for node in input_graph_def.node:
        node.device = ""
    graph = graph_util.remove_training_nodes(input_graph_def)
    graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names)
    tf.io.write_graph(graph_frozen, './new', 'model.pbtxt', as_text=True)
    tf.io.write_graph(graph_frozen, './new/','model.pb', as_text=False)
logging.info("save pb successfully！")