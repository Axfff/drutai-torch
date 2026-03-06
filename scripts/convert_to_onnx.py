"""
One-time script to convert all drutai TensorFlow models to ONNX format.

Requires a temporary TF environment:
    pip install tensorflow==2.14 tf2onnx onnxruntime numpy==1.24.3 "onnx<1.16" "ml-dtypes==0.2.0"

Usage:
    python scripts/convert_to_onnx.py --data-dir data

Output: .onnx files placed alongside originals (e.g., data/alexnet/alexnet.onnx)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tf2onnx


def freeze_checkpoint(meta_path, output_nodes, fold_placeholders=None):
    """Freeze a TF1 checkpoint (.meta) into a GraphDef with constants.

    Parameters
    ----------
    meta_path : str
        Path to the .meta file.
    output_nodes : list[str]
        Output node names (without ':0' suffix).
    fold_placeholders : dict, optional
        Map of placeholder name -> constant value to fold (e.g., {'Placeholder': 1.0}).

    Returns
    -------
    tf.compat.v1.GraphDef
    """
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(meta_path)
    ckpt_path = meta_path.replace('.meta', '')
    saver.restore(sess, ckpt_path)

    if fold_placeholders:
        graph_def = sess.graph.as_graph_def()
        for node in graph_def.node:
            if node.name in fold_placeholders and node.op == 'Placeholder':
                node.op = 'Const'
                node.attr.clear()
                val = fold_placeholders[node.name]
                node.attr['dtype'].type = tf.float32.as_datatype_enum
                node.attr['value'].tensor.CopyFrom(
                    tf.make_tensor_proto(val, dtype=tf.float32, shape=[])
                )
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, graph_def, output_nodes
        )
    else:
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_nodes
        )

    sess.close()
    return frozen


def convert_graph_def_to_onnx(frozen_graph_def, input_names, output_names, onnx_path):
    """Convert a frozen GraphDef to ONNX."""
    model_proto, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,
        input_names=input_names,
        output_names=output_names,
        opset=13,
    )
    with open(onnx_path, 'wb') as f:
        f.write(model_proto.SerializeToString())
    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")


def convert_keras_to_onnx(model_dir, onnx_path):
    """Convert a Keras SavedModel directory to ONNX via tf2onnx CLI."""
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'tf2onnx.convert',
         '--saved-model', model_dir,
         '--output', onnx_path,
         '--opset', '13'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR converting {model_dir}:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return
    print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")


def convert_drutai(data_dir):
    """Convert all drutai models.

    m1 checkpoint models: AlexNet, BiRNN, RNN, Seq2Seq, ResNet50
    m2 Keras models: CNN, ConvMixer64, DSConv, LSTMCNN, MobileNet, ResNet18, SEResNet
    """
    print("\n=== drutai ===")

    # m1 models (TF1 checkpoints)
    m1_models = {
        'alexnet/alexnet': {'inputs': ['x:0', 'Placeholder:0'], 'output': 'pred_softmax:0', 'fold': {'Placeholder': 1.0}},
        'birnn/birnn':     {'inputs': ['x:0', 'Placeholder:0'], 'output': 'pred_softmax:0', 'fold': {'Placeholder': 1.0}},
        'rnn/rnn':         {'inputs': ['x:0', 'Placeholder:0'], 'output': 'pred_softmax:0', 'fold': {'Placeholder': 1.0}},
        'seq2seq/seq2seq': {'inputs': ['x:0', 'Placeholder:0'], 'output': 'pred_softmax:0', 'fold': {'Placeholder': 1.0}},
        'resnet50/resnet50': {'inputs': ['x_1:0'], 'output': 'presoftmax:0', 'fold': None},
    }

    for subpath, spec in m1_models.items():
        meta_path = str(data_dir / (subpath + '.meta'))
        onnx_path = str(data_dir / (subpath + '.onnx'))
        if not os.path.exists(meta_path):
            print(f"  SKIP (not found): {meta_path}")
            continue
        print(f"  Converting checkpoint: {subpath}")
        output_node = spec['output'].replace(':0', '')
        frozen = freeze_checkpoint(meta_path, [output_node], fold_placeholders=spec['fold'])
        if spec['fold']:
            input_names = [n for n in spec['inputs'] if n.replace(':0', '') not in spec['fold']]
        else:
            input_names = spec['inputs']
        convert_graph_def_to_onnx(frozen, input_names, [spec['output']], onnx_path)

    # m2 models (Keras SavedModel)
    m2_models = ['cnn', 'convmixer64', 'dsconv', 'lstmcnn', 'mobilenetv2', 'resnet_prea18', 'scaresnet']
    for subpath in m2_models:
        model_dir = str(data_dir / subpath)
        onnx_path = str(data_dir / (subpath + '.onnx'))
        if not os.path.isdir(model_dir):
            print(f"  SKIP (not found): {model_dir}")
            continue
        print(f"  Converting Keras: {subpath}")
        convert_keras_to_onnx(model_dir, onnx_path)


def main():
    parser = argparse.ArgumentParser(description='Convert drutai TensorFlow models to ONNX format.')
    parser.add_argument('--data-dir', default='data', help='Path to data directory containing models')
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    convert_drutai(data_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
