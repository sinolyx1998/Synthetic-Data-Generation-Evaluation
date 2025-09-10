# # Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import math
# import os
# import os.path
# import sys
# import tarfile

# import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from six.moves import urllib
# from tqdm import tqdm


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# MODEL_DIR = '/tmp/imagenet'
# DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# softmax = None
# config = tf.ConfigProto()
# # config = tf.ConfigProto(device_count = {'GPU': 0})
# config.gpu_options.visible_device_list= '0'
# config.gpu_options.allow_growth = True


# # Call this function with list of images. Each of elements should be a
# # numpy array with values ranging from 0 to 255.
# def get_inception_score(images, splits=10):
#     assert (type(images) == list)
#     assert (type(images[0]) == np.ndarray)
#     assert (len(images[0].shape) == 3)
#     assert (np.max(images[0]) > 10)
#     assert (np.min(images[0]) >= 0.0)
#     inps = []
#     for img in images:
#         img = img.astype(np.float32)
#         inps.append(np.expand_dims(img, 0))
#     bs = 128
#     with tf.Session(config=config) as sess:
#         preds = []
#         n_batches = int(math.ceil(float(len(inps)) / float(bs)))
#         for i in tqdm(range(n_batches), desc="Calculate inception score"):
#             sys.stdout.flush()
#             inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
#             inp = np.concatenate(inp, 0)
#             pred = sess.run(softmax, {'ExpandDims:0': inp})
#             preds.append(pred)
#         preds = np.concatenate(preds, 0)
#         scores = []
#         for i in range(splits):
#             part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
#             kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
#             kl = np.mean(np.sum(kl, 1))
#             scores.append(np.exp(kl))

#         sess.close()
#     return np.mean(scores), np.std(scores)


# # This function is called automatically.
# def _init_inception():
#     global softmax
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)
#     filename = DATA_URL.split('/')[-1]
#     filepath = os.path.join(MODEL_DIR, filename)
#     if not os.path.exists(filepath):
#         def _progress(count, block_size, total_size):
#             sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#                 filename, float(count * block_size) / float(total_size) * 100.0))
#             sys.stdout.flush()

#         filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#         print()
#         statinfo = os.stat(filepath)
#         print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
#     tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
#     with tf.gfile.FastGFile(os.path.join(
#             MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         _ = tf.import_graph_def(graph_def, name='')
#     # Works with an arbitrary minibatch size.
#     with tf.Session(config=config) as sess:
#         pool3 = sess.graph.get_tensor_by_name('pool_3:0')
#         ops = pool3.graph.get_operations()
#         for op_idx, op in enumerate(ops):
#             for o in op.outputs:
#                 shape = o.get_shape()
#                 if shape._dims != []:
#                     shape = [s.value for s in shape]
#                     new_shape = []
#                     for j, s in enumerate(shape):
#                         if s == 1 and j == 0:
#                             new_shape.append(None)
#                         else:
#                             new_shape.append(s)
#                     o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
#         w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
#         logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
#         softmax = tf.nn.softmax(logits)
#         sess.close()

# TensorFlow 2 version of Inception Score
# Requires: tensorflow>=2.x

from __future__ import annotations

import math
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---- GPU config (optional, mirrors TF1 allow_growth) ----
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    # Safe to ignore if called after GPUs have been initialized or if no GPU present
    pass

# Lazily-initialized global model, like the original code's graph
_MODEL = None


def _get_inception_model() -> tf.keras.Model:
    """Load InceptionV3 (ImageNet) with the classification head (softmax)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = InceptionV3(include_top=True, weights="imagenet")
    return _MODEL


def _prepare_batch(batch_np: np.ndarray) -> tf.Tensor:
    """
    Resize to 299x299 and apply InceptionV3 preprocessing.
    Expects batch_np in [0,255], shape (N, H, W, 3), dtype float32/float64/int.
    """
    x = tf.convert_to_tensor(batch_np, dtype=tf.float32)
    x = tf.image.resize(x, (299, 299), method="bilinear")
    x = preprocess_input(x)  # scales to [-1, 1]
    return x


def get_inception_score(images: List[np.ndarray], splits: int = 10, batch_size: int = 128) -> Tuple[float, float]:
    """
    Compute the Inception Score for a list of images using TF2/Keras InceptionV3.

    Args:
        images: list of HxWx3 uint8/float arrays with values in [0, 255].
        splits: number of splits to compute mean/std over (default 10).
        batch_size: inference batch size (default 128).

    Returns:
        (mean_inception_score, std_inception_score)
    """
    # Basic sanity checks to mirror original behavior
    assert isinstance(images, list) and len(images) > 0
    assert isinstance(images[0], np.ndarray)
    assert len(images[0].shape) == 3 and images[0].shape[2] == 3
    assert np.max(images[0]) > 10
    assert np.min(images[0]) >= 0.0

    model = _get_inception_model()

    # Batched softmax predictions
    preds_list = []
    n = len(images)

    for start in tqdm(range(0, n, batch_size), desc="Calculate inception score"):
        end = min(start + batch_size, n)
        batch = np.stack([img.astype(np.float32) for img in images[start:end]], axis=0)
        x = _prepare_batch(batch)
        # model output is already softmax probabilities (1000 classes)
        probs = model(x, training=False).numpy()
        preds_list.append(probs)

    preds = np.concatenate(preds_list, axis=0)
    preds = np.clip(preds, 1e-16, 1.0)  # numerical stability

    # Compute scores over splits (same logic as original)
    scores = []
    # (Like the original code, we ignore the remainder if n % splits != 0)
    split_size = preds.shape[0] // splits
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size, :]
        py = np.expand_dims(np.mean(part, axis=0), 0)
        kl = part * (np.log(part) - np.log(py))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))
