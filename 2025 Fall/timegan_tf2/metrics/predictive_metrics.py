"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import extract_time

 
def _pad_XY_batches(X_list, Y_list, T_list, max_len, feat_dim):
    """Left-align pad X (…, feat_dim) and Y (…, 1) to (B, max_len, *) and return arrays + lengths."""
    B = len(X_list)
    Xb = np.zeros((B, max_len, feat_dim), dtype=np.float32)
    Yb = np.zeros((B, max_len, 1), dtype=np.float32)
    Tb = np.asarray(T_list, dtype=np.int32)
    for i, (x, y, t) in enumerate(zip(X_list, Y_list, Tb)):
        if t > 0:
            Xb[i, :t, :] = x[:t]
            Yb[i, :t, 0] = y[:t]
    return Xb, Yb, Tb


def _masked_mae(y_true, y_pred, lengths, max_len):
    """Mean Absolute Error over valid steps only."""
    mask = tf.sequence_mask(lengths, maxlen=max_len)          # (B, T)
    mask = tf.cast(mask, tf.float32)[..., None]               # (B, T, 1)
    abs_err = tf.abs(y_true - y_pred) * mask                  # (B, T, 1)
    num = tf.reduce_sum(abs_err)
    den = tf.reduce_sum(mask) + 1e-8
    return num / den


def predictive_score_metrics(ori_data, generated_data):
    """
    Report MAE of a post-hoc RNN one-step-ahead predictor trained on synthetic data.
    Returns:
      predictive_score: float (MAE on original data)
    """
    # Basic parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Sequence lengths & max length (fix minor bug in TF1: use generated_data for generated_time)
    ori_time, ori_max_seq_len = extract_time(ori_data)
    gen_time, gen_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, gen_max_seq_len)

    # Network hyperparams (kept consistent with original)
    hidden_dim = max(1, dim // 2)
    iterations = 5000
    batch_size = 128

    # Keras layers & optimizer
    gru = tf.keras.layers.GRU(hidden_dim, activation="tanh", return_sequences=True)
    head = tf.keras.layers.Dense(1, activation="sigmoid")  # outputs in [0,1] like original
    opt = tf.keras.optimizers.Adam()

    # Forward pass with explicit mask
    def forward(x_batch, lengths):
        mask = tf.sequence_mask(lengths, maxlen=max_seq_len - 1)  # (B, T-1)
        h = gru(x_batch, mask=mask)                               # (B, T-1, H)
        y_hat = head(h)                                           # (B, T-1, 1)
        return y_hat

    @tf.function
    def train_step(xb, yb, tb):
        with tf.GradientTape() as tape:
            y_hat = forward(xb, tb)
            loss = _masked_mae(yb, y_hat, tb, max_len=max_seq_len - 1)
        vars_p = gru.trainable_variables + head.trainable_variables
        grads = tape.gradient(loss, vars_p)
        opt.apply_gradients(zip(grads, vars_p))
        return loss

    # -------- Train the predictor on synthetic data --------
    for itt in range(iterations):
        idx = np.random.permutation(len(generated_data))[:batch_size]

        # Build a minibatch: X=all features except last; Y=next-step of last feature
        X_list = [generated_data[i][:-1, :dim-1] for i in idx]
        T_list = [max(0, int(gen_time[i]) - 1) for i in idx]
        Y_list = [generated_data[i][1:, dim-1] for i in idx]  # (T-1,)

        X_mb, Y_mb, T_mb = _pad_XY_batches(X_list, Y_list, T_list, max_len=max_seq_len - 1, feat_dim=dim - 1)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        Y_mb = tf.convert_to_tensor(Y_mb, dtype=tf.float32)
        T_mb = tf.convert_to_tensor(T_mb, dtype=tf.int32)

        _ = train_step(X_mb, Y_mb, T_mb)

    # -------- Evaluate MAE on the original data --------
    idx = np.arange(len(ori_data))  # same as original: evaluate over all
    X_list = [ori_data[i][:-1, :dim-1] for i in idx]
    T_list = [max(0, int(ori_time[i]) - 1) for i in idx]
    Y_list = [ori_data[i][1:, dim-1] for i in idx]

    X_b, Y_b, T_b = _pad_XY_batches(X_list, Y_list, T_list, max_len=max_seq_len - 1, feat_dim=dim - 1)
    X_b = tf.convert_to_tensor(X_b, dtype=tf.float32)
    T_b = tf.convert_to_tensor(T_b, dtype=tf.int32)

    Y_pred = forward(X_b, T_b).numpy()  # (N, T-1, 1)

    # Per-sequence MAE averaged (matches original evaluation style)
    mae_sum = 0.0
    for i in range(no):
        L = int(T_list[i])
        if L > 0:
            y_true_i = Y_b[i, :L, 0]
            y_pred_i = Y_pred[i, :L, 0]
            mae_sum += mean_absolute_error(y_true_i, y_pred_i)
    predictive_score = mae_sum / max(1, sum(1 for t in T_list if t > 0))

    return predictive_score