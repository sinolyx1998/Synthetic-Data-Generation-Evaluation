"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator


def discriminative_score_metrics(ori_data, generated_data):
  """Use post-hoc RNN to classify original vs. synthetic sequences.
  Returns: abs(accuracy - 0.5)
  """
  # Basic parameters
  no, seq_len, dim = np.asarray(ori_data).shape

  # Sequence lengths & max length (fixes minor typo in the TF1 file)
  ori_time, ori_max_seq_len = extract_time(ori_data)
  gen_time, gen_max_seq_len = extract_time(generated_data)
  max_seq_len = max(ori_max_seq_len, gen_max_seq_len)

  # Post-hoc discriminator network (GRU + Dense)
  hidden_dim = max(1, dim // 2)
  iterations = 2000
  batch_size = 128

  gru = tf.keras.layers.GRU(hidden_dim, activation="tanh", return_sequences=False)
  out = tf.keras.layers.Dense(1, activation=None)  # logits
  opt = tf.keras.optimizers.Adam()

  bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def forward(x, t):
    """x: (N,T,D) float32, t: (N,) int32 lengths -> (logits, probs)."""
    # Build boolean mask from lengths for Keras RNN
    mask = tf.sequence_mask(t, maxlen=max_seq_len)
    h = gru(x, mask=mask)          # (N, hidden_dim)
    logits = out(h)                # (N, 1)
    probs = tf.sigmoid(logits)
    return logits, probs

  @tf.function
  def train_step(x_real, t_real, x_fake, t_fake):
    with tf.GradientTape() as tape:
      logit_real, _ = forward(x_real, t_real)
      logit_fake, _ = forward(x_fake, t_fake)
      # labels: real=1, fake=0
      loss_real = bce_logits(tf.ones_like(logit_real), logit_real)
      loss_fake = bce_logits(tf.zeros_like(logit_fake), logit_fake)
      loss = loss_real + loss_fake
    vars_d = gru.trainable_variables + out.trainable_variables
    grads = tape.gradient(loss, vars_d)
    opt.apply_gradients(zip(grads, vars_d))
    return loss

  # Train/test split (unchanged utility)
  train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
      train_test_divide(ori_data, generated_data, ori_time, gen_time)

  # Training loop
  for itt in range(iterations):
    X_mb, T_mb = batch_generator(train_x,      train_t,      batch_size)  # padded to max_seq_len
    Xh_mb,Th_mb= batch_generator(train_x_hat,  train_t_hat,  batch_size)
    X_mb  = tf.convert_to_tensor(X_mb,  dtype=tf.float32)
    T_mb  = tf.convert_to_tensor(T_mb,  dtype=tf.int32)
    Xh_mb = tf.convert_to_tensor(Xh_mb, dtype=tf.float32)
    Th_mb = tf.convert_to_tensor(Th_mb, dtype=tf.int32)
    _ = train_step(X_mb, T_mb, Xh_mb, Th_mb)

  # Evaluation on test set
  test_x  = tf.convert_to_tensor(test_x,  dtype=tf.float32)
  test_t  = tf.convert_to_tensor(test_t,  dtype=tf.int32)
  test_xh = tf.convert_to_tensor(test_x_hat, dtype=tf.float32)
  test_th = tf.convert_to_tensor(test_t_hat, dtype=tf.int32)

  _, y_real = forward(test_x,  test_t)
  _, y_fake = forward(test_xh, test_th)

  y_pred_final  = np.squeeze(np.concatenate([y_real.numpy(), y_fake.numpy()], axis=0))
  y_label_final = np.concatenate([np.ones(len(y_real)), np.zeros(len(y_fake))], axis=0)

  acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
  discriminative_score = abs(0.5 - acc)
  return discriminative_score