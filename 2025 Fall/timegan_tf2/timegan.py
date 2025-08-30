# timegan_v2_compat.py
# TF2/Keras implementation aligned with the original TF1 TimeGAN

import numpy as np
import tensorflow as tf
from utils import extract_time, random_generator, batch_generator

# ----------------------------
# Helpers that mimic TF1 exactly
# ----------------------------

def pad_to_max_3d(ori_data, max_seq_len):
  """Pad a list of (T_i, D) arrays to (N, max_seq_len, D) with zeros."""
  if isinstance(ori_data, np.ndarray) and ori_data.ndim == 3:
    return ori_data  # already padded
  N = len(ori_data)
  D = int(np.asarray(ori_data[0]).shape[1])
  out = np.zeros((N, max_seq_len, D), dtype=np.float32)
  for i, seq in enumerate(ori_data):
    L = int(np.asarray(seq).shape[0])
    out[i, :L, :] = seq
  return out

def minmax_scaler_v1(padded_3d):
  """
  TF1-style MinMaxScaler used in the original code:
    min over (N,T) per feature, subtract; then max over (N,T) per feature (now range).
  Returns: norm_3d, min_val, range_val
  """
  # min over axes (0: batch, 1: time) -> shape (D,)
  min_val = np.min(np.min(padded_3d, axis=0), axis=0)
  shifted = padded_3d - min_val  # broadcast
  # "max_val" in TF1 code is actually the per-feature range after shift
  range_val = np.max(np.max(shifted, axis=0), axis=0)
  range_val[range_val == 0.0] = 1e-7
  norm = shifted / (range_val + 1e-7)
  return norm.astype(np.float32), min_val.astype(np.float32), range_val.astype(np.float32)

def inv_minmax_v1(padded_or_list, min_val, range_val):
  """Inverse TF1-style scaling: x * range + min."""
  if isinstance(padded_or_list, list):
    return [x * range_val + min_val for x in padded_or_list]
  return padded_or_list * range_val + min_val

def rnn_stack_v1(module: str, units: int, num_layers: int):
  """Plain GRU/LSTM stacks (no LayerNorm, no extras) with return_sequences=True."""
  module = module.lower()
  def one():
    if module == 'gru':
      return tf.keras.layers.GRU(units, activation='tanh', return_sequences=True)
    elif module == 'lstm':
      return tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True)
    else:
      # Default to LSTM to mirror TF1's typical fallback
      return tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True)
  return tf.keras.Sequential([one() for _ in range(num_layers)])

# --- Loss functions (unmasked), matching TF1 reductions ---

def mse_unmasked(a, b):
  return tf.reduce_mean(tf.square(a - b))

def bce_logits_unmasked(labels, logits):
  # labels/logits broadcastable; reduce mean over all dims
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  return tf.reduce_mean(loss)

def moment_loss_v1(x_hat, x):
  """
  Match the TF1 'two moments' exactly:
    - moments over axis=[0] (batch-only), keep (T,D) structure
    - then reduce_mean of absolute differences (with sqrt for std part)
  x_hat, x: (N,T,D)
  """
  mean_x,  var_x  = tf.nn.moments(x,     axes=[0])  # shapes: (T,D)
  mean_xh, var_xh = tf.nn.moments(x_hat, axes=[0])
  v1 = tf.reduce_mean(tf.abs(tf.sqrt(var_xh + 1e-6) - tf.sqrt(var_x + 1e-6)))
  v2 = tf.reduce_mean(tf.abs(mean_xh - mean_x))
  return v1 + v2

# ----------------------------
# Core model (modules only)
# ----------------------------

class TimeGANCore(tf.keras.Model):
  def __init__(self, dim, hidden_dim, num_layers, module):
    super().__init__()
    if num_layers < 2:
      raise ValueError("TF1 implementation uses supervisor with num_layers-1; require num_layers >= 2.")

    sup_layers = num_layers - 1

    # Embedder / Recovery
    self.embedder_rnn = rnn_stack_v1(module, hidden_dim, num_layers)
    self.embedder_proj = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')
    self.recovery_rnn = rnn_stack_v1(module, hidden_dim, num_layers)
    self.recovery_proj = tf.keras.layers.Dense(dim, activation='sigmoid')

    # Generator / Supervisor
    self.generator_rnn = rnn_stack_v1(module, hidden_dim, num_layers)
    self.generator_proj = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')
    self.supervisor_rnn = rnn_stack_v1(module, hidden_dim, sup_layers)
    self.supervisor_proj = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

    # Discriminator
    self.discriminator_rnn = rnn_stack_v1(module, hidden_dim, num_layers)
    self.discriminator_proj = tf.keras.layers.Dense(1, activation=None)

  # Subnets (pass mask to emulate TF1 dynamic_rnn(sequence_length))
  def embedder(self, x, mask):
    h = self.embedder_rnn(x, mask=mask)
    return self.embedder_proj(h)

  def recovery(self, h, mask):
    r = self.recovery_rnn(h, mask=mask)
    return self.recovery_proj(r)

  def generator(self, z, mask):
    g = self.generator_rnn(z, mask=mask)
    return self.generator_proj(g)

  def supervisor(self, h, mask):
    s = self.supervisor_rnn(h, mask=mask)
    return self.supervisor_proj(s)

  def discriminator(self, h, mask):
    d = self.discriminator_rnn(h, mask=mask)
    return self.discriminator_proj(d)

# ----------------------------
# Training procedure (phases)
# ----------------------------

def timegan(ori_data, parameters):
  """
  TF2/Keras version aligned with the TF1 reference implementation.

  Args:
    - ori_data: list of (T_i, D) arrays or a padded (N, T, D) array
    - parameters: dict with keys:
        hidden_dim, num_layer (>=2), iterations, batch_size, module
  Returns:
    - generated_data: list length N with arrays (T_i, D) (denormalized)
  """
  # Unpack
  hidden_dim  = int(parameters['hidden_dim'])
  num_layers  = int(parameters['num_layer'])
  iterations  = int(parameters['iterations'])
  batch_size  = int(parameters['batch_size'])
  module_name = str(parameters['module']).lower()
  gamma       = 1.0

  # Sequence lengths & max_len from original (pre-normalization) data
  ori_time, max_seq_len = extract_time(ori_data)
  no = len(ori_time)
  dim = int(np.asarray(ori_data[0]).shape[1]) if isinstance(ori_data, list) else int(ori_data.shape[2])

  # Pad to 3D (N, max_seq_len, D) BEFORE normalization (like TF1)
  ori_data_3d = pad_to_max_3d(ori_data, max_seq_len)

  # TF1-style MinMaxScaler on padded 3D
  ori_data_3d, min_val, range_val = minmax_scaler_v1(ori_data_3d)

  # Build model & optimizers
  model = TimeGANCore(dim, hidden_dim, num_layers, module_name)
  opt_E  = tf.keras.optimizers.Adam()
  opt_E0 = tf.keras.optimizers.Adam()
  opt_G  = tf.keras.optimizers.Adam()
  opt_GS = tf.keras.optimizers.Adam()
  opt_D  = tf.keras.optimizers.Adam()

  # Convenience
  z_dim = dim

  # -------- Phase 1: Embedding pretrain (autoencoder) --------
  print('Start Embedding Network Training')

  @tf.function
  def step_E(X_mb, mask_mb):
    with tf.GradientTape() as tape:
      H  = model.embedder(X_mb, mask_mb)
      Xt = model.recovery(H, mask_mb)
      E_loss_T0 = mse_unmasked(X_mb, Xt)
      E_loss0   = 10.0 * tf.sqrt(E_loss_T0 + 1e-8)
    vars_E = (model.embedder_rnn.trainable_variables + model.embedder_proj.trainable_variables +
              model.recovery_rnn.trainable_variables + model.recovery_proj.trainable_variables)
    grads = tape.gradient(E_loss0, vars_E)
    opt_E.apply_gradients(zip(grads, vars_E))
    return E_loss_T0

  for itt in range(iterations):
    X_mb, T_mb = batch_generator(ori_data_3d, ori_time, batch_size)
    mask_mb = tf.sequence_mask(T_mb, maxlen=max_seq_len)
    X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
    e_t0 = step_E(X_mb, mask_mb)
    if itt % 1000 == 0:
      print(f'step: {itt}/{iterations}, e_loss: {np.sqrt(float(e_t0.numpy())):.4f}')
  print('Finish Embedding Network Training')

  # -------- Phase 2: Supervised loss only (G + S) --------
  print('Start Training with Supervised Loss Only')

  @tf.function
  def step_GS(X_mb, Z_mb, mask_mb):
    with tf.GradientTape() as tape:
      H  = model.embedder(X_mb, mask_mb)
      Hs = model.supervisor(H, mask_mb)
      # supervised loss: H[:,1:,:] vs Hs[:,:-1,:]  (NO masking, matches TF1)
      G_loss_S = mse_unmasked(H[:, 1:, :], Hs[:, :-1, :])
    vars_GS = (model.generator_rnn.trainable_variables + model.generator_proj.trainable_variables +
               model.supervisor_rnn.trainable_variables + model.supervisor_proj.trainable_variables)
    grads = tape.gradient(G_loss_S, vars_GS)
    opt_GS.apply_gradients(zip(grads, vars_GS))
    return G_loss_S

  for itt in range(iterations):
    X_mb, T_mb = batch_generator(ori_data_3d, ori_time, batch_size)
    Z_mb = random_generator(len(X_mb), z_dim, T_mb, max_seq_len)
    mask_mb = tf.sequence_mask(T_mb, maxlen=max_seq_len)
    X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
    Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)
    g_s = step_GS(X_mb, Z_mb, mask_mb)
    if itt % 1000 == 0:
      print(f'step: {itt}/{iterations}, s_loss: {np.sqrt(float(g_s.numpy())):.4f}')
  print('Finish Training with Supervised Loss Only')

  # -------- Phase 3: Joint training --------
  print('Start Joint Training')

  @tf.function
  def step_G(X_mb, Z_mb, mask_mb):
    with tf.GradientTape() as tape:
      # Latent synthesis
      Ehat = model.generator(Z_mb, mask_mb)
      Hhat = model.supervisor(Ehat, mask_mb)

      # Adversarial (NO masking in loss)
      Y_fake   = model.discriminator(Hhat, mask_mb)
      Y_fake_e = model.discriminator(Ehat, mask_mb)
      G_loss_U   = bce_logits_unmasked(tf.ones_like(Y_fake),   Y_fake)
      G_loss_U_e = bce_logits_unmasked(tf.ones_like(Y_fake_e), Y_fake_e)

      # Supervised (NO masking)
      H  = model.embedder(X_mb, mask_mb)
      Hs = model.supervisor(H, mask_mb)
      G_loss_S = mse_unmasked(H[:, 1:, :], Hs[:, :-1, :])

      # Two moments on recovered sequences (TF1 axes=[0])
      Xhat = model.recovery(Hhat, mask_mb)
      G_loss_V = moment_loss_v1(Xhat, X_mb)

      # Total
      G_loss = G_loss_U + gamma * G_loss_U_e + 100.0 * tf.sqrt(G_loss_S + 1e-8) + 100.0 * G_loss_V

    vars_G = (model.generator_rnn.trainable_variables + model.generator_proj.trainable_variables +
              model.supervisor_rnn.trainable_variables + model.supervisor_proj.trainable_variables)
    grads = tape.gradient(G_loss, vars_G)
    opt_G.apply_gradients(zip(grads, vars_G))
    return G_loss_U, G_loss_S, G_loss_V

  @tf.function
  def step_E_refine(X_mb, Z_mb, mask_mb):
    with tf.GradientTape() as tape:
      H  = model.embedder(X_mb, mask_mb)
      Xt = model.recovery(H, mask_mb)
      E_loss_T0 = mse_unmasked(X_mb, Xt)
      Hs = model.supervisor(H, mask_mb)
      G_loss_S = mse_unmasked(H[:, 1:, :], Hs[:, :-1, :])
      E_loss = 10.0 * tf.sqrt(E_loss_T0 + 1e-8) + 0.1 * G_loss_S
    vars_E = (model.embedder_rnn.trainable_variables + model.embedder_proj.trainable_variables +
              model.recovery_rnn.trainable_variables + model.recovery_proj.trainable_variables)
    grads = tape.gradient(E_loss, vars_E)
    opt_E.apply_gradients(zip(grads, vars_E))
    return E_loss_T0

  @tf.function
  def compute_D_loss(X_mb, Z_mb, mask_mb):
    H_real = model.embedder(X_mb, mask_mb)
    Ehat   = model.generator(Z_mb, mask_mb)
    Hhat   = model.supervisor(Ehat, mask_mb)

    Y_real   = model.discriminator(H_real, mask_mb)
    Y_fake   = model.discriminator(Hhat, mask_mb)
    Y_fake_e = model.discriminator(Ehat, mask_mb)

    D_loss_real   = bce_logits_unmasked(tf.ones_like(Y_real),   Y_real)
    D_loss_fake   = bce_logits_unmasked(tf.zeros_like(Y_fake),  Y_fake)
    D_loss_fake_e = bce_logits_unmasked(tf.zeros_like(Y_fake_e),Y_fake_e)
    return D_loss_real + D_loss_fake + gamma * D_loss_fake_e

  @tf.function
  def step_D_apply(X_mb, Z_mb, mask_mb):
    with tf.GradientTape() as tape:
      D_loss = compute_D_loss(X_mb, Z_mb, mask_mb)
    vars_D = (model.discriminator_rnn.trainable_variables + model.discriminator_proj.trainable_variables)
    grads = tape.gradient(D_loss, vars_D)
    opt_D.apply_gradients(zip(grads, vars_D))
    return D_loss

  for itt in range(iterations):
    # Generator-side steps (twice)
    for _ in range(2):
      X_mb, T_mb = batch_generator(ori_data_3d, ori_time, batch_size)
      Z_mb = random_generator(len(X_mb), z_dim, T_mb, max_seq_len)
      mask_mb = tf.sequence_mask(T_mb, maxlen=max_seq_len)
      X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
      Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

      g_u, g_s, g_v = step_G(X_mb, Z_mb, mask_mb)
      e_t0          = step_E_refine(X_mb, Z_mb, mask_mb)

    # Discriminator step (only if needed)
    X_mb, T_mb = batch_generator(ori_data_3d, ori_time, batch_size)
    Z_mb = random_generator(len(X_mb), z_dim, T_mb, max_seq_len)
    mask_mb = tf.sequence_mask(T_mb, maxlen=max_seq_len)
    X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
    Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

    d_loss_val = compute_D_loss(X_mb, Z_mb, mask_mb)
    if float(d_loss_val.numpy()) > 0.15:
      d_loss_val = step_D_apply(X_mb, Z_mb, mask_mb)

    if itt % 1000 == 0:
      print(f'step: {itt}/{iterations}, '
            f'd_loss: {float(d_loss_val.numpy()):.4f}, '
            f'g_loss_u: {float(g_u.numpy()):.4f}, '
            f'g_loss_s: {np.sqrt(float(g_s.numpy())):.4f}, '
            f'g_loss_v: {float(g_v.numpy()):.4f}, '
            f'e_loss_t0: {np.sqrt(float(e_t0.numpy())):.4f}')
  print('Finish Joint Training')

  # -------- Synthesis (match lengths) --------
  Z = random_generator(no, z_dim, ori_time, max_seq_len)
  mask_all = tf.sequence_mask(ori_time, maxlen=max_seq_len)
  Z = tf.convert_to_tensor(Z, dtype=tf.float32)

  Hhat = model.supervisor(model.generator(Z, mask_all), mask_all)
  Xhat = model.recovery(Hhat, mask_all).numpy()  # (N,T,D)

  generated = []
  for i in range(no):
    L = int(ori_time[i])
    generated.append(Xhat[i, :L, :])

  # De-normalize (per-feature range + min), like TF1
  generated = inv_minmax_v1(generated, min_val, range_val)
  return generated
