import tensorflow as tf
import tensorflow.keras.backend as K

class MSE:

  def __init__(self, image_sigma=1.0):
    self.image_sigma = image_sigma

  def mse(self, y_true, y_pred):
    return K.square(y_true - y_pred)

  def loss(self, y_true, y_pred, reduce='mean'):
    y_true = y_true[:,150:430,...]
    y_pred = y_pred[:,150:430,...]
    mse = self.mse(y_true, y_pred)
    if reduce == 'mean':
      mse = K.mean(mse)
    elif reduce == 'max':
      mse = K.max(mse)
    elif reduce is not None:
      raise ValueError(f'Unknown MSE reduction type: {reduce}')

    return 1.0 / (self.image_sigma ** 2) * mse

class Grad:
  def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
    self.penalty = penalty
    self.loss_mult = loss_mult
    self.vox_weight = vox_weight

  def _diffs(self, y):
    y = y[:,200:430,...]
    vol_shape = y.get_shape().as_list()[1:-1]
    ndims = len(vol_shape)

    df = [None] * ndims
    for i in range(ndims):
      d = i + 1
      # permute dimensions to put the ith dimension first
      r = [d, *range(d), *range(d + 1, ndims + 2)]
      yp = K.permute_dimensions(y, r)
      dfi = yp[1:, ...] - yp[:-1, ...]

      if self.vox_weight is not None:
          w = K.permute_dimensions(self.vox_weight, r)

          dfi = w[1:, ...] * dfi

      r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
      df[i] = K.permute_dimensions(dfi, r)

    return df

  def loss(self, _, y_pred):

    if self.penalty == 'l1':
      dif = [tf.abs(f) for f in self._diffs(y_pred)]
    else:
      assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
      dif = [f * f for f in self._diffs(y_pred)]

    df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
    grad = tf.add_n(df) / len(df)

    if self.loss_mult is not None:
      grad *= self.loss_mult

    return grad