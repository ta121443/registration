from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
# 自作関数
from utils import *

class SpatialTransformer(Layer):
  def __init__(self,
                interp_method='linear',
                indexing='ij',
                single_transform=False,
                fill_value=None,
                shift_center=True,
                shape=None,
                **kwargs):
    self.interp_method = interp_method
    assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
    self.indexing = indexing
    self.single_transform = single_transform
    self.fill_value = fill_value
    self.shift_center = shift_center
    self.shape = shape
    super().__init__(**kwargs) 

  def config(self):
    config = super().get_config().copy()
    config.update({
        'interp_method': self.interp_method,
        'indexing': self.indexing,
        'single_transform': self.single_transform,
        'fill_value': self.fill_value,
        'shift_center': self.shift_center,
        'shape': self.shape,
    })
    return config

  def build(self, input_shape):

    if len(input_shape) > 2:
      raise ValueError('Spatial Transformer must be called on a list of length 2: '
                       'first argument is the image, second is the transform')

    self.ndims = len(input_shape[0]) - 2
    self.imshape = input_shape[0][1:]
    self.trfshape = input_shape[1][1:]
    # self.is_affine = utils.is_affine_shape(input_shape[1][1:])
    self.is_affine = is_affine_shape(input_shape[1][1:])

    if self.is_affine:
      expected = (self.ndims, self.ndims + 1)
      actual = tuple(self.trfshape[-2:])
      if expected != actual:
        raise ValueError(f'Expected {expected} affine matrix, got {actual}')
    else:
      image_shape = tuple(self.imshape[:-1])
      dense_shape = tuple(self.trfshape[:-1])
      if image_shape != dense_shape:
        warnings.warn(f'Dense transform shape {dense_shape} does not match '
                      f'image shape {image_shape}.')

    self.built = True

  def call(self, inputs):
    vol = K.reshape(inputs[0], (-1, *self.imshape))
    trf = K.reshape(inputs[1], (-1, *self.trfshape))

    if self.is_affine:
      shape = vol.shape[1:-1] if self.shape is None else self.shape
      # fun = lambda x: utils.affine_to_dense_shift(x, shape, shift_center=self.shift_center, indexing=self.indexing)
      fun = lambda x: affine_to_dense_shift(x, shape, shift_center=self.shift_center, indexing=self.indexing)

      trf = tf.map_fn(fun, trf)

    if self.indexing == 'xy':
      trf_split = tf.split(trf, trf.shape[-1], axis=-1)
      trf_list = [trf_split[1], trf_split[0], *trf_split[2:]]
      trf = tf.concat(trf_list, -1)

    if self.single_transform:
      return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]), vol)
    else:
      return tf.map_fn(self._single_transform, [vol, trf], fn_output_signature=vol.dtype)

  def _single_transform(self, inputs):
    # return utils.transform(inputs[0], inputs[1], interp_method=self.interp_method, fill_value=self.fill_value)
    return transform(inputs[0], inputs[1], interp_method=self.interp_method, fill_value=self.fill_value)

class RescaleTransform(Layer):

  def __init__(self, zoom_factor, interp_method='linear', **kwargs):
    self.zoom_factor = zoom_factor
    self.interp_method = interp_method
    super().__init__(**kwargs)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'zoom_factor': self.zoom_factor,
      'interp_method': self.interp_method
    })
    return config

  def build(self, input_shape):
    self.is_affine = is_affine_shape(input_shape[1:])
    self.ndims = input_shape[-1] - 1 if self.is_affine else input_shape[-1]

  def compute_output_shape(self, input_shape):
    if self.is_affine:
      return (input_shape[0], self.ndims, self.ndims + 1)
    else:
      shape = [int(d * self.zoom_factor) for d in input_shape[1:-1]]
      return (input_shape[0], *shape, self.ndims)

  def call(self, transform):
    if self.is_affine:
      return rescale_affine(transform, self.zoom_factor)
    else:
      return rescale_dense_transform(transform, self.zoom_factor, interp_method=self.interp_method)

class VecInt(Layer):
  def __init__(self,
               indexing='ij',
               method='ss',
               int_steps=7,
               out_time_pt=1,
               ode_args=None,
               odeint_fn=None,
               **kwags):

    assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
    self.indexing = indexing
    self.method = method
    self.int_steps = int_steps
    self.inshape = None
    self.out_time_pt = out_time_pt
    self.odeint_fn = odeint_fn
    self.ode_args = ode_args
    if ode_args is None:
      self.ode_args = {'rtol': 1e-6, 'atol': 1e-12}
    super(self.__class__, self).__init__(**kwags)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'indexing': self.indexing,
      'method': self.method,
      'int_steps': self.int_steps,
      'out_time_pt': self.out_time_pt,
      'ode_args': self.ode_args,
      'odeint_fn': self.odeint_fn,
    })
    return config

  def build(self, input_shape):
    self.built = True

    trf_shape = input_shape
    if isinstance(input_shape[0], (list, tuple)):
      trf_shape = input_shape[0]
    self.inshape = trf_shape

    if trf_shape[-1] != len(trf_shape) - 2:
      raise Exception('transform ndims %d does not match expected ndims %d' % (trf_shape[-1], len(trf_shape) - 2))

  def call(self, inputs):
    if not isinstance(inputs, (list, tuple)):
      inputs = [inputs]
    loc_shift = inputs[0]

    loc_shift = K.reshape(loc_shift, [-1, *self.inshape[1:]])
    if hasattr(inputs[0], '_keras_shape'):
      loc_shift._keras_shape = inputs[0]._keras_shape

    return out

  def _single_int(self, inputs):
    vel = inputs[0]
    out_time_pt = self.out_time_pt
    if len(inputs) == 2:
      out_time_pt = inputs[1]
    return integrate_vec(vel, method=self.method, nb_steps=self.int_steps, ode_args=self.ode_args, out_time_pt=self.out_time_pt, odeint_fn=self.odeint_fn)