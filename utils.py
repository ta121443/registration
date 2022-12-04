import tensorflow as tf
import neurite as ne

# マスク画像を二値化する
def binarization(img):
  threshold = 120
  img[img < threshold] = 0
  img[img >= threshold] = 255
  return img

# マスク画像間の一致度を計算する関数
def dsc(fixed_mask, moved_mask):
  side = fixed_mask.shape[-1]
  same = 0
  one = 0
  for i in range(side):
    for j in range(side):
      if fixed_mask[i][j] == 255:
        one += 1
        if moved_mask[i][j] == 255:
          one += 1
          same += 1
      elif moved_mask[i][j] == 255:
        one += 1
  return (2*same) / one

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

def is_affine_shape(shape):
  if len(shape) == 2 and shape[-1] != 1:
    validate_affine_shape(shape)
    return True
  return False

def rescale_affine(mat, factor):
  scaled_translation = tf.expand_dims(mat[..., -1] * factor, -1)
  scaled_matrix = tf.concat([mat[..., :-1], scaled_translation], -1)
  return scaled_matrix

def rescale_dense_transform(transform, factor, interp_method='linear'):
  def signle_batch(trf):
    if factor < 1:
      trf = ne.utils.resize(trf, factor, interp_method=interp_method)
      trf = trf * factor
    else:
      trf = trf * factor
      trf = ne.utils.resize(trf, factor, interp_method=interp_method)
    return trf

  if len(transform.shape) > (transform.shape[-1] + 1):
    rescaled = tf.map_fn(signle_batch, transform)
  else:
    rescaled = signle_batch(transform)

  return rescaled

def integrate_vec(vec, time_dep=False, method='ss', **kwargs):

  if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
    raise ValueError("method has to be 'scaling_and_squaring' or 'ode'. found: %s" % method)

  if method in ['ss', 'scaling_and_squaring']:
    nb_steps = kwargs['nb_steps']
    assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

    if time_dep:
      svec = K.permute_dimensions(vec, [-1, *range(0, vec.shape[-1] - 1)])
      assert 2**nb_steps == svec.shape[0], "2**nb_steps and vector shape don't match"

      svec = svec / (2**nb_steps)
      for _ in range(nb_steps):
        svec = svec[0::2] + tf.map_fn(transform, svec[1::2, :], svec[0::2, :])

      disp = svec[0, :]

    else:
      vec = vec / (2**nb_steps)
      for _ in range(nb_steps):
        vec += transform(vec, vec)
      disp = vec

  elif method == 'quadrature':

    nb_steps = kwargs['nb_steps']
    assert nb_steps >= 1, 'nb_steps should be >= 1, found: %d' % nb_steps

    vec = vec / nb_steps

    if time_dep:
      disp = vec[..., 0]
      for si in range(nb_steps - 1):
        disp += transform(vec[..., si + 1], disp)
    else:
      disp = vec
      for _ in range(nb_steps - 1):
        disp += transform(vec, disp)

  else:
    assert not time_dep, "odeint not implemented with time-dependent vector field"
    fn = lambda disp, _: transform(vec, disp)

    # process time point.
    out_time_pt = kwargs['out_time_pt'] if 'out_time_pt' in kwargs.keys() else 1
    out_time_pt = tf.cast(K.flatten(out_time_pt), tf.float32)
    len_out_time_pt = out_time_pt.get_shape().as_list()[0]
    assert len_out_time_pt is not None, 'len_out_time_pt is None :('
    # initializing with something like tf.zeros(1) gives a control flow issue.
    z = out_time_pt[0:1] * 0.0
    K_out_time_pt = K.concatenate([z, out_time_pt], 0)

    # enable a new integration function than tf.contrib.integrate.odeint
    odeint_fn = tf.contrib.integrate.odeint
    if 'odeint_fn' in kwargs.keys() and kwargs['odeint_fn'] is not None:
      odeint_fn = kwargs['odeint_fn']

    # process initialization
    if 'init' not in kwargs.keys() or kwargs['init'] == 'zero':
      disp0 = vec * 0  # initial displacement is 0
    else:
      raise ValueError('non-zero init for ode method not implemented')

    # compute integration with odeint
    if 'ode_args' not in kwargs.keys():
        kwargs['ode_args'] = {}
    disp = odeint_fn(fn, disp0, K_out_time_pt, **kwargs['ode_args'])
    disp = K.permute_dimensions(disp[1:len_out_time_pt + 1, :], [*range(1, len(disp.shape)), 0])

    # return
    if len_out_time_pt == 1:
      disp = disp[..., 0]

  return disp

def transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None, shift_center=True, shape=None):
  ftype = tf.float32
  if not tf.is_tensor(vol) or not vol.dtype.is_floating:
    vol = tf.cast(vol, ftype)
  if not tf.is_tensor(loc_shift) or not loc_shift.dtype.is_floating:
    loc_shift = tf.cast(loc_shift, ftype)

  # convert affine to location shift (will validate affine shape)
  if is_affine_shape(loc_shift.shape):
    loc_shift = affine_to_dense_shift(loc_shift,
                                      shape=vol.shape[1:-1] if shape is None else shape,
                                      shift_center=shift_center,
                                      indexing=indexing)

  # parse spatial location shape, including channels if available
  loc_volshape = loc_shift.shape[:-1]
  if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
    loc_volshape = loc_volshape.as_list()

  # volume dimensions
  nb_dims = len(vol.shape) - 1
  is_channelwise = len(loc_volshape) == (nb_dims + 1)
  assert loc_shift.shape[-1] == nb_dims, \
    'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
    'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

  # location should be mesh and delta
  mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing=indexing)  # volume mesh
  for d, m in enumerate(mesh):
    if m.dtype != loc_shift.dtype:
      mesh[d] = tf.cast(m, loc_shift.dtype)
  loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

  # if channelwise location, then append the channel as part of the location lookup
  if is_channelwise:
    loc.append(mesh[-1])

  # test single
  return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)

def affine_to_dense_shift(matrix, shape, shift_center=True, indexing='ij'):
  if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
    shape = shape.as_list()

  if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
    matrix = tf.cast(matrix, tf.float32)

  # check input shapes
  ndims = len(shape)
  if matrix.shape[-1] != (ndims + 1):
    matdim = matrix.shape[-1] - 1
    raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
  validate_affine_shape(matrix.shape)

  # list of volume ndgrid
  # N-long list, each entry of shape
  mesh = ne.utils.volshape_to_meshgrid(shape, indexing=indexing)
  mesh = [f if f.dtype == matrix.dtype else tf.cast(f, matrix.dtype) for f in mesh]

  if shift_center:
    mesh = [mesh[f] - (shape[f] - 1) / 2 for f in range(len(shape))]

  # add an all-ones entry and transform into a large matrix
  flat_mesh = [ne.utils.flatten(f) for f in mesh]
  flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype=matrix.dtype))
  mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

  # compute locations
  loc_matrix = tf.matmul(matrix, mesh_matrix)  # N+1 x nb_voxels
  loc_matrix = tf.transpose(loc_matrix[:ndims, :])  # nb_voxels x N
  loc = tf.reshape(loc_matrix, list(shape) + [ndims])  # *shape x N

# 画像を標準化する
def normalize(img):
  img *= 255
  img[img < 0] = 0
  img[img > 255] = 255
  return img