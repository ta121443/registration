import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI
import neurite as ne

from layers import RescaleTransform, VecInt, SpatialTransformer

class Unet(tf.keras.Model):
  def __init__(self, inshape=None, input_model=None, nb_features=None, nb_levels=None, max_pool=2, feat_mult=1, nb_conv_per_level=1, do_res=False, nb_upsample_skips=0, hyp_input=None, hyp_tensor=None, final_activation_function=None, kernel_initializer='he_normal', name='unet'):

    if input_model is None:
      if inshape is None:
        raise ValueError('inshape must be supplied if input_model is None')
      unet_input = KL.Input(shape=inshape, name='%s_input' % name)
      model_inputs = [unet_input]
    else:
      if len(input_model.outputs) == 1:
        unet_input = input_model.outputs[0]
      else:
        unet_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
      model_inputs = input_model.inputs

    if hyp_input is not None and not any([hyp_input is inp for inp in model_inputs]):
      model_inputs = model_inputs + [hyp_input]

    if nb_features is None:
      nb_features = default_unet_features()

    # nb_featuresがlistなら問題なし
    if isinstance(nb_features, int):
      if nb_levels is None:
        raise ValueError('must provide unet nb_levels if nb_features is an integer')
      feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
      nb_features = [
        np.repeat(feats[:-1], nb_conv_per_level),
        np.repeat(np.flip(feats), nb_conv_per_level)
      ]
    elif nb_levels is not None:
      raise ValueError('cannot use nb_levels if nb_features is not an integer')

    ndims = len(unet_input.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)   # <class 'keras.layers.pooling.max_pooling2d.MaxPooling2D'>

    # 余分なデコーダーを抽出する
    enc_nf, dec_nf = nb_features    # [[32, 32, 32, 32], [32, 32, 32, 32, 32, 16]]
    nb_dec_convs = len(enc_nf)    # 4
    final_convs = dec_nf[nb_dec_convs:]   #[32, 16]
    dec_nf = dec_nf[:nb_dec_convs]        #[32, 32, 32, 32]
    nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1   # 5

    if isinstance(max_pool, int):
      max_pool = [max_pool] * nb_levels   # [2, 2, 2, 2, 2]

    # エンコーダーの構築
    enc_layers = []
    last = unet_input   # KerasTensor(type_spec=TensorSpec(shape=(None, 512, 512, 2), dtype=tf.float32, name='unet_input'), name='unet_input', description="created by layer 'unet_input'")
    for level in range(nb_levels - 1):
      for conv in range(nb_conv_per_level):
        nf = enc_nf[level * nb_conv_per_level + conv]
        layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
        last = _conv_block(last, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor, kernel_initializer=kernel_initializer)
        enc_layers.append(last)

        last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

    # [<KerasTensor: shape=(None, 512, 512, 32) dtype=float32 (created by layer 'unet_enc_conv_0_0_activation')>, 
    # <KerasTensor: shape=(None, 512, 512, 32) dtype=float32 (created by layer 'unet_enc_conv_1_0_activation')>, 
    # <KerasTensor: shape=(None, 512, 512, 32) dtype=float32 (created by layer 'unet_enc_conv_2_0_activation')>, 
    # <KerasTensor: shape=(None, 512, 512, 32) dtype=float32 (created by layer 'unet_enc_conv_3_0_activation')>]

    # activate = <function Unet.__init__.<locals>.<lambda> at 0x7fa16ff82950
    if final_activation_function is not None and len(final_convs) == 0:
      activate = lambda lvl, c: not (lvl == (nb_levels - 2) and c == (nb_conv_per_level - 1))
    else:
      activate = lambda lvl, c: True

    # デコーダーの構築
    for level in range(nb_levels - 1):
      real_level = nb_levels - level - 2
      for conv in range(nb_conv_per_level):
        nf = dec_nf[level * nb_conv_per_level + conv]
        layer_name = '%s_dec_conv_%d_%d' % (name, real_level, conv)
        last = _conv_block(last, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor, include_activation=activate(level, conv), kernel_initializer=kernel_initializer)

        if level < (nb_levels - 1 - nb_upsample_skips):
          layer_name = '%s_dec_upsample_%d' % (name, real_level)
          last = _upsample_block(last, enc_layers.pop(), factor=max_pool[real_level], name=layer_name)

    if final_activation_function is not None:
      activate = lambda n: n != (len(final_convs) - 1)
    else:
      activate = lambda n: True

    for num, nf in enumerate(final_convs):
      layer_name = '%s_dec_final_conv_%d' % (name, num)
      last = _conv_block(last, nf, name=layer_name, hyp_tensor=hyp_tensor, include_activation=activate(num), kernel_initializer=kernel_initializer)

    if final_activation_function is not None:
      last = KL.Activation(final_activation_function, name='%s_final_activation' % name)(last)

    super().__init__(inputs=model_inputs, outputs=last, name=name)

class VxmDense(ne.modelio.LoadableModel):

  @ne.modelio.store_config_args
  def __init__(self,
               inshape,
               nb_unet_features=None,
               nb_unet_levels=None,
               unet_feat_mult=1,
               nb_unet_conv_per_level=1,
               int_steps=7,
               svf_resolution=1,
               int_resolution=2,
               int_downsize=None,
               bidir=False,
               use_probs=False,
               src_feats=1,
               trg_feats=1,
               unet_half_res=False,
               input_model=None,
               hyp_model=None,
               fill_value=None,
               reg_field='preintegrated',
               name='vxm_dense'):

    ndims = len(inshape)  # 2次元なら2、3次元なら3
    if input_model is None:
      # configure default input layers if an input model is not provided
      source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
      target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
      input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
    else:
      source, target = input_model.outputs[:2]

    inputs = input_model.inputs # [<KerasTensor: shape=(None, 512, 512, 1) dtype=float32 (created by layer 'vxm_dense_source_input')>, 
                                #  <KerasTensor: shape=(None, 512, 512, 1) dtype=float32 (created by layer 'vxm_dense_target_input')>]

    inputs = input_model.inputs
    if hyp_model is not None:
      hyp_input = hyp_model.input
      hyp_tensor = hyp_model.output
      if not any([hyp_input is inp for inp in inputs]):
        inputs = (*inputs, hyp_input)
    else:
      hyp_input = None
      hyp_tensor = None

    if int_downsize is not None:
      warnings.warn('int_downsize is deprecated, use the int_resolution parameter.')
      int_resolution = int_downsize

    if unet_half_res:
      warnings.warn('unet_half_res is deprecated, use the svf_resolution parameter.')
      svf_resolution = 2

    nb_upsample_skips = int(np.floor(np.log(svf_resolution) / np.log(2))) #0

    unet_model = Unet(
      input_model=input_model,
      nb_features=nb_unet_features,
      nb_levels=nb_unet_levels,
      feat_mult=unet_feat_mult,
      nb_conv_per_level=nb_unet_conv_per_level,
      nb_upsample_skips=nb_upsample_skips,
      hyp_input=hyp_input,
      hyp_tensor=hyp_tensor,
      name='%s_unet' % name
    )

    # Unetの出力を変形場に出力する
    Conv = getattr(KL, 'Conv%dD' % ndims)   # <class 'keras.layers.convolutional.conv2d.Conv2D'>
    flow_mean = Conv(ndims, kernel_size=3, padding='same',
                     kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)

    if use_probs:
      # initialize the velocity variance very low, to start stable
      flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='%s_log_sigma' % name)(unet_model.output)
      flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
      flow_inputs = [flow_mean, flow_logsigma]
      flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)(flow_inputs)
    else:
      flow = flow_mean

    pre_svf_size = np.array(flow.shape[1:-1])
    svf_size = np.array([np.round(dim / svf_resolution) for dim in inshape])
    if not np.array_equal(pre_svf_size, svf_size):
        rescale_factor = svf_size[0] / pre_svf_size[0]
        flow = RescaleTransform(rescale_factor, name=f'{name}_svf_resize')(flow)

    svf = flow

    if int_steps > 0 and int_resolution > 1:
      int_size = np.array([np.round(dim / int_resolution) for dim in inshape])
      if not np.array_equal(svf_size, int_size):
        rescale_factor = int_size[0] / svf_size[0]
        flow = RescaleTransform(rescale_factor, name=f'{name}_flow_resize')(flow)

    # pre-integrated flow
    preint_flow = flow

    # 双方向ネットワークにも対応
    pos_flow = flow
    if bidir:
      neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

    if int_steps > 0:
      pos_flow = VecInt(method='ss',
                               name='%s_flow_int' % name,
                               int_steps=int_steps)(pos_flow)
      if bidir:
        neg_flow = VecInt(method='ss',
                                 name='%s_flow_int' % name,
                                 int_steps=int_steps)(neg_flow)

    postint_flow = pos_flow   # (512, 512, 2)

    # resize to final resolution
    if int_steps > 0 and int_resolution > 1:
      rescale_factor = inshape[0] / int_size[0]
      pos_flow = RescaleTransform(rescale_factor, name='%s_diffflow' % name)(pos_flow)
      if bidir:
        neg_flow = RescaleTransform(rescale_factor,
                                            name='%s_neg_diffflow' % name)(neg_flow)


    # 画像を変形する
    y_source = SpatialTransformer(
        interp_method='linear',
        indexing='ij',
        fill_value=fill_value,
        name='%s_transformer' % name)([source, pos_flow])

    if bidir:
      st_inputs = [target, neg_flow]
      y_target = SpatialTransformer(interp_method='linear',
                                    indexing='ij',
                                    fill_value=fill_value,
                                    name='%s_neg_transformer' % name)(st_inputs)


    outputs = [y_source, y_target] if bidir else [y_source]

    reg_field = reg_field.lower()   # preintegrated .lower()で文字列を小文字に変換

    if use_probs:
      # compute loss on flow probabilities
      outputs.append(flow_params)
    elif reg_field == 'svf':
      # regularize the immediate, predicted SVF
      outputs.append(svf)
    elif reg_field == 'preintegrated':
      # regularize the rescaled, pre-integrated SVF
      outputs.append(preint_flow)
    elif reg_field == 'postintegrated':
      # regularize the rescaled, integrated field
      outputs.append(postint_flow)
    elif reg_field == 'warp':
      # regularize the final, full-resolution deformation field
      outputs.append(pos_flow)
    else:
      raise ValueError(f'Unknown option "{reg_field}" for reg_field')

    super().__init__(name=name, inputs=inputs, outputs=outputs)

    self.references = ne.modelio.LoadableModel.ReferenceContainer()
    self.references.unet_model = unet_model
    self.references.source = source
    self.references.target = target
    self.references.svf = svf
    self.references.preint_flow = preint_flow
    self.references.postint_flow = postint_flow
    self.references.pos_flow = pos_flow
    self.references.neg_flow = neg_flow if bidir else None
    self.references.y_source = y_source
    self.references.y_target = y_target if bidir else None
    self.references.hyp_input = hyp_input

  def get_registration_model(self):
    # Returns a reconfigured model to predict only the final transform.
    return tf.keras.Model(self.inputs, self.references.pos_flow)

  def register(self, src, trg):
      # Predicts the transform from src to trg tensors.
      return self.get_registration_model().predict([src, trg])

  def apply_transform(self, src, trg, img, interp_method='linear'):
      # Predicts the transform from src to trg and applies it to the img tensor.
      warp_model = self.get_registration_model()
      img_input = tf.keras.Input(shape=img.shape[1:])
      st_input = [img_input, warp_model.output]
      y_img = SpatialTransformer(interp_method=interp_method)(st_input)
      return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

  # print(self.references.unet_model)


def _conv_block(x, nfeat, strides=1, name=None, do_res=False, hyp_tensor=None, include_activation=True, kernel_initializer='he_nomal'):
  ndims = len(x.get_shape()) - 2
  assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims

  extra_conv_params = {}

  # hyp_tensorが必要になったときのみ使用
  # if hyp_tensor is not None:
  #   # 今はいったん放置
  # else:

  if hyp_tensor is None:
    Conv = getattr(KL, 'Conv%dD' % ndims)
    extra_conv_params['kernel_initializer'] = kernel_initializer
    conv_inputs = x

  convolved = Conv(nfeat, kernel_size=3, padding='same', strides=strides, name=name, **extra_conv_params)(conv_inputs)

  if include_activation:
    name = name + '_activation' if name else None
    convolved = KL.LeakyReLU(0.2, name=name)(convolved)
  
  return convolved

def _upsample_block(x, connection, factor=2, name=None):

  ndims = len(x.get_shape()) - 2
  assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
  UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

  size = (factor,) * ndims if ndims > 1 else factor
  upsampled = UpSampling(size=size, name=name)(x)
  name = name + '_concat' if name else None
  return KL.concatenate([upsampled, connection], name=name)
  