import numpy as np
import tensorflow as tf
import cv2
import os
import gc
import json
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 自作関数
from networks import Unet, VxmDense
from layers import SpatialTransformer
from losses import MSE, Grad
from utils import normalize, binarization, dsc

def plot_history(hist, data_path, nb_epochs, img_size, loss_name='loss'):
  plt.figure()
  plt.plot(hist.epoch, hist.history[loss_name], '.-')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.title('loss')
  plt.legend()
  plt.savefig(f'{data_path}/loss_{img_size}_{nb_epochs}.png')

def vxm_data_generator(moving, fixed, batch_size=8):
  vol_shape = moving.shape[1:]
  ndims = len(vol_shape)

  zero_phi = np.zeros([batch_size, *vol_shape, ndims])

  while True:
    idx = np.random.randint(0, moving.shape[0], size=batch_size)
    moving_images = moving[idx, ..., np.newaxis]
    fixed_images = fixed[idx, ..., np.newaxis]
    inputs = [moving_images, fixed_images]
    outputs = [fixed_images, zero_phi]

    yield(inputs, outputs)

def test_data_generator(moving, fixed, mov_mask, fix_mask, batch_size=8):
  while True:
    idx = np.random.randint(0, moving.shape[0], size=batch_size)
    moving_images = moving[idx, ..., np.newaxis]
    fixed_images = fixed[idx, ..., np.newaxis]
    inputs = [moving_images, fixed_images]
    moving_mask = mov_mask[idx, ..., np.newaxis]
    fixed_mask = fix_mask[idx, ..., np.newaxis]
    inputs_mask = [moving_mask, fixed_mask]

    yield(inputs, inputs_mask)

def split_data(data, nb_val, nb_test):
  data_val = data[-nb_val:, ...]
  data_test = data[-(nb_val + nb_test):-nb_val, ...]
  data_train = data[:-(nb_val + nb_test), ...]
  return data_train, data_val, data_test

def save_data(input, pred, mask_input, data_dir_path, img_size, kind):
  moving = normalize(input[0].squeeze())
  fixed = normalize(input[1].squeeze())
  moved = normalize(pred[0].squeeze())
  flow = pred[1]

  cv2.imwrite(f'{data_dir_path}/{kind}_moving_{img_size}.png', moving)
  cv2.imwrite(f'{data_dir_path}/{kind}_fixed_{img_size}.png', fixed)
  cv2.imwrite(f'{data_dir_path}/{kind}_moved_{img_size}.png', moved)
  np.save(f'{data_dir_path}/{kind}_input_{img_size}', input)
  np.save(f'{data_dir_path}/{kind}_flow_{img_size}', flow)

  mask_moving = normalize(mask_input[0].squeeze())
  mask_fixed = normalize(mask_input[1].squeeze())
  mask_moved = SpatialTransformer(interp_method='linear', indexing='ij', fill_value=None)([mask_input[0], flow])
  mask_moved = mask_moved.numpy().squeeze()
  np.save(f'{data_dir_path}/{kind}_mask_moving', mask_moving)
  np.save(f'{data_dir_path}/{kind}_mask_moved', mask_moved)
  np.save(f'{data_dir_path}/{kind}_mask_fixed', mask_fixed)

  mask_fixed = binarization(mask_fixed)
  mask_moved = binarization(mask_moved)
  return dsc(mask_fixed, mask_moved)

def main(ndim, bw_path, min_delta, patience, nb_epochs, steps_per_epoch, model_path, data_dir_path):

  # データのロード
  mov = np.load('../moving.npy')
  fix = np.load('../fixed.npy')
  mov_mask = np.load('../moving_mask.npy')
  fix_mask = np.load('../fixed_mask.npy')
  img_size = mov.shape[1]

  # データの分割
  nb_val = 1
  nb_test = 1
  moving, mov_val, mov_test = split_data(mov, nb_val, nb_test)
  fixed, fix_val, fix_test = split_data(fix, nb_val, nb_test)
  moving_mask, mov_mask_val, mov_mask_test = split_data(mov_mask, nb_val, nb_test)
  fixed_mask, fix_mask_val, fix_mask_test = split_data(fix_mask, nb_val, nb_test)

  print(moving.shape, mov_val.shape, mov_test.shape)

  # メモリの開放
  del mov, mov_mask, fix, fix_mask
  gc.collect()

  unet_input_features = 2
  inshape = (*moving.shape[1:], unet_input_features)

  nb_features = [
    [32, 32, 32, 32],
    [32, 32, 32, 32, 32, 16]
  ]

  inshape = moving.shape[1:]
  vxm_model = VxmDense(inshape, nb_features, int_steps=0)

  # lossの設定
  losses = [MSE().loss, Grad('l2').loss]
  lambda_param = 0.05
  loss_weights = [1, lambda_param]

  # モデルのコンパイル
  vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

  #EarlyStoppingを設定
  early_stopping = EarlyStopping(monitor='loss', min_delta=min_delta, patience=patience)
  # bestweightの保存先を設定
  checkpoint = ModelCheckpoint(filepath=bw_path, monitor='loss', verbose=1, save_best_only=True)

  train_generator = vxm_data_generator(moving, fixed)

  hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[early_stopping, checkpoint])
  vxm_model.save(model_path)

  if not os.path.exists(data_dir_path):
    os.makedirs(data_dir_path)
  plot_history(hist, data_dir_path, nb_epochs, img_size)

  vxm_model.load_weights(bw_path)

  dsc_json = {}
  val_generator = test_data_generator(mov_val, fix_val, mov_mask_val, fix_mask_val, batch_size=1)
  val_input, val_mask_input = next(val_generator)
  val_pred = vxm_model.predict(val_input)

  val_dsc = save_data(val_input, val_pred, val_mask_input, data_dir_path, img_size, 'val')
  dsc_json['val'] = val_dsc

  test_generator = test_data_generator(mov_test, fix_test, mov_mask_test, fix_mask_test, batch_size=1)
  test_input, test_mask_input = next(test_generator)
  test_pred = vxm_model.predict(test_input)

  test_dsc = save_data(test_input, test_pred, test_mask_input, data_dir_path, img_size, 'test')
  dsc_json['test'] = test_dsc

  with open(f'{data_dir_path}/dsc', 'w') as f:
    json.dump(dsc_json, f, ensure_ascii=False, indent=4)


if __name__ == ('__main__'):
  ndim = 2  # 二次元データ
  min_delta = 1e-6
  patience = 100
  nb_epochs = 1000
  steps_per_epoch = 20
  data_path = '/home/uchiyama/work/VoxelMorph/MR-MR/data'
  bw_path = f'{data_path}/bestweight/bestweight.hdf5'
  model_path = f'{data_path}/model.hdf5'
  data_dir_path = f'{data_path}/{nb_epochs}epochs'
  main(ndim, bw_path, min_delta, patience, nb_epochs, steps_per_epoch, model_path, data_dir_path)

