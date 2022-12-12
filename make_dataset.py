import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom as pd
import os
import cv2

from dicom import MRImage
from dicom import *

def save_npy(npy_path, img, img_shape):
  """
  画像やマスクのndarrayを保存する。その際、すでにデータセットが存在している場合は、そのデータセットの末尾に追加する形で作成する。
  """
  if os.path.isfile(npy_path):
    exist_imgs = np.load(npy_path)
    img = np.append(exist_imgs, img)
    img = img.reshape(-1, *img_shape)
  np.save(npy_path, img)

def make_dataset(contour_dcm_path, contour_list, mr_dir_path, dt_type):

  """
  contour_dcm_math : 輪郭情報を持つDICOMファイルのパス
  contour_list     : マスク画像を作成したい輪郭の名前を保有するリスト
  mr_dir_path      : MRIが格納されているディレクトリのパス
  dt_type          : 画像のタイプ。movingかfixed
  """

  # contours辞書の作成
  rtss = pd.dcmread(contour_dcm_path)
  contours = make_contours(rtss, contour_list)

  # mrの作成
  mr = MRImage()
  mr.load(mr_dir_path)

  img_shape = mr.volume[0,:,:].shape
  dcm_num = len(mr.volume)  # dicomファイルの個数

  img_array = np.array([])
  mask_array = np.array([])
  counter = 0
  st = contours[contour_list[0]]
  for frame in range(dcm_num):

    # prostateのminimum所持数が30
    if counter >= 30:
      break

    if mr.z_array[frame] in st['paths']:
      # 画像を追加
      img = mr.volume[frame,:,:]
      img /= np.amax(img) # 画像を正規化
      img_array = np.append(img_array, img)

      dpi = img_shape[0] / 4.61   # 4.61はmatplotlibにより画像が小さくなることへの対応
      fig = plt.figure(figsize=(6,6), dpi=dpi)
      ax = fig.add_subplot(111)

      # マスク画像を作成する
      extent = (mr.x_min, mr.x_max, mr.y_max, mr.y_min)
      black = np.zeros(shape=img_shape)
      ax.imshow(black, cmap='gray', extent=extent)
      # ax.imshow(mr.volume[frame,:,:], cmap='gray', extent=extent)

      patch = patches.PathPatch(st['paths'][mr.z_array[frame]], fill=True, ec=st['ec'], fc=st['fc'], lw=2)
      ax.add_patch(patch)
      counter += 1

      ax.set_aspect('equal')
      ax.axis('off')
      # plt.show()
      plt.savefig('mask.png', facecolor='azure', bbox_inches='tight', pad_inches=0)
      plt.close()

      # 画像として保存したマスクをndarrayで呼び出して保存
      mask = cv2.imread('mask.png')
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      mask[mask != 0] = 255
      mask_array = np.append(mask_array, mask)

  os.remove('mask.png')

  img_array = img_array.reshape(-1, *img_shape)
  mask_array = mask_array.reshape(-1, *img_shape)

  # もしすでにあればその末尾に追加していく
  save_npy(f'../{dt_type}.npy', img_array, img_shape)
  save_npy(f'../{dt_type}_mask.npy', mask_array, img_shape)


if __name__ == '__main__':
  # dates = ['0829'] * 4 + ['0831'] * 4
  dates = ['0831', '0902', '0906', '0908', '0829', '0902', '0906', '0908']
  # dt_type = 'moving'
  dt_type = 'fixed'
  for date in dates:
    fr_image_path = '/home/uchiyama/work/VoxelMorph/MR-MR/16/FrImage'
    contour_dcm_path = f'{fr_image_path}/contour/dcm/contour_{date}.dcm'
    mr_dir_path = f'{fr_image_path}/{date}/dcm/'
    # contour_list = ['Bladder D+E', 'Rectum D+E', 'Prostate D+E', 'CTV D+E']
    contour_list = ['Prostate D+E']

    make_dataset(contour_dcm_path, contour_list, mr_dir_path, dt_type)

  moving = np.load('../fixed.npy')
  print(moving.shape)