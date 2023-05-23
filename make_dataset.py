import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom as pd
import os
import gc
import cv2

from dicom import *

def around_prostate(prostate_st, dcm_num, mr, st, img_shape):
  img_array = np.array([])
  mask_array = np.array([])

  prostate_start = 0
  for i in range(dcm_num):
    if mr.z_array[i] in prostate_st['paths']:
      if i >= 10:
        prostate_start = i - 10
        break
      else:
        prostate_start = i
        break

  for frame in range(prostate_start, prostate_start+50):
    img = mr.volume[frame,:,:]
    img /= np.amax(img)
    img_array = np.append(img_array, img)

    dpi = img_shape[0] / 4.61
    fig = plt.figure(figsize=(6,6), dpi=dpi)
    ax = fig.add_subplot(111)

    extent = (mr.x_min, mr.x_max, mr.y_max, mr.y_min)
    black = np.zeros(shape=img_shape)
    ax.imshow(black, cmap='gray', extent=extent)

    if mr.z_array[frame] in st['paths']:
      patch = patches.PathPatch(st['paths'][mr.z_array[frame]], fill=True, ec=st['ec'], fc=st['fc'], lw=2)
      ax.add_patch(patch)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig('mask.png', facecolor='azure', bbox_inches='tight', pad_inches=0)
    plt.close()

    mask = cv2.imread('mask.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask != 0] = 255
    mask_array = np.append(mask_array, mask)

  return img_array, mask_array

def organs_specific(st, dcm_num, mr):
  img_array = np.array([])
  mask_array = np.array([])

  for frame in range(dcm_num):
    if mr.z_array[frame] in st['paths']:
      img = mr.volume[frame,:,:]
      img /= np.amax(img)
      img_array = np.append(img_array, img)

      dpi = img_shape[0] / 4.61
      fig = plt.figure(figsize=(6,6), dpi=dpi)
      ax = fig.add_subplot(111)

      extent = (mr.x_min, mr.x_max, mr.y_max, mr.y_min)
      black = np.zeros(shape=img_shape)
      ax.imshow(black, cmap='gray', extent=extent)
      patch = patches.PathPatch(st['paths'][mr.z_array[frame]], fill=True, ec=st['ec'], fc=st['fc'], lw=2)
      ax.add_patch(patch)



  return img_array, mask_array


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

  dcm_num = len(mr.volume)  # dicomファイルの個数
  img_shape = mr.volume[0,:,:].shape

  prostate_st = contours[contour_list[-1]]
  st = contours[contour_list[0]]

  img_array, mask_array = around_prostate(prostate_st, dcm_num, mr, st, img_shape)

  os.remove('mask.png')
  del contours, mr
  gc.collect()
  img_array = img_array.reshape(-1, *img_shape)
  mask_array = mask_array.reshape(-1, *img_shape)

  # もしすでにあればその末尾に追加していく
  save_npy(f'../{dt_type}.npy', img_array, img_shape)
  save_npy(f'../{dt_type}_mask.npy', mask_array, img_shape)


def inter_patient(mov_contour_list, fix_contour_list, dates):
  patients = [7, 8, 9, 10, 11, 14, 17, 20]
  for patient in patients:
    patient_path = f'/home/uchiyama/work/voxelmorph/dataset/person_{patient}'
    for date in dates:
      mov_contour_path = f'{patient_path}/moving/contour/contour.dcm'
      fix_contour_path = f'{patient_path}/{date}/contour/contour.dcm'
      mov_mr_dir_path = f'{patient_path}/moving/img/'
      fix_mr_dir_path = f'{patient_path}/{date}/img/'

      make_dataset(mov_contour_path, mov_contour_list, mov_mr_dir_path, 'moving')
      make_dataset(fix_contour_path, fix_contour_list, fix_mr_dir_path, 'fixed')
      print(f'person_{patient}, moving: moving, fixed: {date}')

def intra_patient():
  fr_image_path = '/home/uchiyama/work/voxelmorph/dataset/person_4'
  # contour_list = ['Bladder D+E', 'Rectum D+E', 'Prostate D+E', 'CTV D+E', 'SV D+E]

  for mov in dates:
    for fix in dates:
      if mov == fix: continue
      mov_contour_dcm_path = f'{fr_image_path}/{mov}/contour/contour.dcm'
      fix_contour_dcm_path = f'{fr_image_path}/{fix}/contour/contour.dcm'
      mov_mr_dir_path = f'{fr_image_path}/{mov}/img/'
      fix_mr_dir_path = f'{fr_image_path}/{fix}/img/'

      make_dataset(mov_contour_dcm_path, contour_list, mov_mr_dir_path, 'moving')
      make_dataset(fix_contour_dcm_path, contour_list, fix_mr_dir_path, 'fixed')
      print(mov, fix)

if __name__ == '__main__':
  mov_contour_list = ['4_Bladder', '1_Prostate']
  fix_contour_list = ['Bladder D+E', 'Prostate D+E']
  dates = ['fixed_1', 'fixed_2', 'fixed_3', 'fixed_4', 'fixed_5']
  # intra_patient()
  inter_patient(mov_contour_list, fix_contour_list, dates)
