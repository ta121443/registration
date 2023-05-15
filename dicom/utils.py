from matplotlib.path import Path
import pydicom as pd
import numpy as np
import os

def save_npy(npy_path, img, img_shape):
  """
  画像やマスクのndarrayを保存する。その際、すでにデータセットが存在している場合は、そのデータセットの末尾に追加する形で作成する。
  """
  if os.path.isfile(npy_path):
    exist_imgs = np.load(npy_path)
    img = np.append(exist_imgs, img)
    img = img.reshape(-1, *img_shape)
  np.save(npy_path, img)
  

def make_contours(rtss, contour_list):

  structures = {}
  for roi in rtss.StructureSetROISequence:
    structures[roi.ROINumber] = roi.ROIName

  contours = {}
  for contour in rtss.ROIContourSequence:
    structure = structures[contour.ReferencedROINumber]
    if structure in contour_list:
      contours[structure] = {}

      points = {}
      for c in contour.ContourSequence:
        if c.ContourGeometricType != 'CLOSED_PLANAR': continue
        contour_data = c.ContourData
        z = float(contour_data[2])
        x = [float(x) for x in contour_data[::3]]
        y = [float(y) for y in contour_data[1::3]]
        xy = list(zip(x, y))
        xy.append(xy[0])
        if z not in points:
          points[z] = []
        points[z].append(xy)
      contours[structure]['points'] = points

      paths = {}
      for z, p in points.items():
        if len(p) > 1:
          for i, c in enumerate(p):
            codes = np.ones(len(c)) * Path.LINETO
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            if i == 0:
              all_paths = c 
              all_codes = codes 
            else:
              all_paths = np.concatenate((all_paths, c))
              all_codes = np.concatenate((all_codes, codes))
        else:
          all_paths = p[0]
          all_codes = np.ones(len(p[0])) * Path.LINETO
          all_codes[0] = Path.MOVETO
          all_codes[-1] = Path.CLOSEPOLY
        path = Path(all_paths, all_codes)
        paths[z] = path
      contours[structure]['paths'] = paths

      color = [
        float(contour.ROIDisplayColor[0]) / 255.,
        float(contour.ROIDisplayColor[1]) / 255.,
        float(contour.ROIDisplayColor[2]) / 255.,
        0.3 
      ]
      contours[structure]['ec'] = color[0:3]
      contours[structure]['fc'] = color

  return contours