import os
import numpy as np
import pydicom

class MRImage:
  """
  MRを扱うためのクラス
  mr = MRImage()
  mr.load(path)
  でpathに存在するMRシリーズを読み込み
  """
  def __init__(self):
    """コンストラクタ"""
    self.is_loaded = False

  def load(self, path):
    """pathを指定して、MRの情報を取得"""
    if self._has_one_series(path):
      _dcm_files = self._select_mrs(path)
      _ref_mr = None
      self.number_of_slices = len(_dcm_files)
      i = 0
      for dcm_file, position in sorted(_dcm_files.items(),
                                        key=lambda x: -x[1]):
        if _ref_mr is None:
          _ref_mr = pydicom.dcmread(os.path.join(path, dcm_file))
          self.rows = int(_ref_mr.Rows)
          self.columns = int(_ref_mr.Columns)
          self.volume = np.zeros(
              (self.number_of_slices, self.rows, self.columns))
          self.position = _ref_mr.ImagePositionPatient
          self.pixel_spacing = _ref_mr.PixelSpacing
          self.for_uid = _ref_mr.FrameOfReferenceUID
          self.thickness = float(_ref_mr.SliceThickness)
        dcm_data = pydicom.dcmread(os.path.join(path, dcm_file))
        hu_value = dcm_data.pixel_array * dcm_data.RescaleSlope + \
          dcm_data.RescaleIntercept
        self.volume[i, :, :] = hu_value
        i += 1
      self.x_min = float(self.position[0])
      self.x_max = self.x_min + \
        (self.columns - 1) * float(self.pixel_spacing[1])
      self.x_array = np.linspace(self.x_min,
                                  self.x_max,
                                  num=self.columns)
      self.y_min = float(self.position[1])
      self.y_max = self.y_min + \
        (self.rows - 1) * float(self.pixel_spacing[0])
      self.y_array = np.linspace(self.y_min, self.y_max, num=self.rows)
      self.z_min = min(_dcm_files.values())
      self.z_max = max(_dcm_files.values())
      self.z_array = np.linspace(self.z_max,
                                self.z_min,
                                num=self.number_of_slices)
      self.is_loaded = True
    else:
      return "There are no MR seriese or more than 2 MR seriese."

  def _select_mrs(self, path):
    """指定されたディレクトリから、DICOM CTとそのスライス位置を抽出"""
    if self._has_one_series(path):
      _dcm_files = {}
      for file_path in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_path)):
          continue
        if pydicom.misc.is_dicom(os.path.join(path, file_path)):
          dcm_data = pydicom.dcmread(os.path.join(path, file_path))
          if dcm_data.Modality == 'MR':
            slice_position = float(dcm_data.ImagePositionPatient[2])
            _dcm_files[file_path] = slice_position
      return _dcm_files
    else:
      return "There are no MR seriese or more than 2 MR seriese."

  def _has_one_series(self, path):
    """指定されたディレクトリにMRシリーズが1種類だけしかないか判定"""
    uids = []
    for file_path in os.listdir(path):
      if os.path.isdir(os.path.join(path, file_path)):
        continue
      if pydicom.misc.is_dicom(os.path.join(path, file_path)):
        dcm_file = pydicom.dcmread(os.path.join(path, file_path))
        if dcm_file.Modality == 'MR':
          uids.append(dcm_file.SeriesInstanceUID)
    if len(set(uids)) == 1:
      return True
    else:
      return False