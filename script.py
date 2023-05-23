import numpy as np

def make_dsc(dsc):
  for i in range(10):
    dsc[i] = i + 10

if __name__ == '__main__':
  dsc = {}
  make_dsc(dsc)
  print(dsc)
