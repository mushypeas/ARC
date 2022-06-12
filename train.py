import pandas as pd
import numpy as np
import csv

def gaussian_filter(rad):
  sigma = rad/3
  shape = (rad*2+1,rad*2+1)
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  # normalize the filter
  return h / h[rad][rad]

class Cell():
  label_dist = None
  activation_cnt = 0
  strength_sum = 0
  threshold = 0.6

  def __init__(self):
    self.label_dist = {}
    
  def activate(self, label, strength):
    if label in self.label_dist:
      self.label_dist[label] += strength
    else:
      self.label_dist[label] = strength
    self.strength_sum += strength
    self.activation_cnt += 1
    
  def recognize(self):
    pred = None
    if len(self.label_dist) > 0:
      sorted_dist = sorted(self.label_dist.items(), key=lambda item: item[1])
      # without uncertainty
      pred = sorted_dist[-1][0]
      # # with uncertainty
      # if sorted_dist[-1][1] / self.strength_sum > self.threshold:
      #   pred = sorted_dist[-1][0]
    return pred

class MemorySpace():
  cells = []
  kernel = None
  rad = 0

  def __init__(self, size, rad):
    # initialize memory space
    for i in range(size):
      self.cells.append([Cell() for j in range(size)])

    self.rad = rad
    self.kernel = gaussian_filter(rad = rad)
        
    
  def activate(self, label, x, y):
    y_start, x_start = y-self.rad, x-self.rad
    for i in range(self.rad*2+1):
      if i < 0:
        continue
      for j in range(self.rad*2+1):
        if j < 0:
          continue
        self.cells[y_start+i][x_start+j].activate(label=label, strength=self.kernel[i][j])

  def recognize(self, x, y):
    pred = self.cells[y][x].recognize()
    return pred


mem = MemorySpace(size=256, rad=3)

train_dataset = pd.read_csv('data/norm_train.csv')
test_dataset = pd.read_csv('data/norm_test.csv')

for idx, data in train_dataset.iterrows():
  label, x, y = int(data['label']), int(data['x']), int(data['y'])
  mem.activate(label, x, y)
  print(f"Training... [{idx}/{len(train_dataset)}]", end='\r')

correct = 0
for idx, data in test_dataset.iterrows():
  label, x, y = int(data['label']), int(data['x']), int(data['y'])
  pred = mem.recognize(x, y)
  if pred == label:
    correct += 1

print(f"\naccuracy: {correct / len(test_dataset)}")