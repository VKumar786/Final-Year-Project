# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 01:46:24 2023

@author: bakru_k78
"""

import numpy as np
import pickle

loaded_model = pickle.load(open("/home/bakru_k78/VsCodeProject/Final-Year-Project/trained_model.sav", "rb"))


input_data = (0,137,40,35,168,43.1,2.288,33)

# change input to numpy array
id_np_arr = np.asarray(input_data)

# reshape array as we are predicting for one instance
id_reshaped = id_np_arr.reshape(1, -1)

# Standardized as we have done to model
# id_std = scaler.transform(id_reshaped)  

predict = loaded_model.predict(id_reshaped)

if predict[0] == 0:
  print("Person is not diabetic")
else:
  print("Person is diabetic")