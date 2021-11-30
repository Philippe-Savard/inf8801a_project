# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 01:17:35 2021

@author: gp570
"""

import matplotlib.pyplot as plt
import numpy as np
l1 = [0.2576,
 0.3272,
 0.3787,
 0.4308,
 0.4694,
 0.4836,
 0.4923,
 0.5209,
 0.5202,
 0.5279,
 0.5306,
 0.543,
 0.5475,
 0.5439,
 0.5606]

l2 = [0.3277,
 0.4213,
 0.4177,
 0.4342,
 0.4578,
 0.4622,
 0.4815,
 0.4845,
 0.5008,
 0.495,
 0.5125,
 0.5174,
 0.5095,
 0.5333,
 0.5337]

l3 = [0.3576 , 0.4245 , 0.4472 , 0.4482 , 0.4400 , 0.4798 , 0.4804 , 0.4928 , 0.5010 , 0.5022 , 0.5040 , 0.5146 , 0.5223 , 0.5273 , 0.5382] 
x = np.linspace(1,15,15)
plt.plot(x,l1,label="cnn")
plt.plot(x,l2,label="cnn + landmarks")
plt.plot(x,l3,label="cnn + landmarks + HOG")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy on validation set")
