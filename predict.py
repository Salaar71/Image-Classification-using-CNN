from keras.models import load_model
from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2


new_model = load_model('models/imageclassifier.h5')
data_dir='data'
root = Tk()
root.withdraw()
dir=[]
initialdir=data_dir
dir = filedialog.askopenfilenames(
initialdir=initialdir,
title="Open Image File For Prediction",
filetypes=(("PNG files", "*.png"), ("JPG Files", "*.jpg*"), )
)
i=0
j=0
for temp in dir:
    img = cv2.imread(temp)
    resize = cv2.resize(img, (123,123))
    """plt.imshow(img)
    plt.show()
    plt.imshow(resize.numpy().astype(int))
    plt.show()"""
    yhat = new_model.predict(np.expand_dims(resize/255, 0))
    if yhat > 0.5:
        print(f'Predicted class is 1Hex')
        i=i+1
    else:
        print(f'Predicted class is 0Hex')
        j=j+1
print("No. of instances predicted as 1Hex: ",i)
print("No. of instances predicted as 0Hex: ",j)