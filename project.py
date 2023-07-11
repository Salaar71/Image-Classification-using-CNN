import numpy as np
import os
from PIL import Image as im
from tkinter import *
from tkinter import filedialog
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.metrics import Precision, Recall, BinaryAccuracy
from matplotlib import pyplot as plt
import cv2
import shutil
import sys

#--------------------------------------------------Part I-Image Creation--------------------------------------------------

data_dir = 'data'
def create_images(width,height):
    if os.listdir(data_dir)==['0Hex_Images','1Hex_Images']:
        print("Both folders are already created!")
        cnn()
    else:
        sys.set_int_max_str_digits(0)
        
#Hide the root window that comes by default
        root = Tk()
        root.withdraw()
#Browse and select txt files
        dir = []
        initialdir="/Binaries/"
        dir = filedialog.askopenfilenames(
        initialdir=initialdir,
        title="Open Image File For Prediction",
        filetypes=(("Text Files", "*.txt*"),)
        )

#Create new folder for images
        str1=os.path.dirname(dir[-1])
        name=os.path.split(str1)[-1]
        name=name+"_Images"
        new_path= os.path.join(data_dir,name) 
        try:
            os.mkdir(new_path)
        except:
            print("This folder is already being created. Create second one!")
            create_images(width,height)
              
#Reading data in txt files and decoding hexadecimal characters
        for temp in dir:
            txtf = open(temp)  # Open file
            data = txtf.read()  # Read data in file
            data = data.replace('\'','')   # Remove label
            data = data.replace(' ', '')  # Remove whitespaces
            data = data.replace('\n', '') # Remove breaks in lines
            data = [int(data[i:i+2],16) for i in range(0,len(data),2)] #Coonverting hexadecimal data into bytes
            #print(data)
            txtf.close()
            
#Finding optimal factor pair for size of image    
            x = len(data)
            #print(x)
            val1=0
            val2=0
            for i in range(1, int(pow(x, 1 / 2))+1):
                if x % i == 0:
                    val1=i
                    val2=int(x / i)

#Converting 1-D to 3-D numpy array
            data = np.array(data).reshape(val1,val2,-1)
            data = np.broadcast_to(data,(val1,val2,3))           
            #print(data) #Display 3-D array
            #print(val1)
            #print(val2)
    
#Writing array to image
            blue = data[:, :, 0]
            #print(blue)
            green = data[:, :, 1]
            #print(green)
            red = data[:, :, 2]
            #print(red)
            rgb = (np.dstack((red,green,blue))*256).astype(int)
            img = im.fromarray(rgb,'RGB')
    
#Split path into filename and extenion
            pathname, extension = os.path.splitext(f"{temp}")
            filename = pathname.split('/')  # Get filename without txt extension

#Defining name of image file same as txt file    
            filepath = f"{new_path}\{filename[-1]}.png"
            
#Resize image
            img=img.resize((width,height))

#Saving image into path   
            img.save(filepath)
            
        if os.listdir(data_dir)!=['0Hex_Images','1Hex_Images']:
            create_images(width,height)
        else:
            print("Both folders have been created!")
            cnn()


#----------------------------------------Part II-Applying CNN to Newly Created Images----------------------------------------

def cnn():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.list_physical_devices('GPU')

#Remove dodgy images
    image_exts = ['jpeg','jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir): 
        print(image_class)
    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try: 
                img = cv2.imread(image_path)
                extension = os.path.splitext(image_path)[1][1:]
                if extension not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
                # os.remove(image_path)
    s=[]
    a=[]
    b=[]
    c=[]
    d=[]
    start=64
    end=256
    i=0
    j=0
    f=open("Results.txt","a")
#Load Data
    for p in range (start,end+1):  
        s.append(p)
        f.write("\n")
        f.write("\n")
        f.write(f"{p}:")
        f.write("\n")
        f.write("\n")
        data = tf.keras.utils.image_dataset_from_directory(data_dir,image_size=(p,p))
        #print(data) 
        data_iterator = data.as_numpy_iterator()
        batch = data_iterator.next()
        """fig, ax = plt.subplots(ncols=4, figsize=(20,20))
        for idx, img in enumerate(batch[0][:4]):
            ax[idx].imshow(img.astype(int))
            ax[idx].title.set_text(batch[1][idx])"""
        
#Scale Data
        data = data.map(lambda x,y: (x/255, y))
        data.as_numpy_iterator().next()

#Split Data
        train_size = int(len(data)*.7)
        val_size = int(len(data)*.2)+1
        test_size = int(len(data)*.1)+1
        train = data.take(train_size)
        val = data.skip(train_size).take(val_size)
        test = data.skip(train_size+val_size).take(test_size)

#Build Deep Learning Model
        model = Sequential()
        model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(p,p,3)))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3, 3), 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(16, (3, 3), 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile('adam', loss=tf.losses.BinaryCrossentropy() ,metrics=['accuracy'])
        model.summary()

#Train Model 
        logdir='logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        model.fit(train, epochs=50, validation_data=val, callbacks=[tensorboard_callback])

#Plotting Performance of Model
        """fig = plt.figure()
        timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(close_event)
        plt.plot(hist.history['loss'], color='teal', label='loss')
        plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()
        fig = plt.figure()
        timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(close_event)
        plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        timer.start()
        plt.show()"""

#Evaluating Model
        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()
        for batch in test.as_numpy_iterator(): 
            X, y = batch
            yhat = model.predict(X)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)
        print(pre.result(), re.result(), acc.result())
        f.write("Precision: ")
        f.write(f'{pre.result()}\n')
        f.write("Recall: ")
        f.write(f'{re.result()}\n')
        f.write("Accuracy: ")
        f.write(f'{acc.result()}\n')
        f.write("\n")
        dir1=[]
        dir2=[]
        folder1=f'{data_dir}\\0Hex_Images'
        folder2=f'{data_dir}\\1Hex_Images'
        for files in os.listdir(folder1):
            dir1.append(os.path.join(folder1,files))
        for files in os.listdir(folder2):
            dir2.append(os.path.join(folder2,files))
        for temp in dir1:
            img = cv2.imread(temp)
            resize = cv2.resize(img, (p,p))
            """plt.imshow(img)
            plt.show()
            plt.imshow(resize.numpy().astype(int))
            plt.show()"""
            yhat = model.predict(np.expand_dims(resize/255, 0))
            if yhat > 0.5:
                print(f'Predicted class is 1Hex')
                i=i+1
            else:
                print(f'Predicted class is 0Hex')
                j=j+1
        a.append(j)
        c.append(i)
        if a[p-start]==max(a):
#Saving Model            
            model.save(os.path.join('models','imageclassifier0.h5'))
        f.write("Results For 0Hex Folder\n")
        print("No. of instances predicted as 1Hex: ",i)
        f.write("No. of instances predicted as 1Hex: ")
        f.write(f'{i}')
        f.write("\n")
        print("No. of instances predicted as 0Hex: ",j)
        f.write("No. of instances predicted as 0Hex: ")
        f.write(f'{j}')
        f.write("\n")
        f.write("\n")
        i=0
        j=0
        for temp in dir2:    
            img = cv2.imread(temp)
            resize = cv2.resize(img, (p,p))
            """plt.imshow(img)
            plt.show()
            plt.imshow(resize.numpy().astype(int))
            plt.show()"""
            yhat = model.predict(np.expand_dims(resize/255, 0))
            if yhat > 0.5:
                print(f'Predicted class is 1Hex')
                i=i+1
            else:
                print(f'Predicted class is 0Hex')
                j=j+1
        b.append(i)
        d.append(j)
        if b[p-start]==max(b):
#Saving Model
            model.save(os.path.join('models','imageclassifier1.h5'))
        f.write("Results For 1Hex Folder\n")    
        print("No. of instances predicted as 1Hex: ",i)
        f.write("No. of instances predicted as 1Hex: ")
        f.write(f'{i}')
        f.write("\n")
        print("No. of instances predicted as 0Hex: ",j)
        f.write("No. of instances predicted as 0Hex: ")
        f.write(f'{j}')
        f.write("\n")
        f.write("\n")
        f.write("\n")
        i=0
        j=0                   
    print("For 0Hex model is training best on the following dimensions:\nHeight: ",s[a.index(max(a))],"\nWidth: ",s[a.index(max(a))])
    f.write(f'For 0Hex model is training best on the following dimensions:\nHeight: {s[a.index(max(a))]}\n')
    f.write(f'Width: {s[a.index(max(a))]}')
    f.write("\n")
    f.write("\n")
    print("For 1Hex model is training best on the following dimensions:\nHeight: ",s[b.index(max(b))],"\nWidth: ",s[b.index(max(b))])
    f.write(f'For 1Hex model is training best on the following dimensions:\nHeight: {s[b.index(max(b))]}\n')
    f.write(f'Width: {s[b.index(max(b))]}')
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.close()
    
    if min(c)!=0 and min(d)!=0:
        while True:
            ask=input("Are you satisfied with the result or not? Press 'Y' if you are satisfied or 'N' if not satisfied.\nChoice: ")
            if ask=='y' or ask=='Y':
                 print("It seems that you are satisfied with results. Ending the program!")
                 exit()
            elif ask=='n' or ask=='N':
                print("It seems that you are not satisfied with results. Enter the value for dimensions of your choice")
                value=input("Value: ")
                value=int(value)
                try:
                    for folders in os.listdir(data_dir):
                        shutil.rmtree(f'{data_dir}\\{folders}')
                    create_images(value,value)
                except ValueError:
                    print("Data type of one or both of the dimensions is not integer. Please try again!")
            else:
                print("You have entered invalid choice. Please try again!")
    else:
        print("Model is overfitted. Can't have 100 percent classification.")
        exit()
   
if __name__ == "__main__":
    create_images(300,300)
    
    