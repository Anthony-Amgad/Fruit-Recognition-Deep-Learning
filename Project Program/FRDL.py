from PyQt5 import QtWidgets, uic, QtGui
import os
import tkinter
from tkinter import filedialog
import sys
import webbrowser
from keras.models import model_from_json
import tensorflow as tf
from PIL import Image
import numpy as np
#from charset_normalizer import md__mypyc


root = tkinter.Tk()
root.withdraw()
class_indicies = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

class FRDL(QtWidgets.QMainWindow):

    imagePath = ""
    modelv2_0 = None
    modelv2_1 = None
    modelv2_2 = None
    modelv2_3 = None

    def browsePic(self):
        currdir = os.getcwd()
        self.imagePath = filedialog.askopenfilename(parent=root, initialdir=currdir, title="Please select a file", filetypes=[("Images",["*.jpg","*.jpeg"])])
        if len(self.imagePath)>0:
            self.imgView.setPixmap(QtGui.QPixmap(self.imagePath))
            #print(imagePath)

    def openDrive(self):
        webbrowser.open('https://drive.google.com/drive/folders/1MHaG9bKg11tv4XjwgK7bgxnqhBcEbjt5?usp=sharing', new=2)

    def predictImg(self):
        #print(self.imagePath)
        if self.imagePath != "":
            img = Image.open(self.imagePath)
            img = img.resize((224,224))
            img = np.expand_dims(img, axis=0)
            prediction = [0,0,0,0,0]
            n = 0
            if self.modelv20cb.isChecked() or self.modelv21cb.isChecked() or self.modelv22cb.isChecked() or self.modelv23cb.isChecked():
                if self.modelv20cb.isChecked():
                    tempred = self.modelv2_0.predict(img)[0]
                    for i, x in enumerate(tempred):
                        prediction[i] += x
                    n+=1
                if self.modelv21cb.isChecked():
                    tempred = self.modelv2_1.predict(img)[0]
                    for i, x in enumerate(tempred):
                        prediction[i] += x
                    n+=1
                if self.modelv22cb.isChecked():
                    tempred = self.modelv2_2.predict(img)[0]
                    for i, x in enumerate(tempred):
                        prediction[i] += x
                    n+=1
                if self.modelv23cb.isChecked():
                    tempred = self.modelv2_3.predict(img)[0]
                    for i, x in enumerate(tempred):
                        prediction[i] += x
                    n+=1
                for i in range(len(prediction)):
                    prediction[i] /= n
            
                self.appleProgressBar.setValue((prediction[0]*100).astype(int))
                self.bananaProgressBar.setValue((prediction[1]*100).astype(int))
                self.grapeProgressBar.setValue((prediction[2]*100).astype(int))
                self.mangoProgressBar.setValue((prediction[3]*100).astype(int))
                self.strawberryProgressBar.setValue((prediction[4]*100).astype(int))

                ind = prediction.index(max(prediction))
            
                self.predictedText.setText(class_indicies[ind])
            
            else:
                self.predictedText.setText("Please pick at least 1 model")
        else:
            self.predictedText.setText("Please import an image")
    
    def __init__(self):

        super(FRDL,self).__init__()
        uic.loadUi('res/ui/FRDL.ui',self)

        try:
            json_file = open('res\models\modelv2_0.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.modelv2_0 = model_from_json(loaded_model_json)
            self.modelv2_0.load_weights("res\models\modelv2_0.h5")
            self.modelv2_0.compile(
                optimizer = tf.optimizers.SGD(learning_rate=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except:
            self.modelv20cb.setEnabled(False)

        try:
            json_file = open('res\models\modelv2_1.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.modelv2_1 = model_from_json(loaded_model_json)
            self.modelv2_1.load_weights("res\models\modelv2_1.h5")
            self.modelv2_1.compile(
                optimizer = tf.optimizers.SGD(learning_rate=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except:
            self.modelv21cb.setEnabled(False)

        try:
            json_file = open('res\models\modelv2_2.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.modelv2_2 = model_from_json(loaded_model_json)
            self.modelv2_2.load_weights("res\models\modelv2_2.h5")
            self.modelv2_2.compile(
                optimizer = tf.optimizers.SGD(learning_rate=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except:
            self.modelv22cb.setEnabled(False)

        try:
            json_file = open('res\models\modelv2_3.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.modelv2_3 = model_from_json(loaded_model_json)
            self.modelv2_3.load_weights("res\models\modelv2_3.h5")
            self.modelv2_3.compile(
                optimizer = tf.optimizers.SGD(learning_rate=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except:
            self.modelv23cb.setEnabled(False)


        self.setWindowIcon(QtGui.QIcon('res/img.png'))
        self.setWindowTitle("Fruit Recognition")

        self.setFixedSize(1112, 858)

        self.browsePictureBtn.clicked.connect(self.browsePic)
        self.predictBtn.clicked.connect(self.predictImg)
        self.openDriveBtn.clicked.connect(self.openDrive)

        self.show()

        ##This is for finding functions using auto complete as they cannot be found with the item loaded in the .ui
        #self.x = QtWidgets.QCheckBox(self.centralwidget)
        #self.x.setEnabled()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FRDL()
    app.exec_()