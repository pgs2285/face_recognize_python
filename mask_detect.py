import cv2
import cvlib as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
import time
from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from ui import Ui_Form
import sys

class maskDetect(QMainWindow,Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        

       
            
    # open webcam
    def clickButton(self):
        program_status = 0
        model = load_model('model.h5')
        model.summary()
        webcam = cv2.VideoCapture(0)
            
            
        if not webcam.isOpened():
            print("Could not open webcam")
            QtWidgets.QMessageBox.information(self, "ERROR", "연결된 웹캠이 없습니다")

                
            
            # loop through frames
        while webcam.isOpened():
            
                # read frame from webcam 
            status, frame = webcam.read()
                
            if not status:
                print("Could not read frame")
                QtWidgets.QMessageBox.information(self, "ERROR", "프레임을 읽어들일수 없습니다")
            
                # apply face detection
            face, confidence = cv.detect_face(frame)
                
                
                # loop through detected faces
            for idx, f in enumerate(face):
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                    
                if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
                        
                    face_region = frame[startY:endY, startX:endX]
                        
                    face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
                        
                    x = img_to_array(face_region1)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                        
                    prediction = model.predict(x)
                    if prediction < 0.5: # 마스크 미착용으로 판별되면, 
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                        Y = startY - 10 if startY - 10 > 10 else startY + 10
                        text = "No Mask ({:.2f}%)".format((1 - prediction[0][0])*100)
                        cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        program_status = 0
                            
                    else: # 마스크 착용으로 판별되면
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                        Y = startY - 10 if startY - 10 > 10 else startY + 10
                        text = "Mask ({:.2f}%), time ({}sec)".format(prediction[0][0]*100, int(program_status/10))
                        cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        time.sleep(0.01)
                        program_status += 1
                        print("mask - time: ",program_status)
                            
                            

                #4초뒤에 종료
            cv2.imshow("mask detector", frame)
            if program_status > 40:
                break
                # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # release resources
        webcam.release()
        cv2.destroyAllWindows() 
        self.listView.deleteLater()
        self.listView = None
        self.listView_2.deleteLater()
        self.listView_2 = None
        self.pushButton.deleteLater()
        self.pushButton = None
        self.label.deleteLater()
        self.label= None
        self.label_2.deleteLater()
        self.label_2 = None
        QtWidgets.QMessageBox.information(self, "Good!", "마스크 인식 완료! 결제를 진행합니다")
        exit()
    def clickList(self):
        pass 
    def cancelList(self):
        pass   

app = QApplication([])
a = maskDetect()
QApplication.processEvents()
sys.exit(app.exec_())