import os, glob, sys
import cv2
import numpy as np
import sklearn
from sklearn import model_selection, preprocessing,linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

def PreProcessing(img):
    img = cv2.resize(img,(50,50))
    gx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)
    gy = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))
    hist = np.bincount(bins.ravel(), mag.ravel(), bin_n)
    return hist

img_paths = glob.glob('D:\\Project\\Latihan\\Wajah\\Train\\*')
bin_n=32
data = []
label = []
svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC)

for img_path in img_paths:
    word_label = img_path.rsplit('.')[-3]
    category=word_label[-1:]
    if category == 'a':
        img = cv2.imread(img_path, 0)
        #print(word_label)
        hist=PreProcessing(img)
        data.append(hist)
        label.append(0)
    elif category == 'b':
        img = cv2.imread(img_path, 0)
        #print(word_label)
        hist=PreProcessing(img)
        data.append(hist)
        label.append(1)
    elif category == 'c':
        img = cv2.imread(img_path, 0)
        #print(word_label)
        hist=PreProcessing(img)
        data.append(hist)
        label.append(2)
    elif category == 'd':
        img = cv2.imread(img_path, 0)
        #print(word_label)
        hist=PreProcessing(img)
        data.append(hist)
        label.append(3)    
    
responses=np.float32(label)
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data, label, test_size=0.3,random_state=9)
SVM = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',probability=False, random_state=None)
#SVM = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=True)
SVM.fit(train_x, train_y)# predict the labels on validation dataset
i=1
'''
img=cv2.imread('D:\\Project\\Latihan\\Wajah\\Test\\12.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", gray)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    cv2.imshow("Image", gray)
    img_resized = cv2.resize(roi_gray,(32,32))
    hist=PreProcessing(img_resized )
    predictions_SVM = SVM.predict(hist.reshape(1,-1))# Use accuracy_score function to get the accuracy
    if(predictions_SVM==0):
        nama="Edi"
    elif(predictions_SVM==1):
        nama="Qois"
    elif(predictions_SVM==2):
        nama="Fauzi"
    elif(predictions_SVM==3):
        nama="Aris"
    print(nama)
'''
'''
img_paths = glob.glob('D:\\Project\\Latihan\\Wajah\\Test\\*')
for img_path in img_paths:
    print(img_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        img_resized = cv2.resize(roi_gray,(32,32))
        hist=PreProcessing(img_resized )
        predictions_SVM = SVM.predict(hist.reshape(1,-1))# Use accuracy_score function to get the accuracy
        if(predictions_SVM==0):
            nama="Edi"
        elif(predictions_SVM==1):
            nama="Qois"
        elif(predictions_SVM==2):
            nama="Fauzi"
        elif(predictions_SVM==3):
            nama="Aris"
        print(nama)
'''

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
face_crop = []
i=1
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face_crop = frame[y:y+h, x:x+w]
            img_resized = cv2.resize(face_crop,(32,32))
            hist=PreProcessing(img_resized )
            predictions_SVM = SVM.predict(hist.reshape(1,-1))# Use accuracy_score function to get the accuracy
            print(int(predictions_SVM))
            if(predictions_SVM==0):
                nama="Edi"
            elif(predictions_SVM==1):
                nama="Qois"
            elif(predictions_SVM==2):
                nama="Fauzi"
            elif(predictions_SVM==3):
                nama="Aris"
            cv2.putText(frame, "Name : " + nama, (x + x//10, y+h+20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('frame',frame)
            i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
