# how to use css in python_ flask
# flask render_template example
 
from flask import Flask, render_template,Response
import cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np
import os
from cvzone.ClassificationModule import Classifier
import array as arr 
 
# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')

global count 
global my_object
my_object = ""
count = 0
global list_obj
list_obj = []

a = arr.array('i', [0, 0, 0, 0, 0, 0, 0, 0]) 


mydata = Classifier('data/keras_model.h5', 'data/labels.txt')
camera=cv2.VideoCapture(0)


def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            a[0] = a[1]
            a[1] = a[2]
            a[2] = a[3]
            a[3] = a[4]
            a[4] = a[5]
            a[5] = a[6]
            a[6] = a[7]
            predict, a[7] = mydata.getPrediction(frame, color=(0,0,255))
            if a[0] == 0 and a[1] == 0 and a[2] == 0 and a[3] == 0 and a[3] != a[4] and a[4] == a[5] and a[5] == a[6] and a[6] == a[7]:
                global count 
                count = count + 1
                global my_object
                my_object = mydata.list_labels[a[7]]
                global list_obj
                list_obj.insert(0, my_object)
            
            # print(count)
            # cv2.putText(frame, str(count),
            #             pos=(60,50), scale=2, color= (0,255,0))
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

 
@app.route('/')
def home():
    global count
    global my_object
    global list_obj
    return render_template('index.html', index = count, my_object = my_object, list_obj = list_obj)
    # return f"<h1> Index {check1} </h1>"

@app.route('/video123')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')



# @app.route('/count')
# def count():
#     return Response(count())


if __name__=='__main__':
    app.run(debug = True)