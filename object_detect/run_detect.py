import re
import os
import cv2
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify
from picamera2 import Picamera2, Preview
import tflite_runtime.interpreter as tflite
import array as arr

# the TFLite converted to be used with edgetpu
modelPath = 'model.tflite'

# The path to labels.txt that was downloaded with your model
labelPath = 'labels.txt'
 

# This function takes in a TFLite Interptere and Image, and returns classifications
def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    #//size = (1, 480, 640, 4)
    common.set_input(interpreter, cv2.resize(image, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))
    #common.set_input(interpreter, cv2.resize(image, size))
    interpreter.invoke()
    return classify.get_classes(interpreter)

global count 
count = 0
#a = arr.array('i', [0, 0, 0, 0, 0, 0, 0, 0])
a = arr.array('i', [0, 0, 0, 0, 0, 0])
        
def main():
    # Load your model onto the TF Lite Interpreter
    #//interpreter = make_interpreter(modelPath)
    interpreter = tflite.Interpreter(modelPath)
    #//interpreter = tflite.Interpreter(modelPath, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    print(interpreter.get_input_details())
    labels = read_label_file(labelPath)

    cv2.startWindowThread()
    cap = Picamera2()
    cap.configure(cap.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
    cap.start()
    
    number = 0
    
    while True:
        frame = cap.capture_array()
        #//if not ret:
        #//    break
        
        # Flip image so it matches the training input
        frame = cv2.flip(frame, 1)
                
        # Classify and display image
        results = classifyImage(interpreter, frame)
        #Count object
        a[0] = a[1]
        a[1] = a[2]
        a[2] = a[3]
        a[3] = a[4]
        #a[4] = a[5]
        #a[5] = a[6]
        #a[6] = a[7]
        a[4] = results[0].id
        if a[0] == a[1] and a[1] == a[2] and a[2] == a[3] and a[3] == a[4]:# and a[3] != a[4] and a[4] == a[5] and a[5] == a[6] and a[6] == a[7]:
            if(a[5] != a[4]):
                a[5] = a[4]
                if(a[4] != 3):
                    global count
                    count = count + 1
        print(count)
        # Text
        text = f'{labels[results[0].id]} - {round(results[0].score*100,2)}%'
        org = (30, 50)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (0, 255, 25)
        lineType = cv2.LINE_4
        
        img_text = cv2.putText(frame, text, org, fontFace, fontScale, color, lineType)
        #Text 2
        text2 = f'Number of objects: {count}'
        org2 = (30, 100)
        fontFace2 = cv2.FONT_HERSHEY_SIMPLEX
        
        img_text = cv2.putText(img_text, text2, org2, fontFace2, fontScale, color, lineType)
        #show screen
        cv2.imshow('frame', img_text)
        print(f'Label: {labels[results[0].id]}, Score: {results[0].score}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
