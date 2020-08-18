# Import packages
import os
import os.path
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import sys
import winreg
from datetime import datetime
import socket
from ctypes import *
from tkinter import messagebox, Toplevel, Label, Tk, Button
from PyQt5.QtCore import QSettings, QPoint, QElapsedTimer
from PyQt5 import QtCore, QtWidgets, QtGui

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setblocking(0)

try:
    sock.connect(('127.0.0.1', 63366))
except:
    print("An exception occured")

def socketSend(message):
    print("socket send:"+message)
    try:
        sock.send(message.encode())
    except:
        print("An exception occured")

data = ""
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

date = str(datetime.now())

date = date.replace(" ","")
date = date.replace("-","")
date = date.replace(":","")

# Grab path to current working directory
CWD_PATH = 'C:/Vendron/Vendron_AI_Vision/model'
CWD_TXT = 'C:/Vendron/Vendron_AI_Vision/detection_result/detection.dat'
CWD_HISTORY = 'C:/Vendron/Vendron_AI_Vision/detection_result/history/' + date + ".txt"

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def regkey_value(path, name="", start_key = None):
    if isinstance(path, str):
        path = path.split("\\")
    if start_key is None:
        start_key = getattr(winreg, path[0])
        return regkey_value(path[1:], name, start_key)
    else:
        subkey = path.pop(0)
    with winreg.OpenKey(start_key, subkey) as handle:
        assert handle
        if path:
            return regkey_value(path, name, handle)
        else:
            desc, i = None, 0
            while not desc or desc[0] != name:
                desc = winreg.EnumValue(handle, i)
                i += 1
            return desc[1]
def MessageBox(title, text, style):
    sty = int(style) + 4096
    return windll.user32.MessageBoxW(0, text, title, sty)
    
def detection():
    while(True):
        try:
            data = str(sock.recv(1024),'utf-8')
        except:
            data = ""
        if (data == 'start_object_detection;#'):
            try:
                percentage = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "confidency_threshold")
            except:
                percentage = "0.9"

            try:
                index = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "camera_index")
            except:
                index = "0"

            if os.path.isfile(CWD_TXT):
                pass
            else:
                with open(CWD_TXT, 'a+') as f:
                    f.write('[]' + '\n')

            if (percentage == ""):
                percentage = "0.9"
            percentage = float(percentage)
            print (index)
            if index == "":
                index = 0
            f = open(CWD_TXT, 'r+')
            f.truncate(0)
            try:
                video = cv2.VideoCapture(int(index))
                width  = video.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)
                height = video.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT)
                cam_fps = video.get(cv2.cv2.CAP_PROP_FPS)
            except:
                pass

            elapsed_timer = QElapsedTimer()
            elapsed_timer.start()

            if video is None or not video.isOpened():
                MessageBox("Vendron","Error No such Camera exist", 64)
                detection()
            else :
                sent = True
                while(True):
                    try:
                        data = str(sock.recv(1024),'utf-8')
                    except:
                        pass
                
                    if (data == 'end_object_detection;#'):
                        sent = False
                        cv2.destroyWindow('Object detector')
                        video.release()
                        socketSend("object_detection_ended;#")
                        break
                    else:
                        data = []
                        myList = []
                        myScore = []
                        result_list = []
                        name = []

                        try:
                            fps = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "frame_rate")
                        except:
                            fps = cam_fps
                        try:
                            ymin_1 = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "y_min_threshold_1")
                        except:
                            ymin_1 = "80"
                        try:
                            ymax_1 = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "y_max_threshold_1")
                        except:
                            ymax_1 = "240"
                        try:
                            ymin_2 = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "y_min_threshold_2")
                        except:
                            ymin_2 = "240"
                        try:
                            ymax_2 = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "y_max_threshold_2")
                        except:
                            ymax_2 = "400"
                        try:
                            places = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "is_camera_reversed")
                        except:
                            places = "false"
                        try:
                            w2 = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "x")
                        except:
                            w2 = "0"
                        try:
                            h2 = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "y")
                        except:
                            h2 = "0"
                        try:
                            w1 = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "width")
                        except:
                            w1 = "640"
                        try:
                            h1 = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "height")
                        except:
                            h1 = "480"


                        if video is None:
                            pass
                        else:
                            ret, frame = video.read()
                            if(w1 == ""):
                                w1 = "640"
                            if(w2 == ""):
                                w2 = "0"
                            if(h1 == ""):
                                h1 = "480"
                            if(h2 == ""):
                                h2 = "0"

                            w1 = int(w1) + int(w2)
                            h1 = int(h1) + int(h2)
                            frame = frame[int(h2):int(h1),int(w2):int(w1)]
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_expanded = np.expand_dims(frame, axis=0)

                            # Perform the actual detection by running the model with the image as input
                            try:
                                (boxes, scores, classes, num) = sess.run(
                                    [detection_boxes, detection_scores, detection_classes, num_detections],
                                    feed_dict={image_tensor: frame_expanded})
                            except:
                                pass

                            # Draw the results of the detection (aka 'visulaize the results')
                            try:
                                vis_util.visualize_boxes_and_labels_on_image_array(
                                    frame,
                                    np.squeeze(boxes),
                                    np.squeeze(classes).astype(np.int32),
                                    np.squeeze(scores),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=3,
                                    min_score_thresh= percentage)

                                data = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > percentage]
                                for cl in data:
                                   if cl != None :
                                        myList.append(str(cl['name']))

                                objects = []
                                for index, value in enumerate(classes[0]):
                                    object_dict = {}
                                    if scores[0, index] > percentage:
                                        object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                                                                                                        scores[0, index]
                                        objects.append(object_dict)

                                coordinates = vis_util.return_coordinates(
                                                frame,
                                                np.squeeze(boxes),
                                                np.squeeze(classes).astype(np.int32),
                                                np.squeeze(scores),
                                                category_index,
                                                use_normalized_coordinates=True,
                                                line_thickness=3,
                                                min_score_thresh= percentage)
                            except:
                                pass
                        
                            if(sent == True):
                                socketSend("object_detection_started;#")
                                sent = False

                            if(places == ""):
                                places = "false"
                            if(ymin_1 == ""):
                                ymin_1 = "80"
                            if(ymin_2 == ""):
                                ymin_2 = "240"
                            if(ymax_1 == ""):
                                ymax_1 = "240"
                            if(ymax_2 == ""):
                                ymax_2 = "400"

                            try:
                                if(places == "true"):
                                    alpha = 0.3;
                                    overlay = frame.copy()
                                    cv2.rectangle(overlay, (0, int(ymin_1)), (int(width), int(ymin_2)),(0, 0, 255), -1)
                                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha,
                                                    0, frame)
                                    overlay_blue = frame.copy()
                                    cv2.rectangle(overlay_blue, (0, int(ymax_1)), (int(width), int(ymax_2)),(255, 0, 0), -1)
                                    cv2.addWeighted(overlay_blue, alpha, frame, 1 - alpha,
                                                    0, frame)
                            
                                elif(places == "false"):
                                    alpha = 0.3;
                                    overlay = frame.copy()
                                    cv2.rectangle(overlay, (0, int(ymax_1)), (int(width), int(ymax_2)),(0, 0, 255), -1)
                                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha,
                                                0, frame)
                                    overlay_blue = frame.copy()
                                    cv2.rectangle(overlay_blue, (0, int(ymin_1)), (int(width), int(ymin_2)),(255, 0, 0), -1)
                                    cv2.addWeighted(overlay_blue, alpha, frame, 1 - alpha,
                                                    0, frame)
                            except:
                                pass

                            if(fps == ""):
                                fps = cam_fps
                            
                            fps = 1/int(fps)

                            print (type(fps))
        
                            while(elapsed_timer.hasExpired(fps)):
                                if coordinates is None:
                                    print("nothing")
                                else:
                                    if video is None:
                                        sent = False
                                        cv2.destroyWindow('Object detector')
                                        socketSend("object_detection_ended;#")
                                        break
                                    
                                    list_1stesult = myList
                                    coordinates_result = coordinates
                                    for ea_list,ea_coor,score in zip(list_1stesult,coordinates_result,objects):
                                        score = str(score)
                                        score = score.split(":")[1]
                                        score = score.replace("}","")
                                        score = score.replace("]","")
                                        score = float(score) * 100
                                        score = str(round(score))
                                        result = os.path.join(ea_list,",",str(ea_coor),",",score)
                                        result = result.replace("[[","[")
                                        result = result.replace("\\","")
                                        result = result.replace("[","")
                                        result = result.replace("]","")
                                        name.append(ea_list)
                                        result_list.append(result)

                                    print (result_list)
                                    result_list = str(result_list).replace("', '","];[")
                                    result_list = result_list.replace("'","")
                                    result_list = result_list.replace("'","")
                                    result_list = result_list.replace(", ",",")
                                    if result_list:
                                        with open(CWD_TXT, "a") as text_file:
                                            text_file.write(str(result_list) + "\n")

                                    if result_list:
                                        with open(CWD_HISTORY,"a") as text_file:
                                            text_file.write(str(result_list) + "\n")
                            
                                    elapsed_timer.start()
                    

                    # All the results have been drawn on the frame, so it's time to display it.
                            try:
                                path_debug = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\ai_vision_fridge", "debug")
                            except:
                                path_debug = "false"

                            if (path_debug == "true"):
                                try:
                                    cv2.imshow('Object detector', frame)
                                except:
                                    sent = False
                                    cv2.destroyWindow('Object detector')
                                    video.release()
                                    socketSend("object_detection_ended;#")
                                    break
                            else:
                                pass

                            if cv2.waitKey(1) == ord ("q"):
                                pass
 
if not os.path.isfile(PATH_TO_LABELS) or not os.path.isfile(PATH_TO_CKPT) :
    print ("nothing")
    pass
else:
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    tf.image.non_max_suppression
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    detection()
