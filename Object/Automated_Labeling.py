######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
import os
import os.path
import cv2
import glob
import numpy as np
import tensorflow as tf
import sys
import PIL
import winreg
import xml.etree.cElementTree as ET

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from os import path
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = 'model'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

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
try:
    path_images = regkey_value(r"HKEY_CURRENT_USER\Software\Image_path", "path")
except:
    pass

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def Convert(string): 
    li = list(string.split(",")) 
    return li 

if not os.path.isfile(PATH_TO_LABELS) or not os.path.isfile(PATH_TO_CKPT):
    pass
else:
    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

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

    #path_images = 'C:/Users/Henry/Desktop/object_detection/detect_images_row1'
    img_dir = os.path.join(path_images,"train")
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)

    img_dir2 = os.path.join(path_images,"test")
    data_path2 = os.path.join(img_dir2,'*g')
    files2 = glob.glob(data_path2)

    for f1 in files:
        myList = []
        data = []
        all_coordinate = []
        image = cv2.imread(f1)
        filename = os.path.basename(f1)
        true_name = filename.replace(".png",".xml")
        name_path = os.path.join(img_dir,true_name)
        print ( "train")
        if path.exists(name_path):
            print("exits")
        else:
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
            h, w, _ = image.shape
            print('width: ', w)
            print('height:', h)
            depth = "3"

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "images"
            ET.SubElement(root, "filename").text = filename
            ET.SubElement(root, "path").text = f1
            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = "Unknown"
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(w)
            ET.SubElement(size, "height").text = str(h)
            ET.SubElement(size, "depth").text = depth
            ET.SubElement(root, "segmented").text = "0"

            vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
    
            data = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.6]

            for cl in data:
                if cl != None :
                    myList.append(str(cl['name']))
            #print(cl['name'])

            coordinates = vis_util.return_coordinates(
                                image,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8,
                                min_score_thresh=0.50)

            all_coordinate.append(str(coordinates))
            print ((all_coordinate))
            print (type(all_coordinate))

            all1 = []
            for a_coor in all_coordinate:
                all1 = a_coor.split("]")
    
            for each_coor,each_list in zip(all1,myList):
                each_coor = each_coor.replace("''","")
                each_coor = each_coor.replace(", [","")
                each_coor = each_coor.replace("[","")
                #print (each_coor)
                object = ET.SubElement(root, "object")
                ET.SubElement(object, "name").text = each_list
                #print (type(all3))
                ymin,ymax,xmin,xmax = Convert(each_coor)
                #ymin = ymin1.replace(", ","")
                ET.SubElement(object, "pose").text = "Unspecified"
                ET.SubElement(object, "truncated").text = "0"
                ET.SubElement(object, "difficult").text = "0"
                bindbox = ET.SubElement(object, "bndbox")
                ET.SubElement(bindbox, "xmin").text = str(xmin)
                ET.SubElement(bindbox, "ymin").text = str(ymin)
                ET.SubElement(bindbox, "xmax").text = str(xmax)
                ET.SubElement(bindbox, "ymax").text = str(ymax)

            tree = ET.ElementTree(root)
            indent(root)
            tree.write(name_path, encoding="utf-8", xml_declaration=True)

    for f1 in files2:
        myList = []
        data = []
        all_coordinate = []
        image = cv2.imread(f1)
        filename = os.path.basename(f1)
        true_name = filename.replace(".png",".xml")
        name_path = os.path.join(img_dir2,true_name)
        print ( "test")
        if path.exists(name_path):
            print("exits")
        else:
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
            h, w, _ = image.shape
            print('width: ', w)
            print('height:', h)
            depth = "3"

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "images"
            ET.SubElement(root, "filename").text = filename
            ET.SubElement(root, "path").text = f1
            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = "Unknown"
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(w)
            ET.SubElement(size, "height").text = str(h)
            ET.SubElement(size, "depth").text = depth
            ET.SubElement(root, "segmented").text = "0"

            vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
    
            data = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.6]

            for cl in data:
                if cl != None :
                    myList.append(str(cl['name']))
            #print(cl['name'])

            coordinates = vis_util.return_coordinates(
                                image,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8,
                                min_score_thresh=0.50)

            all_coordinate.append(str(coordinates))
            print ((all_coordinate))
            print (type(all_coordinate))

            all1 = []
            for a_coor in all_coordinate:
                all1 = a_coor.split("]")
    
            for each_coor,each_list in zip(all1,myList):
                each_coor = each_coor.replace("''","")
                each_coor = each_coor.replace(", [","")
                each_coor = each_coor.replace("[","")
                #print (each_coor)
                object = ET.SubElement(root, "object")
                ET.SubElement(object, "name").text = each_list
                #print (type(all3))
                ymin,ymax,xmin,xmax = Convert(each_coor)
                #ymin = ymin1.replace(", ","")
                ET.SubElement(object, "pose").text = "Unspecified"
                ET.SubElement(object, "truncated").text = "0"
                ET.SubElement(object, "difficult").text = "0"
                bindbox = ET.SubElement(object, "bndbox")
                ET.SubElement(bindbox, "xmin").text = str(xmin)
                ET.SubElement(bindbox, "ymin").text = str(ymin)
                ET.SubElement(bindbox, "xmax").text = str(xmax)
                ET.SubElement(bindbox, "ymax").text = str(ymax)

            tree = ET.ElementTree(root)
            indent(root)
            tree.write(name_path, encoding="utf-8", xml_declaration=True)


# All the results have been drawn on image. Now display the image.

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
