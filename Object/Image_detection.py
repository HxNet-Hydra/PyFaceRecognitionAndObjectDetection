# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
import re
import sqlite3
import collections
import datetime

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from collections import Counter

#This is a video capture in frame every 30 frame


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH_MODEL = 'model/'
CWD_PATH = 'object_detection/'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH_MODEL,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH_MODEL,'training','labelmap.pbtxt')

PATH_TO_DATABASE = "product.db"

# Number of classes the object detector can identify
NUM_CLASSES = 90

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
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
rData = []
total = []
def readSingleRow():
    try:
        sqliteConnection = sqlite3.connect(PATH_TO_DATABASE)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_select_query = """SELECT ID FROM CURRENT_PRODUCT ORDER BY id DESC LIMIT 1"""
        cursor.execute(sqlite_select_query)
        record = cursor.fetchone()
        print (record)
        total.append(record)
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to read single row from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
            
def createRecordsIfNotExist():
    try:
        sqliteConnection = sqlite3.connect(PATH_TO_DATABASE)
        sqlite_create_table_query = '''CREATE TABLE IF NOT EXISTS CURRENT_PRODUCT (
	    "ID"	INTEGER,
	    "Unique_ID"	vachar(127),
	    "CREATED"	DATETIME DEFAULT CURRENT_TIME,
	    "UPDATED"	DATETIME DEFAULT CURRENT_TIME,
	    "QUANTITY"	INTEGER,
	    "LEVEL"	INTEGER,
	    "REMARK"	TEXT,
	    PRIMARY KEY("ID"));'''

        cursor = sqliteConnection.cursor()
        print("Successfully Connected to SQLite")
        cursor.execute(sqlite_create_table_query)
        sqliteConnection.commit()
        print("SQLite table created")

        cursor.close()

    except sqlite3.Error as error:
        print("Error while creating a sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("sqlite connection is closed")
            
#Database insert and update on row 1
def insertMultipleRecords(recordList):
    try:
        sqliteConnection = sqlite3.connect(PATH_TO_DATABASE)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_insert_query = """INSERT INTO CURRENT_PRODUCT
                          (Unique_ID,CREATED,UPDATED,QUANTITY,LEVEL) 
                          VALUES (?, ?, ?, ?, ?);"""

        cursor.executemany(sqlite_insert_query, recordList)
        sqliteConnection.commit()
        print("Total", cursor.rowcount, "Records inserted successfully into CURRENT_PRODUCT table")
        sqliteConnection.commit()
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert multiple records into sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")

def updateMultipleColumns(Unique_ID, UPDATED, QUANTITY, LEVEL):
    try:
        sqliteConnection = sqlite3.connect(PATH_TO_DATABASE)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_update_query = """UPDATE CURRENT_PRODUCT SET UPDATED = ?, QUANTITY = ? WHERE Unique_ID = ? AND LEVEL = ?;"""
        columnValues = ( UPDATED, QUANTITY, Unique_ID, LEVEL)
        cursor.execute(sqlite_update_query, columnValues)
        sqliteConnection.commit()
        print("Multiple columns updated successfully")
        sqliteConnection.commit()
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to update multiple columns of sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("sqlite connection is closed")

def updateMultipleColumnsNotExist(ID, QUANTITY, UPDATED):
    try:
        sqliteConnection = sqlite3.connect(PATH_TO_DATABASE)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_update_query = """Update CURRENT_PRODUCT set QUANTITY = ?, UPDATED = ? where  ID = ?"""
        columnValues = (QUANTITY, UPDATED, ID)
        cursor.execute(sqlite_update_query, columnValues)
        sqliteConnection.commit()
        print("Multiple columns updated successfully")
        sqliteConnection.commit()
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to update multiple columns not exist of sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("sqlite connection is closed")


def readRecords(LEVEL):
    try:
        sqliteConnection = sqlite3.connect(PATH_TO_DATABASE)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_select_query = """SELECT * from CURRENT_PRODUCT where LEVEL = ?"""
        cursor.execute(sqlite_select_query, (LEVEL, ))
        print("Reading single row \n")
        record = cursor.fetchall()
        print("Total rows are:  ", len(record))
        for row in record:
            if record is None:
                print("nothing exist")
            else :
                print(row[1])
                rData.append(str(row[1]))
            
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to read single row from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
readSingleRow()
for m in total:
    if m is None:
        print ("nothing")
    else:
        amount1 = int(m[0])
        amount = amount1 + 1
        for i in range(1, amount):
            updateMultipleColumnsNotExist(i, 0, datetime.datetime.now())

for x in range(1, 11):
    img_dir = os.path.join(CWD_PATH,"detect_images_row"+str(x))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    data = []
    myList = []
    myList2 = []

    for f1 in files:
        image = cv2.imread(f1)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.50)
        
        data = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
        
        for cl in data:
            if cl != None :
                myList.append(str(cl['name']))
                myList2.append(int(cl['id']))

        print (myList)
            
    strList = ''.join(str(myList))
    strList2 = ''.join(str(myList2))
    change = strList.replace('[',' ')
    change2 = change.replace(']','')
    change3 = change2.replace("'","")
    wordcount={}
        
    for word in change3.split(","):
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

    print ((wordcount))

    counting = str(wordcount)
    divide = counting.split(",")
    result = counting.replace(' ','')
    result2 = result.replace(':',',')
    result3 = re.findall('\d+', result)
    print (result)
    myall = []
    myallname = []

    for s in divide:
        b = s.replace('}','')
        c = b.replace(' ','')
        h = c.split(':')[1]
        myall.append(h)

    for n in divide:
        b = n.replace('}','')
        c = b.replace(' ','')
        h = c.split(':')[0]
        myallname.append(h)

    dirlist = os.listdir(img_dir) # dir is your directory path
    number_files = len(dirlist)

    list2 = re.findall('\d+', strList2)
    results2 = list(map(int, list2))
    list1 = re.findall("'[^']*'", result2)

    no_dupes = [x for n, x in enumerate(results2) if x not in results2[:n]]
        
    createRecordsIfNotExist()

    readRecords(x)
    print (rData)
    for f, b, l in zip(myall, myallname, no_dupes):
        name1 = b.replace("'","")
        name = name1.replace("{","")
        number = int(f)
        average = number/number_files
        print (average)
        print (name)

        if any(name in s for s in rData):
            updateMultipleColumns(name,datetime.datetime.now(), round(average),x)
        else :
            recordsToInsert = [(name, datetime.datetime.now(), datetime.datetime.now(),round(average), x)]
            insertMultipleRecords(recordsToInsert)

    rData.clear()

    filelist = [ f for f in os.listdir(img_dir) if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join(img_dir, f))

#readRecords(1)
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
    





