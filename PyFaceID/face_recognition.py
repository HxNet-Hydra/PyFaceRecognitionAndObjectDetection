import face_recognition
import cv2
import numpy as np
import socket
import winreg
import os
import glob
from datetime import datetime

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Connect Socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 63388))

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
path_images = regkey_value(r"HKEY_CURRENT_USER\Software\", "path")

def facecrop():
    #capture images for face crop
    cam = cv2.VideoCapture(0)
    s, img = cam.read()
    now = datetime.now()
    date_time = now.strftime("%m%d%Y%H%M%S")	
    real = date_time + ".png"
    cv2.imwrite(os.path.join(path_images , real), img)
    cam.release()

    #facecrop start
    image = os.path.join(path_images , real)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(image)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)
    counter = 0
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        cv2.imwrite(fname+"_cropped_"+str(counter)+ext, sub_face)
        counter += 1

    #remove original image
    os.remove(image)
    

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
encoding_list = []
known_face_names2 = []
known_face_names = []
known_face_encodings = []
img_dir = path_images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
#known_face_encodings = []
for f1 in files:
    path = f1
    name2 = os.path.basename(f1)
    name3 = name2.replace(".jpg","")
    name1 = name3.replace(".png","")
    name  = os.path.basename(f1)
    name_encoding = name1 + "face_encoding"
    name = face_recognition.load_image_file(path)
    encoding_list.append(name_encoding)
    known_face_names.append(name1)
    known_face_names2.append(name)

print (known_face_names)
for ea,ea_name in zip(encoding_list,known_face_names2):
    ea = face_recognition.face_encodings(ea_name)[0]
    known_face_encodings.append(ea)

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True
Unknown = "Unknown"
previous_face_names = "";
active = True

while active:
    face_names = []
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = Unknown

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # append matched name to array    
            face_names.append(name)
            print (face_names)
            face_name_string = ';'.join(map(str, face_names))

            # procss only if face names is different with previous face names
            if(face_name_string != previous_face_names):
                print(face_name_string)
                for each in face_names:
                    print (each)
                    sock.send(each.encode())
                    if each is Unknown:
                        #TODO: capture and append to facial id image path 
                        video_capture.release()
                        facecrop()
                        active = False
                previous_face_names = face_name_string
            
    process_this_frame = not process_this_frame

    
    # Display the results 
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    #cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
