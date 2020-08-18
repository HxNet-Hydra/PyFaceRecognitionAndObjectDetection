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
sock.setblocking(0)
try:
    sock.connect(('127.0.0.1', 63377))
except:
    print("An exception occurred")

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

def socketSend(message):
    print("Socket send: "+message)
    try: 
        sock.send(message.encode())
    except:
        print("An exception occurred")

path_images = ""
try: 
    path_images = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\face_regconition", "image_path")
except:
    path_images = "C:\\Vendron\\face_recognition_vendron\\image\\"

camera_index = -1
try: 
    camera_index = regkey_value(r"HKEY_CURRENT_USER\Software\Silkron\Vendron\face_regconition", "camera_index")
except:
    camera_index = 1

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(camera_index)
image_list = []
known_face_names = []
known_face_encodings = []
img_dir = path_images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

for f1 in files:
    path = f1
    name = os.path.basename(f1)
    name = name.replace(".jpg","")
    name = name.replace(".png","")
    image = face_recognition.load_image_file(path)
    image_list.append(image)
    known_face_names.append(name)

for image in image_list:
    encoded_image = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoded_image)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
previous_face_info = ""
active = True

while active:
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

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            if(len(known_face_encodings) > 0):
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)
            print (face_names)

    process_this_frame = not process_this_frame

    # read data from socket
    data = ""
    try:
        data = str(sock.recv(1024), 'utf-8')
    except:
        pass

    # check data receive from socket
    captureUnknown = False;
    if(data.count("capture_unknown") > 0):
        captureUnknown = True


    # process the face info
    face_info_list = []
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        fi = name + ","+ str(top) + "," + str(right) + ","  + str(bottom) + ","  + str(left)
        face_info_list.append(fi)

        #TODO: capture unknown face if receive capture_unknown from socket
        if(captureUnknown == True):
            if(name == "Unknown"):
                now = datetime.now()
                date_time = now.strftime("%Y%m%d%H%M%S%f")[:-3]
                real = date_time + ".png"
                face = frame[top:bottom, left:right]
                cv2.imwrite(os.path.join(path_images , real), face)
                # append to known face
                encoded_image = face_recognition.face_encodings(face)[0]
                known_face_encodings.append(encoded_image)
                known_face_names.append(date_time)

    # procss and send to socket only if face info is different with previous face info    
    face_info = ';'.join(map(str, face_info_list))
    if(face_info != previous_face_info):
        socketSend("face_recognized;"+face_info+"#")
        previous_face_info = face_info

    """
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
    cv2.imshow('video', frame)
    """

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
