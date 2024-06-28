import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython.display import display, Image
import ipywidgets as widgets
import threading
from ultralytics import YOLO
import math 
from notifypy import Notify
import datetime


def predict_ShowVideo(source_model, source_video) :
    
    model = YOLO(source_model)
    source = source_video
    #model = YOLO("weights/best-yolov8m.pt")
    # Use source = 0 to specify the web cam as the video source, OR
    # specify the pathname to a video file to read.
    #source = 'video/Man walk cycle.mp4'

    # Create a video capture object from the VideoCapture Class.
    video_cap = cv2.VideoCapture(source)
    #video_cap.set(3, 640)
    #video_cap.set(4, 480)
    video_cap.set(3, 800)
    video_cap.set(4, 600)

    # Create a named window for the video display.
    win_name = 'Video Preview'
    cv2.namedWindow(win_name)
    classNames = ["larangan berhenti", "larangan masuk bagi kendaraan bermotor dan tidak bermotor", 
                  "larangan parkir", "lampu hijau", "lampu kuning", "lampu merah", "larangan belok kanan", "larangan belok kiri", 
                 "larangan berjalan terus wajib berhenti sesaat", "larangan memutar balik", "peringatan alat pemberi isyarat lalu lintas", 
                 "peringatan banyak pejalan kaki menggunakan zebra cross", "peringatan pintu perlintasan kereta api", "peringatan simpang tiga sisi kiri", 
                 "peringatan penegasan rambu tambahan", "perintah masuk jalur kiri", "perintah pilihan memasuki salah satu jalur", 
                 "petunjuk area parkir", "petunjuk lokasi pemberhentian bus", "petunjuk lokasi putar balik", "petunjuk-penyeberangan-pejalan-kaki"]
    
    listnotify = ["lampu hijau", "lampu merah", "lampu kuning", "larangan berhenti", "larangan belok kanan", "larangan belok kiri" ]
    #listnotify = ["lampu hijau", "lampu merah", "lampu kuning", "larangan berhenti"]
    t_lastnotify = datetime.datetime(year=2024, month=1, day=1)
    t_currtime =  datetime.datetime.now()
    interval = datetime.timedelta(seconds=1)
    max_interval = 7
    # Enter a while loop to read and display the video frames one at a time.
    while True:
        # Read one frame at a time using the video capture object.
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        results = model(frame, stream=False)
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                #print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                #print(f"Class name -->{classNames[cls]}<")
                
                ## notify
                if classNames[cls] in listnotify :
                    #print(f"Class name -->{classNames[cls]}, confidence : {confidence}")
                    if confidence > 0.5 :
                        t_currtime = datetime.datetime.now()
                        interval = t_currtime - t_lastnotify
                        print(f"Class name -->{classNames[cls]}, confidence : {confidence}, interval :{interval.total_seconds()}")
                        if interval.total_seconds() > max_interval :
                            notifyme(classNames[cls], confidence)
                            t_lastnotify = t_currtime

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)



        # Display the current frame in the named window.
        cv2.imshow(win_name, frame)

        # Use the waitKey() function to monitor the keyboard for user input.
        # key = cv2.waitKey(0) will display the window indefinitely until any key is pressed.
        # key = cv2.waitKey(1) will display the window for 1 ms
        key = cv2.waitKey(1)

        # The return value of the waitKey() function indicates which key was pressed.
        # You can use this feature to check if the user selected the `q` key to quit the video stream.
        if key == ord('Q') or key == ord('q') or key == 27:
            # Exit the loop.
            break
        #break
    video_cap.release()
    cv2.destroyWindow(win_name)

def notifyme(classNames, confidence):
    #print(f"inside classNames : {classNames} ")
    # Create a notification object
    notification = Notify()

    # Set the title and message for the notification
    notification.title = f"Detect Rambu"
    notification.message = f"{classNames} terdeteksi"
    notification.audio = f"{classNames}.wav"

    notification.send(block=False)

def main():
    predict_ShowVideo("best-8n-ep100-auto.pt", 0) 

if __name__ == "__main__" : 
    main()