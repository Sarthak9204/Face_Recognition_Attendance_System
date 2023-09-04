import csv

import dlib
import cmake
import face_recognition
import cv2
import numpy as np
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Loading known's faces

my_image = face_recognition.load_image_file("miphoto.jpg")
my_image_encoding = face_recognition.face_encodings(my_image)[0]
papa_image = face_recognition.load_image_file("papa.png")
papa_image_encoding = face_recognition.face_encodings(papa_image)[0]
mom_image = face_recognition.load_image_file("mom.jpg")
mom_image_encoding = face_recognition.face_encodings(mom_image)[0]
sidd_image = face_recognition.load_image_file("sidd.jpg")
sidd_image_encoding = face_recognition.face_encodings(sidd_image)[0]
known_face_encodings = [my_image_encoding, papa_image_encoding, mom_image_encoding, sidd_image_encoding]
known_face_names = ["Sarthak", "Deepak", "Smita", "King Siddharth"]

#expected people
people = known_face_names.copy()

face_locations = []
face_encodings = []

# get current date and time

now = datetime.now()
current_date = datetime.strftime(now, "%d-%m-%Y")

f = open(f"{current_date}.csv", "a+")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()  # first arg is (is video capture successful)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # CVT = Convert

    # Recognize Faces
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for face_encoding in face_encodings:
        cmp = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_index = np.argmin(face_distance)

        if (cmp[best_index]):
            name = known_face_names[best_index]

        # Add text if person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_TRIPLEX
            bottomLeftCorner = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            linetype = 2
            cv2.putText(frame, name + " Present", bottomLeftCorner, font, fontScale, fontColor, thickness, linetype)

            if name in people:
                people.remove(name)
                current_time = datetime.strftime(now, "%H:%M:%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance_Window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
