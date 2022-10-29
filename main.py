import numpy as np
import cv2
import sys
import os
import cvlib
import face_recognition

faces_directory = 'Faces' 

def detect_human(video_file,
                 take_every_number_frame=1,
                 yolo='yolov4',
                 confidence=.65,
                 gpu=False):

    frame_number = 0
    while video_file.isOpened():
        frame_number = frame_number + 1

        flag, frame = video_file.read()
        if not flag:
            continue

        frame_number = frame_number + 1

        if frame_number % take_every_number_frame != 0:
            continue

        try:
            bbox, labels, conf = cvlib.detect_common_objects(frame,
                                                             model=yolo,
                                                             confidence=confidence,
                                                             enable_gpu=gpu)
        except Exception as e:
            print(e)
            break

        if 'person' in labels:
            frame = cvlib.object_detection.draw_bbox(frame,
                                                     bbox,
                                                     labels,
                                                     conf,
                                                     write_conf=True)
            print('Frame number ' + str(frame_number)
                  + ': people count is ' + str(labels.count('person')))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        cv2.imshow("Video", frame)


def recognize_face(video_file,
                   skipped_frames_count=12):
    frame_number = 0
    while video_file.isOpened():
        frame_number = frame_number + 1
        if frame_number != skipped_frames_count:
            continue

        # Frame from cam
        _, frame = video_file.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces_list = face_recognition.face_encodings(frame_rgb)
        
        if len(faces_list) > 0:
            for face_filename in os.scandir(faces_directory):
                if not face_filename.is_file():
                    continue

                # Face from cam frame
                image_encoding = faces_list[0]

                # Current face from directory
                known_face = cv2.imread(face_filename.path)
                known_face_rgb = cv2.cvtColor(known_face, cv2.COLOR_BGR2RGB)
                known_face_rgb_encoded = face_recognition.face_encodings(known_face_rgb)[0]

                # Trying to identify a person 
                result = face_recognition.compare_faces([known_face_rgb_encoded], image_encoding)

                if (len(result) > 0) and (result[0] is True):
                    print(f"Face match found with face presented in {face_filename}")
                    break
            
        frame_number = 0

        cv2.imshow("Video", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Can't open the camera...")
        sys.exit()

    print("Starting...")
    # detect_human(cap, 5, 'yolov4-tiny', 0.55, False)
    recognize_face(cap)

