import cv2
import sys
# import os
import cvlib

# DLLs
# os.add_dll_directory("E:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")

def detect_human(video_file,
                 take_every_number_frame=1,
                 yolo='yolov4',
                 continuous=False,
                 confidence=.65,
                 gpu=False):

    frame_number = 0
    while video_file.isOpened():
        frame_number = frame_number + 1
        if frame_number % take_every_number_frame != 0:
            continue

        _, frame = video_file.read()

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

            if continuous is False:
                break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cv2.imshow("Video", frame)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Can't open the camera...")
        sys.exit()

    print("Starting...")
    detect_human(cap,
                 3,
                 'yolov4-tiny',
                 True,
                 0.55,
                 False)