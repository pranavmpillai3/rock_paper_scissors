from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import cv2
import random
import threading
import time

model = YOLO('/home/hackerpro/CV/RockPaperScissor_LiveCam/models/best.pt')

image_path = ['images/rock.png', 'images/paper.png', 'images/scissors.png']
computer_choice = None
user_choice = None

cap = cv2.VideoCapture(0)

game_on = True
player = 0
computer = 0

def delayed_loop():
    for i in range(5):
        time.sleep(1)

def display_images():
    global computer_choice
    while not exit_event.is_set():
        computer_choice = random.randint(0, 2)
        image = cv2.imread(image_path[computer_choice])
        image = cv2.resize(image, (640,640))
        cv2.imshow('Computer', image)
        cv2.waitKey(1)


def video_feed():
    global user_choice
    while not exit_event.is_set():
        success, img = cap.read()
        if success:
            resized_img = cv2.resize(img, (640, 640))
            results = model(resized_img, conf=0.5)
            annotated_frame = results[0].plot()
            choice = results.names[0]
            if choice == 'Rock':
                user_choice = 0
            elif choice == 'Paper':
                user_choice = 1
            elif choice == 'Scissors':
                user_choice = 2
            cv2.imshow('Player', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

exit_event = threading.Event()

image_thread = threading.Thread(target=display_images)
video_thread = threading.Thread(target=video_feed)
loop_thread = threading.Thread(target=delayed_loop)


image_thread.start()
video_thread.start()
loop_thread.start()

while game_on:
    if player < 10 and computer < 10:
        if computer_choice == 0 and user_choice == 0:
            continue
        elif computer_choice == 0 and user_choice == 1:
            player += 1
        elif computer_choice == 0 and user_choice == 2:
            computer += 1
        elif computer_choice == 1 and user_choice == 0:
            computer += 1
        elif computer_choice == 1 and user_choice == 1:
            continue
        elif computer_choice == 1 and user_choice == 2:
            player += 1
        if computer_choice == 2 and user_choice == 0:
            computer += 1
        elif computer_choice == 2 and user_choice == 1:
            player += 1
        elif computer_choice == 2 and user_choice == 2:
            continue
    else:
        game_on = False

exit_event.set()

image_thread.join()
video_thread.join()
loop_thread.join()

cap.release()
cv2.destroyAllWindows()
