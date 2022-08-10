import cv2
import mediapipe as mp
import numpy as np
import math

def dist(lnd1, lnd2):
	return ((lnd1[0] - lnd2[0]) ** 2 + (lnd1[1] - lnd2[1]) ** 2) ** 0.5
	

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
i = 0
zoom_flag = False
center = 0
center = 0
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image_height, image_width, _ = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
      lnd_list = []
      #print("\n ========================================\n\n")
      #print("len: ", len(results.multi_hand_landmarks))
      for hand_landmarks in results.multi_hand_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
          lnd_list.append([lm.x * image_width, lm.y * image_height])
        #print(lnd_list)
        if dist(lnd_list[4], lnd_list[8]) < image_width / 40:
          print(i, "\n")
          i+=1
          print("here\n\n")
          zoom_flag = True
        if zoom_flag:
          center_x = int((lnd_list[4][0] + lnd_list[8][0]) / 2)
          center_y = int((lnd_list[4][1] + lnd_list[8][1]) / 2)
          dist_x = int(abs(lnd_list[4][0] - lnd_list[8][0]))
          dist_y = int(abs(lnd_list[4][1] - lnd_list[8][1]))
          dist_x += 10
          dist_y += 10
          dist_x, dist_y = max(dist_x, dist_y), max(dist_x, dist_y)
          print("dist_y: ", dist_y)
          print("dist_x: ", dist_x)

          
          print("center_x, center_y: ", center_x, center_y)
          if ((center_x + 50 > image_width or center_y + 50 > image_height) or
	    (center_x - 50 < 0 or center_y - 50 < 0)):
            continue
          else:
            new_img = image[center_y - 50: center_y + 50, center_x - 50: center_x + 50]
            new_img = cv2.resize(new_img, dsize=(dist_x, dist_y))
          rnd = 0
          if dist_y % 2 == 0:
            rang_y = int(dist_y / 2)
          else:
            rang_y = int(dist_y / 2)
            rnd = 1
          if dist_x % 2 == 0:
            rang_x = int(dist_x / 2)
          else:
            rang_x = int(dist_x / 2)
            rnd = 1
          print("new_img.shape: ", new_img.shape)
          print("rang_x: ",center_x + rang_x + rnd)
          print("rang_x: ",rang_x)
          print("dist_x: ",dist_x)
          print("dist_y: ",dist_y)
          if ((center_y + rang_y + rnd > image_height or center_x + rang_x + rnd > image_width)
	    or (center_y - rang_y < 0 or center_x - rang_x < 0)):
            continue
          else:
            image[center_y - rang_y: center_y + rang_y + rnd, center_x - rang_x: center_x + rang_x + rnd] = new_img
          #cv2.imshow("a", new_img)
        #print("x: ", hand_landmarks[4])
        #print("x: ", hand_landmarks[4])
        #print("x: ", hand_landmarks[4])
        #print("x: ", hand_landmarks[4])
        #print("y: ", hand_landmarks[4])
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    else:
      zoom_flag = False
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
