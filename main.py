import cv2
import mediapipe as mp
import numpy as np
import math

def mosaic(img, alpha):
	h, w, ch = img.shape

	img = cv2.resize(img, (int(w*alpha), int(h*alpha)))
	img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

	return img

def decreaseColor(img):
    dst = img.copy()
    
    idx = np.where((0<=img) & (32>img))
    dst[idx] = 16
    idx = np.where((32<=img) & (64>img))
    dst[idx] = 48   
    idx = np.where((64<=img) & (96>img))
    dst[idx] = 80
    idx = np.where((96<=img) & (128>img))
    dst[idx] = 112
    idx = np.where((128<=img) & (160>img))
    dst[idx] = 144
    idx = np.where((160<=img) & (192>img))
    dst[idx] = 176
    idx = np.where((192<=img) & (224>img))
    dst[idx] = 208
    idx = np.where((224<=img) & (256>=img))
    dst[idx] = 240
    
    return dst

def dist(lnd1, lnd2):
	return ((lnd1[0] - lnd2[0]) ** 2 + (lnd1[1] - lnd2[1]) ** 2) ** 0.5
	

def check_hand_pose(hand1, hand2):
	thumb1 = np.array(hand1[4])
	pointer1 = np.array(hand1[8])
	corner1 = np.array(hand1[0])

	thumb2 = np.array(hand2[4])
	pointer2 = np.array(hand2[8])
	corner2 = np.array(hand2[0])

	vector1_1 = thumb1 - corner1
	vector1_2 = pointer1 - corner1

	vector2_1 = thumb2 - corner2
	vector2_2 = pointer2 - corner2

	i1 = np.inner(vector1_1, vector1_2)
	i2 = np.inner(vector2_1, vector2_2)
	n1 = np.linalg.norm(vector1_1) * np.linalg.norm(vector1_2)
	n2 = np.linalg.norm(vector2_1) * np.linalg.norm(vector2_2)
	c1 = i1 / n1
	c2 = i2 / n2
	ang1 = np.rad2deg(np.arccos(np.clip(c1, -1.0, 1.0)))
	ang2 = np.rad2deg(np.arccos(np.clip(c2, -1.0, 1.0)))
	print("angle1: ", ang1, "\n")
	print("angle2: ", ang2, "\n")
	if ang1 > 45 and ang2 > 45:
		return True
	return False

def draw_stealth(img, hand1, hand2):
	bg_img = cv2.imread('./bg.png')
	zoom_size_x = int(abs(hand1[0][0] - hand2[0][0]))
	zoom_size_y = int(abs(hand1[0][1] - hand2[0][1]))

	print("zoom_size_x: ", zoom_size_x, "\n")
	print("zoom_size_y: ", zoom_size_y, "\n")
	try:
		print(img.shape)
		new_bg = cv2.resize(bg_img, dsize=(img.shape[1], img.shape[0]))
		pos_x = int(min(hand1[0][0], hand2[0][0]))
		pos_y = int(min(hand1[0][1], hand2[0][1]))
		cut_bg = new_bg[pos_y: pos_y + zoom_size_y, pos_x: pos_x + zoom_size_x]
		img[pos_y: pos_y + zoom_size_y, pos_x: pos_x + zoom_size_x] = cut_bg
	except:
		print("error")
	return img

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
dot_flag = False
thresh_flag = False
threshold=100
line_x = 500
line2_y = 300
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
    image_tmp = image
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    cv2.line(image, pt1=(line_x, 0), pt2=(line_x, image_height),
    	color=(255, 0, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
    cv2.line(image, pt1=(0, line2_y), pt2=(image_width, line2_y),
    	color=(255, 0, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
    #ret, image_thresh = cv2.threshold(image_tmp, threshold, 255, cv2.THRESH_BINARY)
    image_mosaic = mosaic(image_tmp, 0.2)
    image_dot = decreaseColor(image_mosaic)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    image[line2_y:, :] = np.array(image_dot[line2_y:, :])
    image[:, :line_x] = np.array(image_thresh[1][:, :line_x])
    
    if results.multi_hand_landmarks:
      lnd_list = []
      #print("\n ========================================\n\n")
      #print("len: ", len(results.multi_hand_landmarks))
      hand1, hand2= [], []
      hand_cnt = 0
      for hand_landmarks in results.multi_hand_landmarks:
        hand_num = len(results.multi_hand_landmarks)
        if hand_num == 1:
            for id, lm in enumerate(hand_landmarks.landmark):
              lnd_list.append([lm.x * image_width, lm.y * image_height])
            #print(lnd_list)
            center_x = (lnd_list[4][0] + lnd_list[8][0]) / 2
            center_y = (lnd_list[4][1] + lnd_list[8][1]) / 2
            print("center_x: ", center_x)
            print("dist: ", dist(lnd_list[4], lnd_list[8]) < image_width / 40)
            if dist(lnd_list[4], lnd_list[8]) < image_width / 30\
                and center_x > line_x - 60 and center_x < line_x + 60:
              thresh_flag = True
              print("thresh_flag: ", thresh_flag)
            elif dist(lnd_list[4], lnd_list[8]) < image_width / 30\
                and center_y > line2_y- 60 and center_y < line2_y + 60:
              dot_flag = True
            elif dist(lnd_list[4], lnd_list[8]) < image_width / 40:
              print(i, "\n")
              i+=1
              print("here\n\n")
              zoom_flag = True
            if thresh_flag:
              print("thresh_flag if")
              if dist(lnd_list[4], lnd_list[8]) > image_width / 40:
                thresh_flag = False
              line_x = int((lnd_list[4][0] + lnd_list[8][0]) / 2)
              cv2.line(image, pt1=(line_x, 0), pt2=(line_x, image_height),
              	color=(255, 0, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
              #ret, image_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
              image[:, :line_x] = image_thresh[1][:, :line_x]
            if dot_flag:
              print("thresh_flag if")
              if dist(lnd_list[4], lnd_list[8]) > image_width / 40:
                dot_flag = False
              line2_y = int((lnd_list[4][1] + lnd_list[8][1]) / 2)
              cv2.line(image, pt1=(0, line2_y), pt2=(image_width, line2_y),
              	color=(255, 0, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
              #ret, image_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
              image[line2_y:, :] = np.array(image_dot[line2_y:, :])
            if zoom_flag and not thresh_flag:
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
        elif hand_num == 2:
            if hand_cnt == 0:
                for id, lm in enumerate(hand_landmarks.landmark):
                  hand1.append([lm.x * image_width, lm.y * image_height])
            else:
                for id, lm in enumerate(hand_landmarks.landmark):
                  hand2.append([lm.x * image_width, lm.y * image_height])
                if check_hand_pose(hand1, hand2):
                    image = draw_stealth(image, hand1, hand2)
            hand_cnt += 1
        #print("x: ", hand_landmarks[4])
        #print("x: ", hand_landmarks[4])
        #print("x: ", hand_landmarks[4])
        #print("x: ", hand_landmarks[4])
        #print("y: ", hand_landmarks[4])
        print("line_x: ", line_x)
        print("image_thresh[1].shape: ", image_thresh[1].shape)
        print("image_thresh[1]: ", image_thresh[1])
        print("image_tmp.shape: ", image_tmp.shape)
        #image[:, :line_x] = np.array(image_thresh[1][:, :line_x])
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
