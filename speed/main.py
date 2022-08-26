import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

def is_inrect(lst, target):
	sum_x = 0
	sum_y = 0
	for l in lst:
		sum_x += l[0]
		sum_y += l[1]
	if len(lst) != 0:
		sum_x /= len(lst)
		sum_y /= len(lst)
	if sum_x - 100 < target[0] and sum_x +100 > target[0] and sum_y -100 < target[1] and sum_y + 100 > target[1]:
		print("in")
		return True
	print("not in")
	return False

def touch_number():
	mp_drawing = mp.solutions.drawing_utils
	mp_drawing_styles = mp.solutions.drawing_styles
	mp_hands = mp.solutions.hands
	# For webcam input:
	cap = cv2.VideoCapture(0)
	_, image = cap.read()
	image_height, image_width, _ = image.shape
	i = 0
	search_number = 1
	num_max = 15
	
	image = cv2.resize(image, (1920, 1080))
	point_list = []
	for _ in range(num_max):
		x = random.randint(100, image_width)
		y = random.randint(80, image_height)
		point_list.append((x, y))
	
	with mp_hands.Hands(
			model_complexity=0,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as hands:
		while cap.isOpened():
			success, image = cap.read()
			for n in range(num_max):
				cv2.putText(image,
					text = "{}".format(n+1),
					org=point_list[n],
					fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=1.0,
					color=(0, 255, 0),
					thickness=2,
					lineType=cv2.LINE_4)
	
			cv2.putText(image,
			       text = "{}".format(search_number),
			       org=(50, 100),
			       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			       fontScale=5.0,
			       color=(100, 100, 255),
			       thickness=10,
			       lineType=cv2.LINE_4)
			if not success:
				print("Ignoring empty camera frame.")
				# If loading a video, use 'break' instead of 'continue'.
				continue
	
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
				hand1, hand2= [], []
				hand_cnt = 0
				for hand_landmarks in results.multi_hand_landmarks:
					hand_num = len(results.multi_hand_landmarks)
					if hand_num == 1:
						for id, lm in enumerate(hand_landmarks.landmark):
								lnd_list.append([lm.x * image_width, lm.y * image_height])
				target = point_list[search_number - 1]
				print(search_number)
				g = is_inrect(lnd_list, target)
				if g:
					search_number += 1
					if search_number == 16:
						break
	
				mp_drawing.draw_landmarks(
						image,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing_styles.get_default_hand_landmarks_style(),
						mp_drawing_styles.get_default_hand_connections_style())
			# Flip the image horizontally for a selfie-view display.
			cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
			if cv2.waitKey(5) & 0xFF == 27:
				break
	cv2.destroyWindow("MediaPipe Hands")
	cap.release()
	print("released")
	return 10
import PySimpleGUI as sg

def main():
	s_button = sg.Submit('Start', size=(10, 10), button_color=('black', '#4adcd6'))
	score_text1 = sg.Text('1: ', font=('Noto Serif CJK JP', 50), key='-RANK1-')
	score_text2 = sg.Text('2: ', font=('Noto Serif CJK JP', 50), key='-RANK2-')
	score_text3 = sg.Text('3: ', font=('Noto Serif CJK JP', 50), key='-RANK3-')
	layout1 = sg.Frame(layout=[[score_text1],
				#[score_text1_1],
				[score_text2],
				#[score_text2_1],
				[score_text3]],
				#[score_text3_1],
				title='score1',
				title_color='white',
				font=('メイリオ', 12),
				relief=sg.RELIEF_SUNKEN,
				element_justification='left')
	layout = [[layout1], [s_button]]
	window = sg.Window('tomo viewr2', layout,
				location=(30,30),
				alpha_channel=1.0,
				no_titlebar=False,
				grab_anywhere=False,
				resizable=True).Finalize()
	while True:
		event, values = window.read(timeout=20)
		if event == "Start":
			score = touch_number()
			window['-RANK1-'].update(score)
			print("score: ", score)
		if event == sg.WIN_CLOSED:
			break
if __name__ == "__main__":
	main()
