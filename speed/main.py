import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import pandas as pd

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
	num_max = 10
	
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
		start = time.time()
		while cap.isOpened():
			success, image = cap.read()
			if time.time() - start < 3:
				print("here")
				image = cv2.flip(image, 1)
				cv2.putText(image,
					text = "{}".format(3 - int(time.time()-start)),
					org=(int(image_width/2), int(image_height/2)),
					fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=1.0,
					color=(0, 255, 0),
					thickness=2,
					lineType=cv2.LINE_4)
				cv2.imshow('MediaPipe Hands', image)
				if cv2.waitKey(5) & 0xFF == 27:
					break
				continue
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
								lnd_list.append([image_width - lm.x * image_width, lm.y * image_height])
				target = point_list[search_number - 1]
				print(search_number)
				g = is_inrect(lnd_list, target)
				if g:
					search_number += 1
					if search_number == num_max + 1:
						break
	
				mp_drawing.draw_landmarks(
						image,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing_styles.get_default_hand_landmarks_style(),
						mp_drawing_styles.get_default_hand_connections_style())
			image = cv2.flip(image, 1)
			cv2.putText(image,
			       text = "{}".format(search_number),
			       org=(50, 100),
			       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			       fontScale=5.0,
			       color=(100, 100, 255),
			       thickness=10,
			       lineType=cv2.LINE_4)
			for n in range(num_max):
				cv2.putText(image,
					text = "{}".format(n+1),
					org=point_list[n],
					fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=1.0,
					color=(0, 255, 0),
					thickness=2,
					lineType=cv2.LINE_4)
			# Flip the image horizontally for a selfie-view display.
			#cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
			cv2.imshow('MediaPipe Hands', image)
			if cv2.waitKey(5) & 0xFF == 27:
				break
	score = time.time() - start - 3
	cv2.destroyWindow("MediaPipe Hands")
	cap.release()
	print("released")
	return score
import PySimpleGUI as sg

def main():
	df = pd.read_csv('./time.csv')
	df = df.sort_values(by="time")
	print(df)
	print("=====")
	print(df["time"].iloc[0])
	score1 = 100
	score2 = 100
	score3 = 100
	score4 = 100
	score5 = 100
	if len(df) >= 1:
		score1 = df["time"].iloc[0]
	if len(df) >= 2:
		score2 = df["time"].iloc[1]
	if len(df) >= 3:
		score3 = df["time"].iloc[2]
	if len(df) >= 4:
		score4 = df["time"].iloc[3]
	if len(df) >= 5:
		score5 = df["time"].iloc[4]
	s_button = sg.Submit('Start', size=(10, 10), button_color=('black', '#4adcd6'))
	score_text1 = sg.Text('1位: {:.2f}'.format(score1), font=('Noto Serif CJK JP', 50), key='-RANK1-')
	score_text2 = sg.Text('2位: {:.2f}'.format(score2), font=('Noto Serif CJK JP', 50), key='-RANK2-')
	score_text3 = sg.Text('3位: {:.2f}'.format(score3), font=('Noto Serif CJK JP', 50), key='-RANK3-')
	score_text4 = sg.Text('3位: {:.2f}'.format(score4), font=('Noto Serif CJK JP', 50), key='-RANK4-')
	score_text5 = sg.Text('3位: {:.2f}'.format(score5), font=('Noto Serif CJK JP', 50), key='-RANK5-')
	score_text6 = sg.Text('今回のタイム: {}'.format(0), font=('Noto Serif CJK JP', 50), key='-RANK6-')
	layout1 = sg.Frame(layout=[[score_text1],
				#[score_text1_1],
				[score_text2],
				#[score_text2_1],
				[score_text3],
				[score_text4],
				[score_text5]],
				#[score_text3_1],
				title='score1',
				title_color='white',
				font=('メイリオ', 12),
				relief=sg.RELIEF_SUNKEN,
				element_justification='left')
	layout = [[layout1, score_text6], [s_button]]
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
			df = df.append({'time': score}, ignore_index=True)
			print("===df===")
			print(df)
			df.to_csv('./time.csv', index=False)
			df = df.sort_values(by="time")
			score1 = df["time"].iloc[0]
			score2 = 100
			score3 = 100
			score4 = 100
			score3 = 100
			if len(df) >= 2:
				score2 = df["time"].iloc[1]
			if len(df) >= 3:
				score3 = df["time"].iloc[2]
			if len(df) >= 4:
				score4 = df["time"].iloc[3]
			if len(df) >= 5:
				score5 = df["time"].iloc[4]
			window['-RANK1-'].update('1位: {:.2f}'.format(score1))
			window['-RANK2-'].update('2位: {:.2f}'.format(score2))
			window['-RANK3-'].update('3位: {:.2f}'.format(score3))
			window['-RANK4-'].update('3位: {:.2f}'.format(score4))
			window['-RANK5-'].update('3位: {:.2f}'.format(score5))
			window['-RANK6-'].update('今回のスコア {:.2f}'.format(score))
		if event == sg.WIN_CLOSED:
			break
if __name__ == "__main__":
	main()
