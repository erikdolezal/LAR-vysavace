from __future__ import print_function

from datetime import datetime

from robolab_turtlebot import Turtlebot, sleep, Rate

import numpy as np
import os
import cv2

BUTTON_NAMES = ['Button0', 'Button1', 'Button2']
STATE = ['RELEASED', 'PRESSED']
WINDOW = 'image'

save_dir = datetime.today().strftime('%Y-%m-%d_%H-%M-%S') 
os.makedirs(save_dir, exist_ok=True)

def button_cb(msg):
    print('Button: {} {}'.format(BUTTON_NAMES[msg.button], STATE[msg.state]))
    if STATE[msg.state] == 'PRESSED':
        img_name = datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + '.npy'
        img = turtle.get_rgb_image()
        np.save(save_dir + '/' + img_name, img)
        print('Image saved as {}'.format(os.path.join(save_dir, img_name)))
        turtle.play_sound()
        cv2.imshow(WINDOW, img)
        cv2.waitKey(1000)


if __name__ == '__main__':
    cv2.namedWindow(WINDOW)
    turtle = Turtlebot(rgb=True, depth=True, pc=True)
    
    print("waiting for button event")
    turtle.register_button_event_cb(button_cb)

    while not turtle.is_shutting_down():
        rate = Rate(1)
