from __future__ import print_function

from datetime import datetime

from robolab_turtlebot import Turtlebot, sleep

from scipy.io import savemat
import numpy as np
import os

# initialize turlebot
turtle = Turtlebot(rgb=True, depth=True, pc=True)

# sleep 2 set to receive images
sleep(2)

save_dir = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(save_dir, exist_ok=True)

img = turtle.get_rgb_image()
np.save(os.path.join(save_dir, datetime.today().strftime("%Y-%m-%d-%H-%M-%S") + '.npy'), img)
turtle.play_sound()

print('Data saved in {}'.format(save_dir))
