
#pip3 install --upgrade tensorflow

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
#model = load_model('model.h5')
model = tf.keras.models.load_model('D:\Final Year Project\CARLA_0.9.5\PythonAPI\examples\model.h5')



import glob
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import pygame
import math
import random
import argparse
import time
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
client.set_timeout(5.0) # seconds
#world = client.load_world('Town05')
world = client.get_world()
map1 = world.get_map() 

start_loc = carla.Location(x=-85, y=145, z=1.8)
blueprint = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint.filter('vehicle.tesla.model3'))
spawn_point = carla.Transform(start_loc, carla.Rotation(pitch=0, yaw=90, roll=0))#-85 145 1 
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
time.sleep(2)
img_h = 800 #600
img_w = 800 #600


def attribute(camera_bp,X,Y,Z,p,ya,r):
    camera_bp.set_attribute('image_size_x', f'{img_w}')
    camera_bp.set_attribute('image_size_y', f'{img_h}')
    camera_bp.set_attribute('fov', '110')
    relative_spawn = carla.Transform(carla.Location(x=X, y=Y, z=Z), carla.Rotation(pitch=p, yaw=ya, roll=r))
    return relative_spawn

def process_img(image):
    i=np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    img = img_preprocess(i3)
    model_control(img)


def img_preprocess(i):
	img = i[375:625,:,:]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	img = cv2.GaussianBlur(img, (3,3),0)
	img = cv2.resize(img, (200, 66))
	return img

def model_control(img):
	i = np.array([img])
	image = tf.cast(i, tf.float32)
	steer = model.predict(image)
	st = float(steer[0,0])
	vehicle_control(0.5,st)

def vehicle_control(th,st):
    vehicle.apply_control(carla.VehicleControl(throttle=th, steer=st))

def main():
	camera_bp =  blueprint.find('sensor.camera.rgb')
	spa=attribute(camera_bp,2.5,0,1,-7,0,0)#2.5,0.0,1.7,0,0,0)#-4,0,3.5,-10,0,0)#0.1,-0.3,1.1,0,0,0)
	rgb_cam = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
	rgb_cam.listen(lambda image: process_img(image))

	time.sleep(10)

	rgb_cam.destroy()
	vehicle.destroy()

main()

