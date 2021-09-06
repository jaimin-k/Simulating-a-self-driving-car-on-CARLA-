import glob
import os
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

throttle_list = []
steer_list = []
data_th_st = [[]]
image_list_cp = []

def createFileList(myDir, format='.png'):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def attribute(camera_bp,X,Y,Z,p,ya,r):
    camera_bp.set_attribute('image_size_x', f'{img_w}')
    camera_bp.set_attribute('image_size_y', f'{img_h}')
    camera_bp.set_attribute('fov', '110')
    relative_spawn = carla.Transform(carla.Location(x=X, y=Y, z=Z), carla.Rotation(pitch=p, yaw=ya, roll=r))
    return relative_spawn

def collect(image):
	image.save_to_disk('D:/Final Year Project/CARLA_0.9.5/PythonAPI/examples/_out/img%d.png' % image.frame_number)
	throttle_list.append(vehicle.get_control().throttle)
	steer_list.append(vehicle.get_control().steer)
	return

def main():
	camera_bp =  blueprint.find('sensor.camera.rgb')
	spa=attribute(camera_bp,2.5,0,1,-7,0,0)#2.5,0.0,1.7,0,0,0)#-4,0,3.5,-10,0,0)#0.1,-0.3,1.1,0,0,0)
	rgb_cam = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
	rgb_cam.listen(lambda image: collect(image))
	
	
	vehicle.set_autopilot(True)
	time.sleep(300)

	rgb_cam.destroy()
	vehicle.destroy()

	image_list = createFileList('D:\Final Year Project\CARLA_0.9.5\PythonAPI\examples\_out')#'/path_to_directory_with_images/')
	print('image collection done!')

	diff = len(image_list) - len(throttle_list)
	tot = len(image_list) - diff  
	i = 1
	for i in range(tot):
		image_list_cp.append(image_list[i])

	
	final_data = np.vstack((image_list_cp,throttle_list, steer_list)).T
	print('stack together done!')

	#print(final_data)

	np.savetxt('final_data.txt', final_data, fmt='%s    %s  %s')	
	print('Data Saved!')
	#data_th_st = np.vstack((throttle_list, steer_list)).T
	#np.savetxt('data_th_st.txt', data_th_st)
	
	
	
	#print(throttle_list)

main()
print('successfull')