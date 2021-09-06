
#------------------------------------------------------------------------------------------
#                               IMPORTS AND GLOBAL VARIABLES
#------------------------------------------------------------------------------------------


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
import math
import random
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from PIL import Image
import pygame
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
#import win32gui, win32ui, win32con, win32api
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption

np.warnings.filterwarnings('ignore')
sensor_list = []
vehicle_list = []
route_map_list = []
og_image = []
global count
count = 0
#print(og_image.shape)

client = carla.Client('localhost', 2000)
client.set_timeout(5.0) # seconds
#world = client.load_world('Town04')
world = client.get_world()

blueprint = world.get_blueprint_library()
global vehicle_bp,vehicle
vehicle_bp = random.choice(blueprint.filter('vehicle.tesla.model3'))
start_loc = carla.Location(x=-85, y=145, z=1.8)
spawn_point = carla.Transform(start_loc, carla.Rotation(pitch=0, yaw=90, roll=0))
#spawn_point = carla.Transform(carla.Location(x=10, y=100, z=1), carla.Rotation(pitch=0, yaw=90, roll=0))#-85 145 1 
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

img_h = 800 #600
img_w = 800 #600

map1 = world.get_map() 

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)


#------------------------------------------------------------------------------------------
#                                       PROCESSES
#------------------------------------------------------------------------------------------


def lanes_data():
    '''waypoint = map1.get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    lane_type = waypoint.lane_type
    left_lm_type = waypoint.left_lane_marking.type
    right_lm_type = waypoint.right_lane_marking.type
    lane_change = waypoint.lane_change
    waypoint_list = map1.generate_waypoints(2.0)
    #behavioural_model()
    print('lane type = ',lane_type)
    print('left lane type = ',left_lm_type)
    print('right lane type = ',right_lm_type)
    print('lane change = ',lane_change)
    print('waypoint = ',map1)
    #print('vehicle location = ',vehicle.get_location())
    print('-----------------------------------------')
    return'''
    pass


def py_disp(image,py,name):
    if py == True:
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = frame.swapaxes(0,1)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0,0))
        pygame.display.update()
    else:
        cv2.imshow("{}!.".format(name), image)
        cv2.waitKey(1)
    return

def mat_disp(image):
    plt.imshow(image)
    plt.show()



def process_Lanes(image,py):
    cc=carla.ColorConverter
    image.convert(cc.CityScapesPalette)
    i = np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    #color1 = np.asarray([128, 64, 128])
    color1 = np.asarray([0,220,220])#([50,234,157])#([128, 64, 128])      
    #color2 = np.asarray([0, 234, 50])
    mask = cv2.inRange(i3, color1, color1)
    i4 = mask
    x=np.dstack((i4,i4))
    y=np.dstack((x,i4))
    wpix = count_of_pixel(y) #1425 2007
    canny_image = canny(y)
    lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 10, np.array([]),minLineLength=0.1, maxLineGap=5)
    line_image = display_lines(og_image[0],lines)
    masked_image = cv2.bitwise_and(og_image[0],y) 
    #combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    py_disp(masked_image,True,'Lanes')
    return

def count_of_pixel(image):
    n_white_pix = np.sum(image == 255)
    print(n_white_pix)
    return n_white_pix

def on_going(image):
    global count
    i = np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    if count == 0:
        og_image.append(i3)
        count = count + 1
    else:
        og_image[0] = i3
    return 




def process_img(image,name,py):
    i=np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    py_disp(i3,py,name)
    return i3


def process_img_convert(image,index,py):
    cc=carla.ColorConverter
    sensors = [
        [cc.Raw, 'Camera RGB'],
        [cc.Raw, 'Camera Depth (Raw)'],
        [cc.Depth, 'Camera Depth (Gray Scale)'],
        [cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
        [cc.Raw, 'Camera Semantic Segmentation (Raw)'],
        [cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
        ]
    image.convert(sensors[index][0])
    i=process_img(image,sensors[index][1],py) 
    return i

def lidar_process(image):
    #dim=[480,640]
    dim=[img_h,img_w]
    points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 3), 3))
    lidar_data = np.array(points[:, :2])
    lidar_data *= min(dim) / 100.0
    lidar_data += (0.5 * dim[0], 0.5 * dim[1])
    lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img_size = (dim[0], dim[1], 3)
    lidar_img = np.zeros(lidar_img_size)
    lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
    l_i = lidar_img.copy()
    frame = l_i.swapaxes(0,1)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))
    pygame.display.update()
    return


def testing_process(image):
    i=np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    frame = cv2.cvtColor(i3, cv2.COLOR_BGR2RGB)
    left_upper_x = (41.6/100)*img_w # 250
    left_upper_y = (50/100)*img_h #300
    right_upper_x = (58.3/100)*img_w #350
    right_upper_y = (50/100)*img_h #300    
    pts1 = np.float32([[left_upper_x, left_upper_y], [right_upper_x, right_upper_y], [img_w, img_h], [0, img_h]])
    pts2 = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (img_w, img_h))
    a = result
    lane_image = np.copy(a)
    canny_image = canny(lane_image)
    #cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(lane_image,lines)
    line_image = display_lines(lane_image,averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    py_disp(combo_image,True,'lanes')
    




def process_trial(image):
    i=np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    imag = i3
    i4 = canny(imag)
    height = i4.shape[0]
    left_upper_x = (41.6/100)*img_w # 250
    left_upper_y = (50/100)*img_h #300
    right_upper_x = (58.3/100)*img_w #350
    right_upper_y = (50/100)*img_h #300 
    polygons = np.array([
        [(left_upper_x,left_upper_y),(right_upper_x,right_upper_y),(img_w,height),(0,height)]#,(300,250)]
        ])
    mask = np.zeros_like(i4)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(i3,mask)
    py_disp(masked_image,True,'aaaa')
    return masked_image


def attribute(camera_bp,X,Y,Z,p,ya,r):
    camera_bp.set_attribute('image_size_x', f'{img_w}')
    camera_bp.set_attribute('image_size_y', f'{img_h}')
    camera_bp.set_attribute('fov', '110')
    relative_spawn = carla.Transform(carla.Location(x=X, y=Y, z=Z), carla.Rotation(pitch=p, yaw=ya, roll=r))
    return relative_spawn;





#------------------------------------------------------------------------------------------
#                                       SENSORS
#------------------------------------------------------------------------------------------



def rgb_camera():  #display
    global rgb_cam
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,-4,0,3.5,-10,0,0)#0.1,-0.3,1.1,0,0,0)
    rgb_cam = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    rgb_cam.listen(lambda data: process_img(data,'rgb',True))
    sensor_list.append(rgb_cam)
    r = True
    return r

def driver_cam(): #display
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,0.1,-0.3,1.1,0,0,0)
    rgb_cam = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    rgb_cam.listen(lambda data: process_img(data,'rgb',True))
    sensor_list.append(rgb_cam)
    r = True
    return r

def rgb_camera2(): #process
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,2.5,0,1,-7,0,0)#0.1,-0.3,1.1,0,0,0)
    rgb_cam2 = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    rgb_cam2.listen(lambda data: process_img_Lanes(data))
    sensor_list.append(rgb_cam2)
    r = True
    return r

def rgb_camera3(): #process
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,2.5,0,2.5,-5,0,0)#0.1,-0.3,1.1,0,0,0)
    rgb_cam3 = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    rgb_cam3.listen(lambda data: testing_process(data))
    sensor_list.append(rgb_cam3)
    r = True
    return r



def ss_cam(): #display / process
    sem_bp = blueprint.find('sensor.camera.semantic_segmentation')
    spas=attribute(sem_bp,2.5,0.0,1.7,0,0,0)
    sem = world.spawn_actor(sem_bp, spas, attach_to=vehicle)
    sensor_list.append(sem)
    sem.listen(lambda data: process_Lanes(data,True))


def rgb_camera4(): #process
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,2.5,0.0,1.7,0,0,0)
    rgb_cam4 = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    rgb_cam4.listen(lambda data: traffic(data))#on_going(data))
    sensor_list.append(rgb_cam4)
    return

def traffic(image):
    i=np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    py_disp(i3,True,'abc')



def ss_cam2(): #process
    sem_bp2 = blueprint.find('sensor.camera.semantic_segmentation')
    spas2=attribute(sem_bp2,50,0,145,-90,0,0)
    sem2 = world.spawn_actor(sem_bp2, spas2)
    sensor_list.append(sem2)
    sem2.listen(lambda data: process_Lanes(data,True))


def lidar_cam(): #process
    lidar_bp = blueprint.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '5000')
    spal = carla.Transform(carla.Location(x=2.5, y=0.0, z=1.4))
    lidar = world.spawn_actor(lidar_bp, spal, attach_to=vehicle)
    sensor_list.append(lidar)
    lidar.listen(lambda data: lidar_process(data))



def gods_eye(): #display
    god_eye_bp =  blueprint.find('sensor.camera.rgb')
    god_eye_bp.set_attribute('image_size_x', f'{img_w}')
    god_eye_bp.set_attribute('image_size_y', f'{img_h}')
    god_eye_bp.set_attribute('fov', '110')
    spg = carla.Transform(carla.Location(x=50, y=0.0, z=145), carla.Rotation(pitch=-90, yaw=0, roll=0))
    god_eye = world.spawn_actor(god_eye_bp, spg)
    sensor_list.append(god_eye)
    god_eye.listen(lambda data: process_img(data,'gods eye',True))
    r = True
    return r


def depth_cam(): #process
    depth_bp = blueprint.find('sensor.camera.depth')
    spad=attribute(depth_bp,2.5,0.0,0.7,0,0,0)
    depth = world.spawn_actor(depth_bp, spad, attach_to=vehicle)
    sensor_list.append(depth)
    depth.listen(lambda data: process_img(data,"depth",True))


def gnss_sen(): #process
    gnss_bp = blueprint.find('sensor.other.gnss')
    spag = carla.Transform(carla.Location(x=50, y=0, z=145), carla.Rotation(pitch=-90, yaw=0, roll=0))
    gnss = world.spawn_actor(gnss_bp, spag, attach_to=vehicle)
    sensor_list.append(gnss)
    gnss.listen(lambda data: print(data))


def destroy_sensors():
    for actor in sensor_list:
        actor.destroy()
    return



#------------------------------------------------------------------------------------------
#                          LANE DETECTION(only straight lanes)
#------------------------------------------------------------------------------------------


def make_coordinates(image, line_parameters, right):
    try:
        slope, intercept = line_parameters
        #print(slope , intercept)
    except TypeError:
        if right == True:
            slope, intercept = 6.25,-3100#-2842
        else:
            slope, intercept = -6.25,1100
    #slope , intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(60/100))#int(y1*(3/5)) 
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2]) 


def extract_coordinate(image,line):
    x1,y1,x2,y2 = line 
    #cv2.circle(image, (x2, y2), 10, (0, 255, 0), -1)
    return np.array([x2, y2])

def avg_coordinate(img,left,right):
    xl,yl = left
    xr,yr = right
    x_avg = int((xl + xr)/2)
    y_avg = int((yl + yr)/2)
    #cv2.circle(img, (x_avg, y_avg), 10, (0, 0, 255), -1)
    return np.array([x_avg,y_avg,xl,xr])



def avg_slope_cal(image,lines,right):
    lane_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if right == True:
            if slope > 0:
                lane_fit.append((slope,intercept))
        else:
            if slope<0:
                lane_fit.append((slope,intercept))
    return lane_fit        


def average_slope_intercept(image,lines):
    if lines is not None:
        left_fit=avg_slope_cal(image,lines,False)
        left_fit_average = np.average(left_fit, axis=0)
    else:
        left_fit_average=correction(False)

    if lines is not None:
        right_fit=avg_slope_cal(image,lines,True)
        right_fit_average = np.average(right_fit, axis=0)    
    else:
        right_fit_average=correction(True)

    left_line , right_line = final_avg(image,right_fit_average,left_fit_average)
    return np.array([left_line,right_line])


def correction(right):
    if right==True:
        lane_fit_average = 6.25,-3100
    else:
        lane_fit_average = -6.25,1100
    return lane_fit_average 

def final_avg(image,right_fit_average,left_fit_average):
    right_line = make_coordinates(image, right_fit_average,True)
    right_line_begin = extract_coordinate(image,right_line)
    left_line = make_coordinates(image, left_fit_average,False)
    left_line_begin = extract_coordinate(image,left_line)
    line_coordinate_avg = avg_coordinate(image,left_line_begin,right_line_begin)
    drive(image,line_coordinate_avg)
    #adv_lanes(image,line_coordinate_avg)
    return np.array([left_line,right_line])




def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)   
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            #print(line)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0),10)
            #cv2.circle(line_image, (x2, y2), 5, (0, 0, 255), -1)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    left_upper_x = (37.5/100)*img_w # 225
    left_upper_y = (50/100)*img_h #300
    right_upper_x = (62.5/100)*img_w #375
    right_upper_y = (50/100)*img_h #300 
    polygons = np.array([[(left_upper_x, left_upper_y), (right_upper_x, right_upper_y), (img_w, img_h), (0, img_h)]], dtype=np.int32)  #([[(0,height),(img_w,height),(300,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask) 
    return masked_image  



def process_img_Lanes(image):
    i=np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    imag = i3 
    lane_image = np.copy(imag)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(lane_image,lines)
    line_image = display_lines(lane_image,averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    py_disp(combo_image,True,'lanes')
    return 


#------------------------------------------------------------------------------------------
#                                   MANUAL DRIVING
#------------------------------------------------------------------------------------------

def pyg_init():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((img_w, img_h))


def game():
    global vehicle
    r = True
    while r:        
        for event in pygame.event.get():
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                vehicle.apply_control(carla.VehicleControl(throttle=2.0, steer=0.0, reverse=False))
                lanes_data()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, reverse=True))
                lanes_data()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.6, reverse=False))
                lanes_data()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.6, reverse=False))
                lanes_data()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, reverse=False))

            if event.type == pygame.KEYDOWN and event.key == pygame.K_9:
                destroy_sensors()
                rgb_camera4()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                destroy_sensors()
                rgb_camera()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_2:
                destroy_sensors()
                driver_cam()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_3:
                destroy_sensors()
                lidar_cam()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_4:
                destroy_sensors()
                ss_cam()
                #rgb_camera4()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_5:
                destroy_sensors()
                rgb_camera2()
                #driver_cam()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_6:
                destroy_sensors()
                ss_cam2()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_7:
                gnss_sen()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_8:
                destroy_sensors()
                rgb_camera3()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                destroy_sensors()
                gods_eye()
                game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                r = False
            else:
                r = True
    return r




#------------------------------------------------------------------------------------------
#                                   AUTOMATIC DRIVING
#------------------------------------------------------------------------------------------
       

def drive(image,avg):
    x_avg,y_avg,xl,xr = avg
    x_set = img_w/2
    x_diff = x_set - x_avg
    tan_theta = x_diff/x_set
    theta = math.degrees(math.atan(tan_theta))
    st = -theta/90
    vehicle_control(0.5,st,theta)





def drive2(image,avg):
    x_avg,y_avg,xl,xr = avg
    print(avg)
    l_l = (41.6/100)*img_w #250
    r_l = (58.3/100)*img_w #350
    r_e = (87.5/100)*img_w #525
    l_e = (12.5/100)*img_w #75
    factor = (60/100)*img_w #360

    if(xl<img_w and xr<img_w):
        x_img = int((img_w)/2)
        if(x_avg<l_l):
           vehicle_control(0.3,-0.6)
        elif(x_avg>r_l):
           vehicle_control(0.3,0.6)
        else:
           vehicle_control(2,0)
    elif(xl>img_w):
        vehicle_control(0.5,-0.1)
    elif(xr>img_w):
        vehicle_control(0.5,0.1)
    elif(x_avg<img_w):
        if(x_avg in range(factor,r_e)):
            vehicle_control(0.3,0.6)
        elif(x_avg>r_e):
            vehicle_control(0.3,-0.6)
        elif(x_avg<l_e):
            vehicle_control(0.3,0.6)           
    else:
        x_avg2 = int(x_avg/2)
        avg2 = np.array([x_avg2,y_avg,xl,xr])
        drive(image,avg2)



def vehicle_control(th,st,theta):
    if theta >=85 or theta <=-85:
        vehicle.apply_control(carla.VehicleControl(throttle=1, steer=0))
    else:

    #debug = world.debug
    #current_w = map1.get_waypoint(vehicle.get_location())
    #next_w = map1.get_waypoint(vehicle.get_location())
        vehicle.apply_control(carla.VehicleControl(throttle=th, steer=st))
    #draw_waypoint_union(current_w, next_w, green, 60)
    #vector = vehicle.get_velocity()
    #debug.draw_string(current_w.transform.location, str('%15.0f km/h' % (3.6 * math.sqrt(vector.x**2 + vector.y**2 + vector.z**2))), False, orange, 60)
    return
    





#------------------------------------------------------------------------------------------
#                                POINT TO POINT TRAVEL
#------------------------------------------------------------------------------------------

    




#------------------------------------------------------------------------------------------
#                                           Map 
#------------------------------------------------------------------------------------------




def draw_transform(trans, col=carla.Color(255, 0, 0), lt=100):#-1):
    debug = world.debug
    yaw_in_rad = math.radians(trans.rotation.yaw)
    pitch_in_rad = math.radians(trans.rotation.pitch)
    p1 = carla.Location(
        x=trans.location.x + math.cos(pitch_in_rad) * math.cos(yaw_in_rad),
        y=trans.location.y + math.cos(pitch_in_rad) * math.sin(yaw_in_rad),
        z=trans.location.z + math.sin(pitch_in_rad))
    debug.draw_arrow(trans.location, p1, thickness=0.05, arrow_size=0.1, color=col, life_time=lt)
    


def draw_waypoint_union(w0, w1, color=carla.Color(255, 0, 0), color1=carla.Color(0,0,255), lt=5):
    debug = world.debug
    debug.draw_line(
        w0.transform.location + carla.Location(z=0.25),
        w1.transform.location + carla.Location(z=0.25),
        thickness=0.5, color=color, life_time=lt, persistent_lines=False)
    #debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.5, color1, lt, False)

def draw_point_waypoint(waypoint,waypoint_separation,color = blue, lt=5):
    debug = world.debug
    x=waypoint.transform.location.x
    y=waypoint.transform.location.y
    xw = x+(2*waypoint_separation)
    wx = x-(2*waypoint_separation)
    yw = y+(2*waypoint_separation)
    wy = y-(2*waypoint_separation)
    loc0 = carla.Location(x=x,y=y,z=0.5)
    loc1 = carla.Location(x=xw,y=yw,z=0.5)
    loc2 = carla.Location(x=wx,y=yw,z=0.5)
    loc3 = carla.Location(x=xw,y=wy,z=0.5)
    loc4 = carla.Location(x=wx,y=wy,z=0.5)

    debug.draw_point(loc0 + carla.Location(z=0.25), 0.5, color, lt, False)
    debug.draw_point(loc1 + carla.Location(z=0.25), 0.5, color, lt, False)
    debug.draw_point(loc2 + carla.Location(z=0.25), 0.5, color, lt, False)
    debug.draw_point(loc3 + carla.Location(z=0.25), 0.5, color, lt, False)
    debug.draw_point(loc4 + carla.Location(z=0.25), 0.5, color, lt, False)
    return xw,wx,yw,wy


def draw_waypoint_info(w, lt=5):
    debug = world.debug
    w_loc = w.transform.location
    debug.draw_string(w_loc + carla.Location(z=0.5), "lane: " + str(w.lane_id), False, yellow, lt)
    debug.draw_string(w_loc + carla.Location(z=1.0), "road: " + str(w.road_id), False, blue, lt)
    debug.draw_string(w_loc + carla.Location(z=-.5), str(w.lane_change), False, red, lt)



def adv_lanes(image,avg):
    global vehicle
    waypoint = map1.get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    lane_type = waypoint.lane_type
    left_lm_type = waypoint.left_lane_marking.type
    right_lm_type = waypoint.right_lane_marking.type
    lane_change = waypoint.lane_change
    
    waypoint_separation = 1
    dao = GlobalRoutePlannerDAO(map1, waypoint_separation)
    grp = GlobalRoutePlanner(dao)
    grp.setup()


    b = carla.Location(x=85, y=-150, z=1.8431)#x=96.0,y=4.45,z=0)

    #next_w = list(waypoint_list.next(waypoint_separation))

    a = carla.Location(x=-85, y=145, z=1.8431)#x=215.0,y=6.23,z=0)

    #lists of waypoints
    w = grp.trace_route(a, b)
    w2 = np.array(w)
    waypoint_list = w2[:,0]
    road_list = w2[:,1]
    location_list = []
    len_waypoint_list = len(waypoint_list)
    len_road_list = len(road_list)
    len_waypoint_list -= 1
    len_road_list -=  1
    
    '''current_l = vehicle.get_location()
    current_lx = current_l.x
    current_ly = current_l.y
    current_l = carla.Location(x=current_lx,y=current_ly, z=0)
    current_t = carla.Transform(current_l)
    current_w = carla.Waypoint.transform(current_t)'''
    way = waypoint_list[0]
    wayl = way.transform.location
    wayr = way.transform.rotation
    wayt = carla.Transform(wayl,wayr)
    #wayw = carla.Waypoint.transform(wayt)
    #print(way,wayt)
    #print(current_l)
    i = 1
    for i in range(len_waypoint_list):
        location_list.append(waypoint_list[i].transform.location)
    #print(location_list)
    for i in range(len_waypoint_list):
        draw_waypoint_union(waypoint_list[i],waypoint_list[i+1], lt = 30)
    
    #draw_point_waypoint(waypoint_list[5],waypoint_separation)
    '''r = True
    while r:        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                if current_l in location_list:
                    vehicle_control(1,0)
                else:
                    vehicle_control(1,-0.3)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                r = False'''



#------------------------------------------------------------------------------------------
#                                     MAIN FUNCTION
#------------------------------------------------------------------------------------------

def main():
    pyg_init()
    game()
    #adv_lanes()
    #main1(10.7974, -196.5, -85, 145)
    pygame.quit()
    return

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------


main()
vehicle.destroy()
destroy_sensors()