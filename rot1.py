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
#import warnings
#warnings.filterwarnings("error")
#np.warnings.filterwarnings('ignore')
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

client = carla.Client('localhost', 2000)
client.set_timeout(5.0) # seconds
#world = client.load_world('Town05')
world = client.get_world()
map1 = world.get_map() 

start_loc = carla.Location(x=-85, y=145, z=1.8)#x=240, y=130, z=1.8)#x=-85, y=145, z=1.8)#x=215.0,y=6.23,z=0)
end_loc = carla.Location(x=96, y=4.45, z=1.8)#carla.Location(x=10.7974, y=-196.5, z=1.8)#carla.Location(x=-85, y=160, z=1.8)#x=96.0,y=4.45,z=0)

blueprint = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint.filter('vehicle.tesla.model3'))
spawn_point = carla.Transform(start_loc, carla.Rotation(pitch=0, yaw=90, roll=0))#-85 145 1 
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
time.sleep(2)

location_list = []
location_list_x = []
location_list_y = []
listxy = [[]]
disp_sensor_list = []
sensor_list = []
og_image = []

global count,pix_var    
count = 0
pix_var = 0

img_h = 800 #600
img_w = 800 #600

def main():
    
    waypoint_separation = 1
    dao = GlobalRoutePlannerDAO(map1, waypoint_separation)
    grp = GlobalRoutePlanner(dao)
    grp.setup()


    #lists of waypoints
    w = grp.trace_route(start_loc, end_loc)
    #print(w)
    w2 = np.array(w)
    #print(w2)
    waypoint_list = w2[:,0]
    #print(waypoint_list[0])
    len_waypoint_list = len(waypoint_list)
    len_waypoint_list -= 1

    for q in range(len_waypoint_list):
        draw_waypoint_union(waypoint_list[q],waypoint_list[q+1], lt = 90)

    i = 1
    for i in range(len_waypoint_list):
        location_list.append(waypoint_list[i].transform.location)
        location_list_y.append(location_list[i].y)
        location_list_x.append(location_list[i].x)
    listxy = np.column_stack((location_list_x,location_list_y))
        
    
    i=2
    limit = len(listxy)-3
    a = 1
    while i<limit:
        i=algo_fn_2(i,listxy,a)

    time.sleep(2)
    vehicle.destroy()



    


def algo_fn_2(i,listxy,a):
    loc=vehicle.get_location()
    vehicle_loc = np.column_stack((loc.x,loc.y))
    vehicle_pt_dist = calculateDistance(vehicle_loc[0,0],vehicle_loc[0,1],listxy[i,0],listxy[i,1])
    if vehicle_pt_dist <= 1:
        i = i+1
        #st=cal_steer(i,listxy,vehicle_loc)
        st,th = cal_radius_coordinates(i,a,listxy,vehicle_loc)
        vehicle_control2(th,st)
    else:
        #st=cal_steer(i,listxy,vehicle_loc)
        st,th = cal_radius_coordinates(i,a,listxy,vehicle_loc)
        vehicle_control2(th,st)
    return i


def cal_radius_coordinates(i,a,listxy,vehicle_loc):
    x1,y1 = vehicle_loc[0]
    x2,y2 = listxy[i+1]
    x3,y3 = listxy[i+a+1]

    a1 = x1-x2
    a11 = x1+x2
    b1 = y2-y1
    b11 = y2+y1

    a2 = x1-x3
    a21 = x1+x3
    b2 = y3-y1
    b21 = y3+y1
    try:
        m1 = a1/b1
        m2 = a2/b2
        c1 = ((b1*b11)-(a1*a11))/(2*b1)
        c2 = ((b2*b21)-(a2*a21))/(2*b2)
    
        h = (c2-c1)/(m1-m2)
        k = (m1*h)+c1

        #draw_point(x2,y2)
        #draw_point(x3,y3)
        #draw_point(h,k)

        #r = calculateDistance(h,k,x1,y1)
        A = vehicle_loc[0]#[x1,y1]
        B = [h,k]
        C = listxy[i+1]
        ba = A-B
        bc = C-B
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        alpha = np.degrees(angle)
        theta = (alpha/2)
        #print(theta)
        #st = -theta/(90)
        #if h > x1:
        #    theta = -theta
        if a==1:
            if theta>45 or theta<(-45):
                st = (-theta)/90
                th = (np.cos(theta * np.pi/180))/2
                print('90')
            else:    
                st = (-theta+23)/400
                th = 0.5#(np.cos(theta * np.pi/180))/2#0.5
                print('normal')
        else:
            if theta>15 or theta<(-15):
                st = (-theta)/90
                th = (np.cos(theta * np.pi/180))/2
                print('90111')
            else:
                st = (-theta)/180
                th = 0.5#(np.cos(theta * np.pi/180))/2
                print('180')
        #if i>58:
         #   print(theta,st)
    except RuntimeWarning:
        #print('RuntimeWarning')
        st,th = cal_radius_coordinates(i,a+1,listxy,vehicle_loc)
        '''if i<542:
    #except RuntimeWarning:
            st = -0.1
            print('st = -0.1')
        else:#if i>542 and i<551:
            print('c1' + str(c1))
            print('c2' + str(c2))
            print('m1' + str(m1))
            print('m2' + str(m2))
            st = 0.1
            print('st = 0.1')'''
    return st,th

def bound_box(i,listxy,vehicle_loc):
    j=0.1
    x,y = vehicle_loc[0]
    box_max = x+j
    box_min = x-j
    boy_max = y+j
    boy_min = y-j
    if listxy[i,0]<=box_max and listxy[i,0]>=box_min and listxy[i,1]<=boy_max and listxy[i,1]>=boy_min:
        a = True
    else:
        a = False
    return a


def vehicle_control2(th,st):
    vehicle.apply_control(carla.VehicleControl(throttle=th, steer=st))              
    

def calculateDistance(x1,y1,x2,y2):  
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist


def draw_waypoint_union(w0, w1, color=carla.Color(255, 0, 0), color1=carla.Color(0,0,255), lt=5):
    debug = world.debug
    debug.draw_line(
        w0.transform.location + carla.Location(z=0.25),
        w1.transform.location + carla.Location(z=0.25),
        thickness=0.5, color=color, life_time=lt, persistent_lines=False)

def draw_point(x,y,color=carla.Color(0, 255, 0),lt=0.25):
    debug = world.debug
    loc0 = carla.Location(x=x,y=y,z=0.5)
    debug.draw_point(loc0 + carla.Location(z=0.25), 0.5, color, lt, False)

#---------------------------------------------------------------------------------------------
#                                      Display Cameras
#---------------------------------------------------------------------------------------------


def rgb_camera():  #display
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,-4,0,3.5,-10,0,0)#0.1,-0.3,1.1,0,0,0)
    rgb_cam = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    rgb_cam.listen(lambda data: process_img(data))
    disp_sensor_list.append(rgb_cam)
    rgb_camera2()
    return 

def driver_cam(): #display
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,0.1,-0.3,1.1,0,0,0)
    driver_cam = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    driver_cam.listen(lambda data: process_img(data))
    disp_sensor_list.append(driver_cam)
    rgb_camera2()
    return 

def process_img(image):
    i=np.array(image.raw_data)
    i2=i.reshape((img_h,img_w,4))
    i3=i2[:, :, :3]
    py_disp(i3)
    return

def attribute(camera_bp,X,Y,Z,p,ya,r):
    camera_bp.set_attribute('image_size_x', f'{img_w}')
    camera_bp.set_attribute('image_size_y', f'{img_h}')
    camera_bp.set_attribute('fov', '110')
    relative_spawn = carla.Transform(carla.Location(x=X, y=Y, z=Z), carla.Rotation(pitch=p, yaw=ya, roll=r))
    return relative_spawn


#-------------------------------------------------------------------------
#                                   Lanes
#-------------------------------------------------------------------------


def rgb_camera2(): #process
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,2.5,0,1,-7,0,0)#0.1,-0.3,1.1,0,0,0)
    rgb_cam2 = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    rgb_cam2.listen(lambda data: process_img_Lanes(data))
    sensor_list.append(rgb_cam2)
    return 


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
    line_image = display_lines(lane_image,averaged_lines,10)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    #py_disp(combo_image)
    return

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)   
    return canny

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

def drive(image,avg):
    x_avg,y_avg,xl,xr = avg
    x_set = img_w/2
    x_diff = x_set - x_avg
    tan_theta = x_diff/x_set
    theta = math.degrees(math.atan(tan_theta))
    st = -theta/90
    vehicle_control(0.5,st,theta)

def vehicle_control(th,st,theta):
    global pix_var
    #print(pix_var)
    if (pix_var in range(600,1022)) or (pix_var in range(4000,5000)):#4260
        vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
    else:    
        if theta >=85 or theta <=-85:
            vehicle.apply_control(carla.VehicleControl(throttle=1, steer=0))
        else:
            vehicle.apply_control(carla.VehicleControl(throttle=th, steer=st))

def display_lines(image,lines,thickness):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            #print(line)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0),thickness)
            #cv2.circle(line_image, (x2, y2), 5, (0, 0, 255), -1)
    return line_image



#---------------------------------------------------------------------------------------------
#                                           Traffic Lights
#---------------------------------------------------------------------------------------------
def ss_cam(): #display / process
    sem_bp = blueprint.find('sensor.camera.semantic_segmentation')
    spas=attribute(sem_bp,2.5,0.0,1.7,0,0,0)
    sem = world.spawn_actor(sem_bp, spas, attach_to=vehicle)
    sensor_list.append(sem)
    sem.listen(lambda data: process_lights(data,True))
    rgb_camera2()

def process_lights(image,py):
    global pix_var
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
    frame = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    left_upper_x = 350#(30/100)*img_w # 250
    left_upper_y = 300#(20/100)*img_h #300
    left_lower_x = 350#(30/100)*img_w # 250
    left_lower_y = 380#(40/100)*img_h #300
    right_upper_x = 450#(70/100)*img_w #350
    right_upper_y = 300#(20/100)*img_h #300
    right_lower_x = 450#(70/100)*img_w #350
    right_lower_y = 380#(40/100)*img_h #300     
    pts1 = np.float32([[left_upper_x, left_upper_y], [right_upper_x, right_upper_y], [right_lower_x, right_lower_y], [left_lower_x, left_lower_y]])
    pts2 = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (img_w, img_h))
    cop = result
    traffic_image = np.copy(cop)
    pix_var = count_of_pixel(traffic_image)
    canny_image = canny(y)
    lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 10, np.array([]),minLineLength=0.1, maxLineGap=5)
    line_image = display_lines(og_image[0],lines,1)
    masked_image = cv2.bitwise_and(og_image[0],y) 
    #combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    py_disp(traffic_image)
    return

def count_of_pixel(image):
    n_white_pix = np.sum(image == 255)
    print(n_white_pix)
    return n_white_pix

def rgb_camera4(): #process
    camera_bp =  blueprint.find('sensor.camera.rgb')
    spa=attribute(camera_bp,2.5,0.0,1.7,0,0,0)
    rgb_cam4 = world.spawn_actor(camera_bp, spa, attach_to=vehicle)
    rgb_cam4.listen(lambda data: on_going(data))
    sensor_list.append(rgb_cam4)
    return

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

#---------------------------------------------------------------------------------------------
#                                           Pygame 
#---------------------------------------------------------------------------------------------

def pyg_init():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((img_w, img_h))

def py_disp(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = frame.swapaxes(0,1)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))
    pygame.display.update()
    return

#---------------------------------------------------------------------------------------------
#                                         game loop
#---------------------------------------------------------------------------------------------


def game():
    r = True
    while r:        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                destroy_sensors(disp_sensor_list)
                rgb_camera()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_2:
                destroy_sensors(disp_sensor_list)
                driver_cam()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_4:
                destroy_sensors(disp_sensor_list)
                ss_cam()
                rgb_camera4()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                destroy_sensors(disp_sensor_list)
                destroy_sensors(sensor_list)
                r = False
            else:
                r = True
    return 


def destroy_sensors(destroy_list):
    for actor in destroy_list:
        actor.destroy()
    return



pyg_init()
game()
#main()
vehicle.destroy()