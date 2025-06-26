# -*- coding: utf-8 -*-
import logging
import sys
import time
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

import getopt
import alsaaudio # for recording using WM8960

from datetime import datetime # get current time

import wave # conversion to .wav

import csv # to output log

import numpy as np
from scipy import signal  # plot spectogram
from scipy.io import wavfile
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d # for 3d plot

# Globals:
FS = 44100 # recording frequency in Hz, change if needed
f_path = "" # to access path outside thread function
fly_on = 0 # keep recording until cf stops flying

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

DEFAULT_HEIGHT = 0.5
BOX_LIMIT = 0.5

deck_attached_event = threading.Event()

logging.basicConfig(level=logging.ERROR)

mydict = [] #saves data for csv file

start_recording = 0

recording_duration = 32 # for recording linear- set 32, for idle- set 50

task = 'init'

def move_linear_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        global task
        task = 'sleep'
        time.sleep(1)
        task = 'forward'
        mc.forward(0.8)
        task = 'sleep'
        time.sleep(1)
        task = 'turn'
        mc.turn_left(180)
        task = 'sleep'
        time.sleep(1)
        task = 'back'
        mc.forward(0.8)
        task = 'sleep'
        time.sleep(1)

def idle_fly(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        global task
        task = 'sleep'
        time.sleep(6)

def log_pos_callback(timestamp, data, logconf):
    
    global start_recording
    if start_recording == 0:
        start_recording = datetime.now()
    now_time = datetime.now()
    time_diff = now_time - start_recording
    
    global mydict

    mydict.append({'x': data['stateEstimate.x'], 'y': data['stateEstimate.y'], 'z': data['stateEstimate.z'],
                   'time': time_diff.total_seconds(), 'supervisor': data['supervisor.info'],'task': task})


def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

def fly_cf():
    cflib.crtp.init_drivers()
    
    global fly_on
    fly_on = 1

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
	
        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name='Position', period_in_ms=10)
        # if we get loco deck- can switch to: logconf.add_variable('locSrv.x', 'float'), logconf.add_variable('locSrv.y', 'float')
        logconf.add_variable('stateEstimate.x', 'float') 
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')
        logconf.add_variable('supervisor.info', 'uint16_t') #added
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        logconf.start()

        #idle_fly(scf)
        move_linear_simple(scf)

        logconf.stop()
        
def record_cf():  
    '''
    # In order to save the exact start time for each recording, 
	# the files (.raw, .wav, .png) use the recording start time as file name!
	# raw- the alsaaudio.PCM command records and saves the file type as raw
	# wav- convert the file to .wav for later use
	# png- image of the spectogram plot, generated from the .wav file
	'''
	# saves .wav file name as current date and time
    f = wave.open(f_path + '.wav', 'wb')  
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(FS)
    
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, channels=1, rate=FS, format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=1024, device='default')
    
    while fly_on == 0:
        pass
    
    start_time = time.time()

    while time.time() - start_time < recording_duration:    
    #while fly_on:
        l, data = inp.read() # Read data from device
        if l < 0:
            print("Capture buffer overrun! Continuing nonetheless ...")
        elif l:
            f.writeframes(data)
        
    f.close()
	

if __name__ == '__main__':

    # format of time: day.month-hour:minute:seconds.milliseconds (change if needed)
	#[:-3] to only save milliseconds, remove brackets to save in microseconds
    curr_time = datetime.now().strftime("%d.%m-%H-%M-%S.%f")[:-3] 
	
	# currently saves at the following directory, change if needed
    f_path = '/home/crazyflie/CF/pyalsaaudio/recordings/' + curr_time

    # different flights locations:
    #f_path = '/home/crazyflie/CF/pyalsaaudio/recordings/no_fly/' + curr_time
    #f_path = '/home/crazyflie/CF/pyalsaaudio/recordings/noise_no_fly/' + curr_time
    #f_path = '/home/crazyflie/CF/pyalsaaudio/recordings/no_move/' + curr_time
    #f_path = '/home/crazyflie/CF/pyalsaaudio/recordings/noise_no_move/' + curr_time
    #f_path = '/home/crazyflie/CF/pyalsaaudio/recordings/move/' + curr_time
    #f_path = '/home/crazyflie/CF/pyalsaaudio/recordings/noise_move/' + curr_time

    t_fly = threading.Thread(target=fly_cf)
    t_record = threading.Thread(target=record_cf)
    
    t_record.start()
    t_fly.start()
    
    t_fly.join()
    fly_on = 0 #stopped flying, stop recording!
    t_record.join()
    
    #log cf fly into csv:
    fields = ['x', 'y', 'z', 'time', 'supervisor','task']
    
    with open(f_path + '.csv', 'w', newline='') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)
     
        # writing data rows
        writer.writerows(mydict)
