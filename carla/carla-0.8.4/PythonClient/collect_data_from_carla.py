#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

# starting point selection: http://carla.org/2018/04/05/release-0.8.1/``
# weather selction: https://carla.readthedocs.io/en/0.8.4/carla_settings/#weather-presets

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

	W            : throttle
	S            : brake
	AD           : steer
	Q            : toggle reverse
	Space        : hand-brake
	P            : toggle autopilot

	R            : restart level

STARTING in a moment...
"""
from __future__ import print_function

import argparse
import logging
import random
import time
import re
from carla.transform import Transform
import numpy as np
import copy
from config import cfg

import struct
import os

def rotationMatrixToEulerAngles(R): 
    assert(isRotationMatrix(R))     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])     
    singular = sy < 1e-6 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0 
    return np.array([x, y, z])

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

import math
try:
	import pygame
	from pygame.locals import K_DOWN
	from pygame.locals import K_LEFT
	from pygame.locals import K_RIGHT
	from pygame.locals import K_SPACE
	from pygame.locals import K_UP
	from pygame.locals import K_a
	from pygame.locals import K_d
	from pygame.locals import K_p
	from pygame.locals import K_q
	from pygame.locals import K_r
	from pygame.locals import K_s
	from pygame.locals import K_w
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from transforms3d import euler

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

####################################################################3
number_of_episodes = cfg.carla_task_dick["NUM_EPISODE"]
frame_max = cfg.carla_task_dick["FRAME_OF_EPISODE"]
frame_start = cfg.carla_task_dick["FRAME_START_OF_EPISODE"]
frame_step = cfg.carla_task_dick["FRAME_STEP"]
epoch = cfg.carla_task_dick["EPOCH"]
print(epoch)

# str_sensor = ['velo_top', 'velo_left', 'velo_right']
str_sensor = ['velo_top']
tf_sensor_left_hand = [
	cfg.carla_sensor_dict["VELO_TOP"],
	cfg.carla_sensor_dict["VELO_LEFT"],
	cfg.carla_sensor_dict["VELO_RIGHT"]
]

Tf_sensor_right_hand = [[], [], []]
temp_tf = np.eye(4, 4)
temp_tf[:3, 3] = np.array([tf_sensor_left_hand[0][0], -tf_sensor_left_hand[0][1], tf_sensor_left_hand[0][2]])
temp_tf[:3,:3] = euler.euler2mat(0, 0, 0)
Tf_sensor_right_hand[0] = temp_tf
temp_tf = np.eye(4, 4)
temp_tf[:3, 3] = np.array([tf_sensor_left_hand[1][0], -tf_sensor_left_hand[1][1], tf_sensor_left_hand[1][2]])
temp_tf[:3,:3] = euler.euler2mat(0, 0, 0)
Tf_sensor_right_hand[1] = temp_tf
temp_tf = np.eye(4, 4)
temp_tf[:3, 3] = np.array([tf_sensor_left_hand[2][0], -tf_sensor_left_hand[2][1], tf_sensor_left_hand[2][2]])
temp_tf[:3,:3] = euler.euler2mat(0, 0, 0)
Tf_sensor_right_hand[2] = temp_tf
####################################################################

for isensor in range(len(str_sensor)):
	print('Sensor type: ', str_sensor[isensor])
	print('Sensor tf (right hand): \n', Tf_sensor_right_hand[isensor])

class Carla_bridge_sensor(object):
	def __init__(self):
		pass
	def set(self, bag_file):
		self._bag = rosbag.Bag(bag_file, 'w')

def make_carla_settings(args):
	"""Make a CarlaSettings object with the settings we need."""
	settings = CarlaSettings()
	if args.with_ped:
		num_of_ped = 300
	else:
		num_of_ped = 0
	settings.set(
		SynchronousMode=False,
		SendNonPlayerAgentsInfo=True,
		NumberOfVehicles=300, # 15
		NumberOfPedestrians=num_of_ped, # 30
		WeatherId=random.choice([1, 3, 7, 8, 14]),
		DisableTwoWheeledVehicles=True,
		QualityLevel=args.quality_level)
	settings.randomize_seeds()

	camera0 = sensor.Camera('image_2')
	camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
	camera0.set_position(2.0, 0.0, 1.4)
	camera0.set_rotation(0.0, 0.0, 0.0)
	settings.add_sensor(camera0)
	# print('camera intrinsics: ', camera0.K)

	camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
	camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
	camera1.set_position(2.0, 0.0, 1.4)
	camera1.set_rotation(0.0, 0.0, 0.0)
	settings.add_sensor(camera1)

	camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
	camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
	camera2.set_position(2.0, 0.0, 1.4)
	camera2.set_rotation(0.0, 0.0, 0.0)
	settings.add_sensor(camera2)

	if args.lidar:
		velo = sensor.Lidar(str_sensor[0])
		x, y, z, roll, pitch, yaw = tf_sensor_left_hand[0]
		velo.set_position(x=x, y=y, z=z) # x y z
		velo.set_rotation(pitch=-pitch/math.pi*180, yaw=-yaw/math.pi*180, roll=-roll/math.pi*180) # pitch yaw roll (degree)
		velo.set(
			Channels=64,
			Range=100,
			PointsPerSecond=250000,
			RotationFrequency=30,
			UpperFovLimit=2,
			LowerFovLimit=-24.8)
		settings.add_sensor(velo)
		# print(velo.get_transform())

		# velo = sensor.Lidar(str_sensor[1])
		# x, y, z, roll, pitch, yaw = tf_sensor_left_hand[1]
		# velo.set_position(x=x, y=y, z=z) # x y z
		# velo.set_rotation(pitch=-pitch/math.pi*180, yaw=-yaw/math.pi*180, roll=-roll/math.pi*180) # pitch yaw roll
		# velo.set(
		# 	Channels=64,
		# 	Range=100,
		# 	PointsPerSecond=250000,
		# 	RotationFrequency=10,
		# 	UpperFovLimit=2,
		# 	LowerFovLimit=-24.8)
		# settings.add_sensor(velo)
		# # print(velo.get_transform())

		# velo = sensor.Lidar(str_sensor[2])
		# x, y, z, roll, pitch, yaw = tf_sensor_left_hand[2]
		# velo.set_position(x=x, y=y, z=z) # x y z
		# velo.set_rotation(pitch=-pitch/math.pi*180, yaw=-yaw/math.pi*180, roll=-roll/math.pi*180) # pitch yaw roll
		# velo.set(
		# 	Channels=64,
		# 	Range=100,
		# 	PointsPerSecond=250000,
		# 	RotationFrequency=10,
		# 	UpperFovLimit=2,
		# 	LowerFovLimit=-24.8)
		# settings.add_sensor(velo)
		# print(velo.get_transform())

	return settings

class Timer(object):
	def __init__(self):
		self.step = 0
		self._lap_step = 0
		self._lap_time = time.time()

	def tick(self):
		self.step += 1

	def lap(self):
		self._lap_step = self.step
		self._lap_time = time.time()

	def ticks_per_second(self):
		return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

	def elapsed_seconds_since_lap(self):
		return time.time() - self._lap_time

class CarlaGame(object):
	def __init__(self, carla_client, args):
		self.client = carla_client
		self._args = args
		self._carla_settings = make_carla_settings(args)
		self._timer = None
		self._display = None
		self._main_image = None
		self._mini_view_image1 = None
		self._mini_view_image2 = None
		self._enable_autopilot = args.autopilot
		self._save_images_to_disk = args.save_images_to_disk
		self._lidar_measurement = None
		self._map_view = None
		self._is_on_reverse = False
		self._city_name = args.map_name
		self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
		self._map_shape = self._map.map_image.shape if self._city_name is not None else None
		self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
		self._position = None
		self._agent_positions = None

	def execute(self):
		"""Launch the PyGame."""
		pygame.init()
		self._initialize_game()

		if self._save_images_to_disk:
			os.system('mkdir -p {}/episode_{:04d}'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch))
			os.system('mkdir -p {}/episode_{:04d}/calib'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch))
			os.system('mkdir -p {}/episode_{:04d}/label_imu'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch))
			os.system('mkdir -p {}/episode_{:04d}/image_2'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch))
			os.system('mkdir -p {}/episode_{:04d}/{}'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch, str_sensor[0]))
			# os.system('mkdir -p {}/episode_{:04d}/{}'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch, str_sensor[1]))
			# os.system('mkdir -p {}/episode_{:04d}/{}'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch, str_sensor[2]))
			os.system('mkdir -p {}/episode_{:04d}/velo_fused'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch))

			self._file_gt = open("{}/episode_{:04d}/tf_vehicle.txt".format(cfg.carla_task_dick["SAVE_FOLD"], epoch), 'w')
			str_output = '#frame_id timestamp x y z roll pitch yaw \n'
			self._file_gt.writelines(str_output)

		frame = (epoch-1)*frame_max
		try:
			while True:
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						return
				frame = frame + 1
				fs = False
				if frame - (epoch-1)*frame_max > frame_start:
					fs = True
				self._on_loop(frame, fs)
				self._on_render()
				if frame  >= epoch*frame_max:
					break
		finally:
			# carla_bridge_sensor._bag.close()
			self._file_gt.close()
			pygame.quit()

	def _initialize_game(self):
		if self._city_name is not None:
			self._display = pygame.display.set_mode(
				(WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
				pygame.HWSURFACE | pygame.DOUBLEBUF)
		else:
			self._display = pygame.display.set_mode(
				(WINDOW_WIDTH, WINDOW_HEIGHT),
				pygame.HWSURFACE | pygame.DOUBLEBUF)

		logging.debug('pygame started')
		self._on_new_episode()

	def _on_new_episode(self):
		self._carla_settings.randomize_seeds()
		self._carla_settings.randomize_weather()
		self._carla_settings.WeatherId = 1
		scene = self.client.load_settings(self._carla_settings)
		number_of_player_starts = len(scene.player_start_spots)
		player_start = np.random.randint(number_of_player_starts)
		print('Starting new episode...')
		self.client.start_episode(player_start) # default: 131
		print('player start: ', player_start)
		self._timer = Timer()
		self._is_on_reverse = False

	def _on_loop(self, frame, fs):
		self._timer.tick()
		measurements, sensor_data = self.client.read_data()
		world_transform = Transform(measurements.player_measurements.transform)
		current_timestamp = int(measurements.game_timestamp/100)

		self._main_image = sensor_data.get('CameraRGB', None)
		self._mini_view_image1 = sensor_data.get('CameraDepth', None)
		self._mini_view_image2 = sensor_data.get('CameraSemSeg', None)
		self._velo_top = sensor_data.get('velo_top', None)
		self._velo_left = sensor_data.get('velo_left', None)
		self._velo_right = sensor_data.get('velo_right', None)

		# Print measurements every second.
		if self._timer.elapsed_seconds_since_lap() > 1.0:
			if self._city_name is not None:
				# Function to get car position on map.
				map_position = self._map.convert_to_pixel([
					measurements.player_measurements.transform.location.x,
					measurements.player_measurements.transform.location.y,
					measurements.player_measurements.transform.location.z])
				# Function to get orientation of the road car is in.
				lane_orientation = self._map.get_lane_orientation([
					measurements.player_measurements.transform.location.x,
					measurements.player_measurements.transform.location.y,
					measurements.player_measurements.transform.location.z])

				self._print_player_measurements_map(
					measurements.player_measurements,
					map_position,
					lane_orientation)
			else:
				self._print_player_measurements(measurements.player_measurements)

			# Plot position on the map as well.

			self._timer.lap()

		control = self._get_keyboard_control(pygame.key.get_pressed())
		# Set the player position
		if self._city_name is not None:
			self._position = self._map.convert_to_pixel([
				measurements.player_measurements.transform.location.x,
				measurements.player_measurements.transform.location.y,
				measurements.player_measurements.transform.location.z])
			self._agent_positions = measurements.non_player_agents

		if control is None:
			self._on_new_episode()
		elif self._enable_autopilot:
			self.client.send_control(measurements.player_measurements.autopilot_control)
		else:
			self.client.send_control(control)

		## parse the vehicle and pedestrain location
		i_near_vehicle = 0
		d = []
		object_label_list = []
		tf_car_bbox_list = []
		tf_world_car = world_transform.matrix

		for agent in measurements.non_player_agents:
			if agent.HasField('vehicle'):
				# get vehicle transform in car
				vehicle_transform = Transform(agent.vehicle.transform)
				tf_world_vehicle = vehicle_transform.matrix
				tf_car_vehicle = np.matmul(np.linalg.inv(tf_world_car), tf_world_vehicle) # left hand coordinate

				# get boundingbox transform in car
				bbox_transform = Transform(agent.vehicle.bounding_box.transform)
				tf_vehicle_bbox = bbox_transform.matrix
				tf_car_bbox = np.matmul(tf_car_vehicle, tf_vehicle_bbox) # left hand coordinate
				tf_car_bbox_list = np.append(tf_car_bbox_list, tf_car_bbox)

				bbox_extent = agent.vehicle.bounding_box.extent

				if np.linalg.norm(np.array([tf_car_bbox[0,3], tf_car_bbox[1,3]]) > 100):
					continue

				d = 0.2
				object_label = []
				object_label.append('Car')
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(bbox_extent.z*2 + d)
				object_label.append(bbox_extent.y*2 + d)
				object_label.append(bbox_extent.x*2 + d)
				object_label.append(tf_car_bbox[0,3])
				object_label.append(-tf_car_bbox[1,3])
				object_label.append(tf_car_bbox[2,3]-bbox_extent.z)
				euler_angles = rotationMatrixToEulerAngles(tf_car_bbox[:3,:3])
				object_label.append(-euler_angles[2])
				object_label_list.append(object_label)

			elif agent.HasField('pedestrian'):
				# get pedestrian transform in pedestrian
				pedestrian_transform = Transform(agent.pedestrian.transform)
				tf_world_pedestrian = pedestrian_transform.matrix
				tf_car_pedestrian = np.matmul(np.linalg.inv(tf_world_car), tf_world_pedestrian) # left hand coordinate

				# get boundingbox transform in pedestrian
				bbox_transform = Transform(agent.pedestrian.bounding_box.transform)
				tf_pedestrian_bbox = bbox_transform.matrix
				tf_car_bbox = np.matmul(tf_car_pedestrian, tf_pedestrian_bbox) # left hand coordinate
				tf_car_bbox_list = np.append(tf_car_bbox_list, tf_car_bbox)

				bbox_extent = agent.pedestrian.bounding_box.extent
				if np.linalg.norm(np.array([tf_car_bbox[0,3], tf_car_bbox[1,3]]) > 100):
					continue

				object_label = []
				object_label.append('Pedestrian')
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(0)
				object_label.append(bbox_extent.z*2)
				object_label.append(bbox_extent.y*2)
				object_label.append(bbox_extent.x*2)
				object_label.append(tf_car_bbox[0,3])
				object_label.append(-tf_car_bbox[1,3])
				object_label.append(tf_car_bbox[2,3]-bbox_extent.z)
				euler_angles = rotationMatrixToEulerAngles(tf_car_bbox[:3,:3])
				object_label.append(-euler_angles[2])
				object_label_list.append(object_label)

		if (self._save_images_to_disk and fs):
			if self._timer.step % frame_step == 0:
				# save measurements
				# str_output = str(frame) + ' ' + str(current_timestamp) + ' ' + world_transform.print_xyzrpy()
				str_output = str(frame) + ' ' + str(current_timestamp)
				self._file_gt.writelines(str_output)
				# point_cloud_fused = np.zeros((0,3))
				for name, measurement in sensor_data.items():
					# save point clouds
					filename = "{}/episode_{:04d}/{}/{:06d}".format(cfg.carla_task_dick["SAVE_FOLD"], epoch, name, frame)
					if re.match('velo', name):
						point_cloud = copy.deepcopy(measurement.point_cloud)
						point_cloud_np = point_cloud._array
						# add gaussian noise on points
						# mu, sigma = 0, 0.05
						# s = np.random.normal(mu, sigma, 3)
						point_cloud_np_x = point_cloud_np[:, 0:1]
						point_cloud_np_y = point_cloud_np[:, 1:2]
						point_cloud_np_z = point_cloud_np[:, 2:3]
						# modify to coincide with CarlaData specified frame
						point_cloud_np = np.hstack([-point_cloud_np_y, -point_cloud_np_x, -point_cloud_np_z])
						np.save("{}.npy".format(filename), point_cloud_np)

						for isensor in range(len(str_sensor)):
							if re.match(str_sensor[isensor], name):
								temp_point_cloud = np.matmul(point_cloud_np, Tf_sensor_right_hand[isensor][:3,:3].transpose()) \
									+ Tf_sensor_right_hand[isensor][:3,3].transpose()
								# point_cloud_fused = np.vstack((point_cloud_fused, temp_point_cloud))

					# save images
					elif re.match('image_2', name):
						measurement.save_to_disk(filename)

				# filename = "{}/episode_{:04d}/{}/{:06d}".format(cfg.carla_task_dick["SAVE_FOLD"], epoch, 'velo_fused', frame)
				# np.save("{}.npy".format(filename), point_cloud_fused)

				# save label: all coordinate system should be changed to the right-hand coordinate
				fs_label = open('{}/episode_{:04d}/label_imu/{:06d}.txt'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch, frame), 'w')
				if len(object_label_list) > 0:
					for i in range(0, len(object_label_list)):
						object_label = object_label_list[i]
						str_output = ''
						for item in object_label:
							str_output = str_output + str(item) + ' '
						# label format: type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z rz score
						fs_label.writelines(str_output+'\n')
				fs_label.close()

				# Note: Modified by pyun (previously the following items is Tr_veloxx_to_imu)
				# right hand rule
				# according to KITTI, Tr_imu_to_velo_top means the translation from top to imu
				fs_calib = open('{}/episode_{:04d}/calib/{:06d}.txt'.format(cfg.carla_task_dick["SAVE_FOLD"], epoch, frame), 'w')
				from numpy.linalg import inv as inv
				fs_calib.writelines('Tr_imu_to_' + str_sensor[0] + ': ' + np.array2string(inv(Tf_sensor_right_hand[0]).reshape(16), max_line_width=1000)[1:-1] + '\n')
				# fs_calib.writelines('Tr_imu_to_' + str_sensor[1] + ': ' + np.array2string(inv(Tf_sensor_right_hand[1]).reshape(16), max_line_width=1000)[1:-1] + '\n')
				# fs_calib.writelines('Tr_imu_to_' + str_sensor[2] + ': ' + np.array2string(inv(Tf_sensor_right_hand[2]).reshape(16), max_line_width=1000)[1:-1] + '\n')
				fs_calib.writelines('Tr_imu_to_' + 'velo_fused' + ': ' + np.array2string(np.eye(4, 4).reshape(16), max_line_width=1000)[1:-1] + '\n')
				fs_calib.close()

	def _get_keyboard_control(self, keys):
		"""
		Return a VehicleControl message based on the pressed keys. Return None
		if a new episode was requested.
		"""
		if keys[K_r]:
			return None
		control = VehicleControl()
		if keys[K_LEFT] or keys[K_a]:
			control.steer = -1.0
		if keys[K_RIGHT] or keys[K_d]:
			control.steer = 1.0
		if keys[K_UP] or keys[K_w]:
			control.throttle = 1.0
		if keys[K_DOWN] or keys[K_s]:
			control.brake = 1.0
		if keys[K_SPACE]:
			control.hand_brake = True
		if keys[K_q]:
			self._is_on_reverse = not self._is_on_reverse
		if keys[K_p]:
			self._enable_autopilot = not self._enable_autopilot
		control.reverse = self._is_on_reverse
		return control

	def _print_player_measurements_map(
			self,
			player_measurements,
			map_position,
			lane_orientation):
		message = 'Step {step} ({fps:.1f} FPS): '
		message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
		message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
		message += '{speed:.2f} km/h, '
		message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
		message = message.format(
			map_x=map_position[0],
			map_y=map_position[1],
			ori_x=lane_orientation[0],
			ori_y=lane_orientation[1],
			step=self._timer.step,
			fps=self._timer.ticks_per_second(),
			speed=player_measurements.forward_speed * 3.6,
			other_lane=100 * player_measurements.intersection_otherlane,
			offroad=100 * player_measurements.intersection_offroad)
		print_over_same_line(message)

	def _print_player_measurements(self, player_measurements):
		message = 'Step {step} ({fps:.1f} FPS): '
		message += '{speed:.2f} km/h, '
		message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
		message = message.format(
			step=self._timer.step,
			fps=self._timer.ticks_per_second(),
			speed=player_measurements.forward_speed * 3.6,
			other_lane=100 * player_measurements.intersection_otherlane,
			offroad=100 * player_measurements.intersection_offroad)
		print_over_same_line(message)

	def _on_render(self):
		gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
		mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

		if self._main_image is not None:
			array = image_converter.to_rgb_array(self._main_image)
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			self._display.blit(surface, (0, 0))

		# if self._mini_view_image1 is not None:
		# 	array = image_converter.depth_to_logarithmic_grayscale(self._mini_view_image1)
		# 	surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
		# 	self._display.blit(surface, (gap_x, mini_image_y))

		if self._mini_view_image2 is not None:
			array = image_converter.labels_to_cityscapes_palette(
				self._mini_view_image2)
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

			self._display.blit(
				surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

		if self._velo_top is not None:
			lidar_data = np.array(self._velo_top.data[:, :2])
			lidar_data *= 2.0
			lidar_data += 100.0
			lidar_data = np.fabs(lidar_data)
			lidar_data = lidar_data.astype(np.int32)
			lidar_data = np.reshape(lidar_data, (-1, 2))
			#draw lidar
			lidar_img_size = (300, 300, 3)
			lidar_img = np.zeros(lidar_img_size)
			lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
			surface = pygame.surfarray.make_surface(lidar_img)
			self._display.blit(surface, (10, 10))

		if self._velo_left is not None:
		    lidar_data = np.array(self._velo_left.data[:, :2])
		    lidar_data *= 2.0
		    lidar_data += 100.0
		    lidar_data = np.fabs(lidar_data)
		    lidar_data = lidar_data.astype(np.int32)
		    lidar_data = np.reshape(lidar_data, (-1, 2))
		    #draw lidar
		    lidar_img_size = (300, 300, 3)
		    lidar_img = np.zeros(lidar_img_size)
		    lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
		    surface = pygame.surfarray.make_surface(lidar_img)
		    self._display.blit(surface, (200, 10))

		# if self._velo_right is not None:
		#     lidar_data = np.array(self._velo_right.data[:, :2])
		#     lidar_data *= 2.0
		#     lidar_data += 100.0
		#     lidar_data = np.fabs(lidar_data)
		#     lidar_data = lidar_data.astype(np.int32)
		#     lidar_data = np.reshape(lidar_data, (-1, 2))
		#     #draw lidar
		#     lidar_img_size = (200, 200, 3)
		#     lidar_img = np.zeros(lidar_img_size)
		#     lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
		#     surface = pygame.surfarray.make_surface(lidar_img)
		#     self._display.blit(surface, (400, 10))

		if self._map_view is not None:
			array = self._map_view
			array = array[:, :, :3]

			new_window_width = \
				(float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
				float(self._map_shape[1])
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

			w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
			h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

			pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
			for agent in self._agent_positions:
				if agent.HasField('vehicle'):
					agent_position = self._map.convert_to_pixel([
						agent.vehicle.transform.location.x,
						agent.vehicle.transform.location.y,
						agent.vehicle.transform.location.z])

					w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
					h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

					pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

			self._display.blit(surface, (WINDOW_WIDTH, 0))

		pygame.display.flip()

def main():
	argparser = argparse.ArgumentParser(
		description='CARLA Manual Control Client')
	argparser.add_argument(
		'-v', '--verbose',
		action='store_true',
		dest='debug',
		help='print debug information')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='localhost',
		help='IP of the host server (default: localhost)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'-a', '--autopilot',
		action='store_true',
		help='enable autopilot')
	argparser.add_argument(
		'-l', '--lidar',
		action='store_true',
		help='enable Lidar')
	argparser.add_argument(
		'-q', '--quality-level',
		choices=['Low', 'Epic'],
		type=lambda s: s.title(),
		default='Epic',
		help='graphics quality level, a lower level makes the simulation run considerably faster.')
	argparser.add_argument(
		'-m', '--map-name',
		metavar='M',
		default=None,
		help='plot the map of the current city (needs to match active map in '
			 'server, options: Town01 or Town02)')
	argparser.add_argument(
		'-i', '--images-to-disk',
		action='store_true',
		dest='save_images_to_disk',
		help='save images (and Lidar data if active) to disk')
	argparser.add_argument(
		'--epoch',
		metavar='e',
		default=1,
		type=int,
		help='epoch')
	argparser.add_argument(
		'--with-ped',
		action='store_true',
		help='enable pedestrians')
	args = argparser.parse_args()

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

	logging.info('listening to server %s:%s', args.host, args.port)

	args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'
	args.out_gt_format = '_out/episode_{:0>4d}/{:s}'

	# print(__doc__)
	episodes = args.epoch
	number_of_episodes = args.epoch
	while True:
		if episodes > number_of_episodes:
			break
		try:
			with make_carla_client(args.host, args.port) as client:
				print('EPOCH: ', episodes)
				global epoch
				epoch = episodes
				game = CarlaGame(client, args)
				game.execute()
				episodes = episodes + 1

		except TCPConnectionError as error:
			logging.error(error)
			time.sleep(1)

if __name__ == '__main__':

	try:
		main()
	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')