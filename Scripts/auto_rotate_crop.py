#!/usr/bin/env python

# Python dependancy in ROS 
import rospy

# Importing math library to perform trigonometric operations
from math import *

# Import numpy package
import numpy as np

# The Pillow package for performing image operations
from PIL import Image

# The opencv library
import cv2

# The odometry command message
from nav_msgs.msg import Odometry, MapMetaData

# The laser scan message
from sensor_msgs.msg import LaserScan

# Package to import the function that converts quaternion to euler angles
from tf.transformations import euler_from_quaternion

# Package to import the messages about the presence of a door on the map
from std_msgs.msg import String

# Package to import the custom crop and rotate function
from auto_crop import crop_rotated_rectangle, inside_rect

# Initializing a class to make instances of the robot
class Autonomy():

	def __init__(self):
		
		#Subscribing to the map message to know the meta details of the map
		self.map_sub = rospy.Subscriber('/map_metadata', MapMetaData, self.map_callback)

		# Publishing the velocity message of the robot
		self.door_sub = rospy.Subscriber('/door_presence', String, self.door_callback)
		
		# Subscribing to the odometry message to know the odom of the robot
		self.odom_sub = rospy.Subscriber('/base_pose_ground_truth', Odometry, self.odom_callback)

		# Subscribing to the laser scan message
		self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

		# Define a counter to keep track of the labels for the cropped images
		self.count = 0

		# Define a flag to keep track of the presence of a door to the right of the robot
		self.door_present = False

	# Setting initial roll, pitch and yaw angles to 0 thereby avoiding offsets	
	roll = pitch = yaw = 0.0

	# Function to get map details and create initial images of the map
	def map_callback(self, msg):
		# Getting the metadata of the map
		self.map_height = msg.height
		self.map_width = msg.width
		self.map_resolution = msg.resolution

		# Loading the actual map which shall be manipulated 
		original_map_image = Image.open('/home/glaurung/catkin_ws/src/reddy_autonomy/maps/world.png')
		
		# Loading the robot image that represents a robot
		self.robot_image = Image.open('/home/glaurung/catkin_ws/src/reddy_autonomy/maps/robot.png')
		
		# Making a copy of the original map image
		grid_map_image = original_map_image.copy()
		
		# Initializing a 2D grid structure for the map where each cell contains a list
		self.grid = [[[] for _ in range(self.map_width)] for _ in range(self.map_height)]
		
		# Converting the map image into a numpy array  
		self.grid_map_image = np.array(grid_map_image)

	# Function to get the odometry data from the /odom topic
	def odom_callback(self, msg):
		# Get the current position (x,y) coordinates of robot
		self.current_x = round(msg.pose.pose.position.x, 3)
		self.current_y = round(msg.pose.pose.position.y, 3)
		
		# Get the current raw x,y,z values of robot from odometry
		self.orientation_list = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
		
		# Convert the raw angles to human understandable euler angles
		(roll, pitch, yaw) = euler_from_quaternion(self.orientation_list)
		
		# The current heading of the robot is given by the yaw
		self.current_heading = yaw*180 / pi

		# Visualizing the grid into an image
		self.grid = self.grid_visualization(msg)

	# Function to get the laser scan data from the /scan topic
	def scan_callback(self, msg):
		# Get the laser data from the laser scan of the robot
		self.laser_ranges = msg.ranges
		self.laser_intensities = msg.intensities
		self.laser_max_range = msg.range_max
		self.laser_min_range = msg.range_min
		self.laser_max_angle = msg.angle_max
		self.laser_min_angle = msg.angle_min
		self.laser_angle_increment = msg.angle_increment

	# Function to check if there the presence of a door on the /door_presence topic
	def door_callback(self, msg):
		# Set a flag in the presence of a door
		if msg:
			self.door_present = True

		else:
			print("----------------------------")

	# Function to visualize the path of the robot on the map
	def grid_visualization(self, msg):
		# Incrementing the label counter for the cropped images
		self.count += 1

		# Getting the current pixel position on the 2D grid structure of the map
		pixel_x = int(msg.pose.pose.position.x / self.map_resolution)
		pixel_y = int(msg.pose.pose.position.y / self.map_resolution)

		# Storing the current position of the robot in a list present in the particular cell of the grid
		self.grid[pixel_x][pixel_y].append((self.current_x, self.current_y))

		# Tracing the path taken by the robot through the occupied pixels		
		if self.grid[pixel_x][pixel_y]:
			
			# Marking the path of the robot in green color
			self.grid_map_image[self.map_height - pixel_y][pixel_x] = (0, 255, 0 ,1)
			
			# Converting the grid into an image
			grid_map_image = Image.fromarray(self.grid_map_image)
			
			# Pasting a robot image to visualize the robot following a path on the map image
			# grid_map_image.paste(self.robot_image, ((pixel_x + 40) - 5, (self.map_height - pixel_y - 180) - 5, (pixel_x + 40) + 5, (self.map_height - pixel_y - 180) + 5))
			
			# Cropping a 30x30 pixel image around the robot while it follows a path
			# cropped_region = grid_map_image.crop(((pixel_x + 40) - 30, (self.map_height - pixel_y - 180)-30, (pixel_x + 40) + 30, (self.map_height - pixel_y - 180) + 30))
			
			# Converting the updated map image from PIL image type to numpy array
			grid_map_image_array = np.array(grid_map_image)

			# Getting the shape of the map image array
			img_rows = grid_map_image_array.shape[0]
			img_cols = grid_map_image_array.shape[1]

			# Checking if the cropped region lies within the boundaries of the original image
			while True:
				center = (pixel_x, self.map_height-pixel_y)
				width = height = 30
				angle = self.current_heading
				rect = (center, (width, height), angle)
				if inside_rect(rect = rect, num_cols = img_cols, num_rows = img_rows):
					break

			# Getting the bounding box points for the cropped region
			box = cv2.boxPoints(rect).astype(np.int0)

			# Representing the bounding box and the heading of robot by drawing a box with an arrow in it
			# cv2.drawContours(grid_map_image_array, [box], 0, (0,0,0), 1)
			# cv2.arrowedLine(grid_map_image_array, center, ((box[1][0]+box[2][0])//2, (box[1][1]+box[2][1])//2), (0,0,0), 1, tipLength = 0.1)
			
			# Getting the cropped image after rotation, using the function from auto_crop.py
			cropped_image = crop_rotated_rectangle(image = grid_map_image_array, rect = rect)

			# Segregating the images based on the presence of a door and saving them in separate folders
			if self.door_present == True:
				
				# Ensuring that images are saved in this folder only in the presence of a door
				self.door_present = False
				
				# Locating the coordinated on the map where there is the presence of a door
				print(" There is a door located at ({0},{1})".format(pixel_x,pixel_y))

				# If the door is present then the cropped images are stored in the Bad_Images folder
				cv2.imwrite('/home/glaurung/catkin_ws/src/reddy_autonomy/Images/Bad_images/test{0}.png'.format(self.count), (cropped_image))
			# If there is no door present then the cropped images are stored in the Good_images folder
			cv2.imwrite('/home/glaurung/catkin_ws/src/reddy_autonomy/Images/Good_images/test{0}.png'.format(self.count), (cropped_image))		

			# Visualizing the movement of the robot on the map using opencv and pillow library
			grid_map_image = np.array(grid_map_image)
			cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
			cv2.resizeWindow('Image', 1000, 1000)
			cv2.imshow('Image', grid_map_image)
			cv2.waitKey(10)

		return self.grid
	
	
if __name__ == '__main__':
	# Initializing a unique node to hold information about the robot
	rospy.init_node('autonomous_robot', anonymous = True)
	# Making an instance of the robot class
	robot = Autonomy()
	# To stop the node when ctrl + C is pressed
	rospy.spin()
