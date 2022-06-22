# Importing libraries
import cv2
import numpy as np
from math import pi
import csv

# Set upper and lower bound of white colour
lower_bound = np.array([0, 0, 137])
upper_bound = np.array([161, 54, 255])

# Constant of computing from rectangle to circle
constant = (1296 * 972) / (pi * (450 ** 2))


# Function for computing percentage of white in the mask
def calc_percentage(msk_def):
    height, width = msk_def.shape[:2]
    num_pixels = height * width
    count_white = cv2.countNonZero(msk_def)
    percent_white = (count_white / num_pixels) * 100
    percent_white = round(percent_white, 2)
    return percent_white


# Main Loop
def main_loop(num_photos, path_to_file, constant_image, debug=False):
    file = open(path_to_file, 'w')
    csv_writer = csv.writer(file)
    i = 0
    while i < num_photos:
        # Reading an image, change it to HSV color spectre and creating mask
        img = cv2.imread(f'C:\\Users\\shanak\\Downloads\\Cm_pi_jr_ex\\image{i}.jpg')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(img_hsv, lower_bound, upper_bound)

        # Computing of percentage, print it for checking is it right and append it to list of percentage
        text = (calc_percentage(msk) * constant_image)
        csv_writer.writerow([str(text)])
        if debug:
            print(i, text)
        i += 1


# Function for concatenate two csv files to one
def concatenate(path_to_file1, path_to_file2, path_to_result):
    file1 = open(path_to_file1, 'r')
    file2 = open(path_to_file2, 'r')
    result = open(path_to_result, 'w')
    file1_list = list(csv.reader(file1))
    file1_list = list(map(''.join, file1_list))
    file2_list = list(csv.reader(file2))
    file2_list = list(map(''.join, file2_list))
    writer_result = csv.writer(result)
    for i in range(684):
        writer_result.writerow((i, file2_list[i], file1_list[i * 2]))


# Call functions
main_loop(684, 'C:\\Users\\shanak\\Downloads\\Percentage.csv', constant)
concatenate('C:\\Users\\shanak\\Downloads\\Percentage.csv',
            'C:\\Users\\shanak\\Downloads\\FinalResult.csv',
            'C:\\Users\\shanak\\Downloads\\Pokus.csv')
