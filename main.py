import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

MIN_AREA = 375
MAX_AREA = 12500
SMALL_COIN_AREA_THRESHOLD = 3750
BIG_COIN_AREA_THRESHOLD = 7000
MIN_ASPECT_RATIO = 0.54
MAX_ASPECT_RATIO = 1.02
MIN_SOLIDITY = 0.85
MAX_EXTENT = 0.8

def load_coin_value_count(file):
    coin_value_count = {}
    with open(file, "r") as f:
        f.readline()
        for line in f.readlines():
            data = line.split(",")

            coin_value_count[data[0]] = data[1]

    return coin_value_count
    
def mean_absolute_error(predicted, actual):
    if len(predicted) != len(actual):
        return -1
    
    sum = 0
    n = len(predicted)
    for i in range(n):
        sum += abs(int(predicted[i]) - int(actual[i]))
    
    return sum / n

def watershed(masked_img, mask, kernel, min_area, max_area):
    contours_all, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask_yellow = np.zeros_like(mask)
    
    for contour in contours_all:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area: 
            cv2.drawContours(cleaned_mask_yellow, [contour], -1, 255, -1)

    # plt.imshow(cleaned_mask_yellow, cmap='gray')
    # plt.title('Filtered Large Objects')
    # plt.show()

    sure_bg = cv2.dilate(cleaned_mask_yellow, kernel, iterations = 4)
    dist_transform = cv2.distanceTransform(cleaned_mask_yellow, cv2.DIST_L2, 5)
    # plt.imshow(dist_transform, cmap='gray')
    # plt.show()

    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0) 
    # plt.imshow(sure_fg, cmap='gray')
    # plt.show()

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # plt.imshow(unknown, cmap='gray')
    # plt.show()

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(masked_img, markers)

    watershed_mask_yellow = np.zeros_like(cleaned_mask_yellow, dtype=np.uint8)
    watershed_mask_yellow[markers > 1] = 255 
    contours_yellow, _ = cv2.findContours(watershed_mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours_yellow, cleaned_mask_yellow

def apply_masks(img):
    lower_yellow = np.array([19, 70, 49])
    upper_yellow = np.array([45, 255, 255])
    lower_red = np.array([0, 135, 80])
    upper_red = np.array([10, 255, 195])

    lower_outlier = np.array([30, 130, 104])
    upper_outlier = np.array([45, 180, 180])

    mask_outlier = cv2.inRange(img, lower_outlier, upper_outlier)
    mask_yellow_original = cv2.inRange(img, lower_yellow, upper_yellow)
    mask_yellow = cv2.bitwise_and(mask_yellow_original, cv2.bitwise_not(mask_outlier))
    mask_red = cv2.inRange(img, lower_red, upper_red)
    
    # plt.figure(figsize=(16, 12))
    # plt.subplot(1, 3, 1)
    # plt.imshow(mask_yellow_original, cmap='gray')
    # plt.title('1. Original Mask')
    
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask_outlier, cmap='gray')
    # plt.title('2. Outlier Mask')

    # plt.subplot(1, 3, 3)
    # plt.imshow(mask_yellow, cmap='gray')
    # plt.title('3. Yellow Mask')
    # plt.show()

    return mask_yellow, mask_red

def calculate_solidity_ratio_extent(contour, area):
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return True
    
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return True
    
    solidity = area / hull_area
    aspect_ratio = w / h
    extent = area / (w * h)

    return False, solidity, aspect_ratio, extent

def process_yellow_coins(contours):
    yellow_small_coins = []
    yellow_big_coins = []
    total_value = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA and area < MAX_AREA:
            zero, solidity, aspect_ratio, extent = calculate_solidity_ratio_extent(contour, area)
            if zero:
                continue

            if (solidity >= MIN_SOLIDITY and aspect_ratio >= MIN_ASPECT_RATIO and aspect_ratio <= MAX_ASPECT_RATIO and extent <= MAX_EXTENT):
                if area >= BIG_COIN_AREA_THRESHOLD and aspect_ratio >= 0.6:
                    total_value += 5
                    yellow_big_coins.append(contour)
                    
                elif area <= SMALL_COIN_AREA_THRESHOLD:
                    total_value += 1
                    yellow_small_coins.append(contour)
    
    return total_value, yellow_small_coins, yellow_big_coins

def process_red_coins(contours):
    total_value = 0
    red_coins = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA and area < MAX_AREA:
            zero, solidity, aspect_ratio, extent = calculate_solidity_ratio_extent(contour, area)
            if zero:
                continue

            if (solidity >= MIN_SOLIDITY and aspect_ratio >= MIN_ASPECT_RATIO and aspect_ratio <= MAX_ASPECT_RATIO and extent <= MAX_EXTENT):
                total_value += 2
                red_coins.append(contour)

    return total_value, red_coins

def detect_value(img):
    yellow_small_coins = []
    yellow_big_coins = []
    red_coins = []
    total_value = 0
    
    white_mask = np.full(img.shape, 255, dtype=np.uint8)
    mask = cv2.rectangle(white_mask, (100, 60), (150, 110), (0, 0, 0), -1)
    mask = cv2.rectangle(mask, (200, 60), (250, 110), (0, 0, 0), -1)
    blacked_img = cv2.bitwise_and(img, mask)

    # plt.figure(figsize=(16, 12))
    # plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    # plt.title('Blacked Image)')
    # plt.show()

    hsv = cv2.cvtColor(blacked_img, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    mask_yellow, mask_red = apply_masks(hsv)

    # searching for yellow coins
    img_dil_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)
    mask_yellow = cv2.erode(img_dil_yellow, kernel, iterations=3)

    contours_yellow_small_coin, mask_small_coin = watershed(blacked_img, mask_yellow, kernel, MIN_AREA, SMALL_COIN_AREA_THRESHOLD)
    contours_yellow_big_coin, mask_big_coin = watershed(blacked_img, mask_yellow, kernel, BIG_COIN_AREA_THRESHOLD, MAX_AREA)

    contours_yellow = contours_yellow_big_coin + contours_yellow_small_coin
    mask_yellow = cv2.bitwise_or(mask_big_coin, mask_small_coin)

    total_value, yellow_small_coins, yellow_big_coins = process_yellow_coins(contours_yellow)

    # searching for red coins
    img_dil_red = cv2.dilate(mask_red, kernel, iterations=1)
    mask_red = cv2.erode(img_dil_red, kernel, iterations=3)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_red, red_coins = process_red_coins(contours_red)
    total_value += total_red

    img_contours = img.copy()
    cv2.drawContours(img_contours, yellow_small_coins, -1, (0, 0, 255), 2)
    cv2.drawContours(img_contours, yellow_big_coins, -1, (0, 255, 0), 2)
    cv2.drawContours(img_contours, red_coins, -1, (255, 0, 0), 2)

    cleaned_mask = cv2.bitwise_or(mask_yellow, mask_red) 

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cleaned_mask, cmap='gray')
    plt.title('Color Mask Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.title(f'Total Coin Value: {total_value}     Yellow: {len(yellow_small_coins)}     Big: {len(yellow_big_coins)}     Red: {len(red_coins)}')

    plt.show()
    
    return total_value

if __name__ == "__main__":
    data_folder = sys.argv[1]
    predicted_values = []
    actual_values = []

    coin_value_count = load_coin_value_count(os.path.join(data_folder, "coin_value_count.csv"))
    
    for img_name in coin_value_count.keys():
        img = cv2.imread(os.path.join(data_folder, img_name))

        predicted_values.append(detect_value(img))
        actual_values.append(coin_value_count[img_name])

    mae = mean_absolute_error(predicted_values, actual_values)

    print(mae)
