# Super Mario Coin Recognition

This project is a computer vision application built with **Python** and **OpenCV** designed to detect and calculate the total value of different types of coins from Super Mario game screenshots. It uses classical image processing techniques to distinguish between small yellow coins, big yellow coins, and red coins.

## Features

* **Multi-Coin Detection:** Recognizes three types of coins with different values:
    * **Small Yellow Coin:** 1 point
    * **Red Coin:** 2 points
    * **Big Yellow Coin:** 5 points
* **Custom Masking:** Implements an "Outlier Mask" to filter out background elements that share similar colors with coins.
* **Watershed Segmentation:** Efficiently separates touching objects for more accurate counting.
* **Shape Validation:** Uses geometric properties like Solidity and Aspect Ratio to distinguish coins from UI elements and blocks.
* **Performance Metrics:** Automatically calculates the **Mean Absolute Error (MAE)** by comparing results against ground truth data.

##  How It Works

The detection pipeline consists of several key stages:

### 1. Image Preprocessing
* **ROI Masking:** The script blacks out specific areas (like UI/Score elements) to prevent false detections.
* **HSV Conversion:** Converts images to the HSV color space for more robust color segmentation.

### 2. Advanced Masking & Noise Reduction
* **Dual Masking:** Beyond standard yellow/red masks, a custom **Outlier Mask** is applied to subtract background colors that overlap with the yellow spectrum.
* **Morphological Operations:** Uses `Dilate` and `Erode` (Opening/Closing) to remove small noise and fill gaps in the masks.

### 3. Object Separation (Watershed)
To handle coins that are close to each other, the project uses the **Watershed Algorithm**:
* Computes the **Distance Transform**.
* Identifies "Sure Foreground" and "Sure Background".
* Markers are created to segment individual coins accurately.

### 4. Contour Filtering
Objects are filtered based on their properties to ensure only coins are counted:
* **Area:** $375 < Area < 12500$
* **Solidity:** $\ge 0.85$ (Ensures the object is "full" and not a thin ring or hollow shape).
* **Aspect Ratio:** $0.54 - 1.02$ (Checks if the object is roughly circular/oval).
* **Extent:** $\le 0.8$ (Helps distinguish coins from perfect squares/blocks).

## Evaluation

The script automatically processes all images in a provided directory and compares the predicted total value with the actual values stored in `coin_value_count.csv`.

```python
MAE = sum(abs(predicted - actual)) / n
```

## Installation

### Start python virtual environment
```python
py -3.12 -m venv env    
env\Scripts\activate
```

### Install requirements
```python
pip install opencv-contrib-python numpy matplotlib
```

### Run the program
```python
python main.py data
```