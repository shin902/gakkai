import cv2
import numpy as np
import sys

def calculate_sharpness(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpness
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python image_sharpness.py <image_path1> <image_path2>")
        sys.exit(1)

    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]

    sharpness1 = calculate_sharpness(image_path1)
    sharpness2 = calculate_sharpness(image_path2)

    if sharpness1 != -1 and sharpness2 != -1:
        print(f"Sharpness of {image_path1}: {sharpness1}")
        print(f"Sharpness of {image_path2}: {sharpness2}")
        if sharpness1 > sharpness2:
            print(f"Image {image_path1} is sharper.")
            print(image_path1) # sharper image path
        elif sharpness2 > sharpness1:
            print(f"Image {image_path2} is sharper.")
            print(image_path2) # sharper image path
        else:
            print("Images have equal sharpness.")
