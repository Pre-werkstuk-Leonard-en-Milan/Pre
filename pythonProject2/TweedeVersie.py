import cv2
import numpy as np
import os

# giving directory name
dirname = 'C:\\Users\\Caroline\\Desktop\\fotos_robots\\pythonProject2'

# giving file extension
ext = ('.PNG', 'jpg')

# iterating over all files
for files in os.listdir(dirname):
    if files.endswith(ext):
        print(files)  # printing file name of desired extension

        # Load image
        image = cv2.imread(files, 0)

        contrast = np.std(image)

        print(f"Contrast of the image: {contrast:.2f}")

        img = cv2.imread(files, cv2.IMREAD_COLOR)

        # Convert the image to HSV color space
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Extract the saturation channel
        saturation = image_hsv[:, :, 1]

        # Calculate mean saturation (normalized)
        mean_saturation = np.mean(saturation) / 255.0  # Normalized between 0 and 1

        # Convert the image to RGB and split channels
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        R, G, B = cv2.split(image_rgb)

        # Calculate the standard deviation of each color channel
        std_R = np.std(R)
        std_G = np.std(G)
        std_B = np.std(B)

        # Compute the overall colorfulness using Hasler and Süsstrunk's method
        sigma_rg = np.sqrt(std_R ** 2 + std_G ** 2)
        mu_rg = np.sqrt(np.mean(R) ** 2 + np.mean(G) ** 2)

        # Combined colorfulness index (taking saturation into account)
        colorfulness = (sigma_rg + 0.3 * mu_rg) * mean_saturation

        # Print the colorfulness index

        print(f'Colorfulness Index: {colorfulness:.2f}')

        # Detect corners using the Shi-Tomasi method
        corners = cv2.goodFeaturesToTrack(image, 100, 0.01, 100)
        corners = np.int8(corners)
        num_corners = len(corners)
        print(f"corners in image: {num_corners:.2f}")

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, 0, -1)

        cv2.imshow('Shi-Tomasi Corner Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Set our filtering parameters
        # Initialize parameter setting using cv2.SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()

        # Set Area filtering parameters
        params.filterByArea = True
        params.minArea = 50

        # Set Circularity filtering parameters
        params.filterByCircularity = True
        params.minCircularity = 0.5  # default 0.9

        # Set Convexity filtering parameters
        params.filterByConvexity = True
        params.minConvexity = 0.000005  # default 0.2

        # Set inertia filtering parameters
        params.filterByInertia = True
        params.minInertiaRatio = 0.000000010  # dewfault 0.01

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(image)

        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        number_of_blobs = len(keypoints)
        text = "Number of Circular Blobs: " + str(len(keypoints))

        print("Number of Circular Blobs: " + str(len(keypoints)))
        cv2.putText(blobs, text, (20, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)



        # Show blobs
        cv2.imshow("Filtering Circular Blobs Only", blobs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()






    else:
        continue





