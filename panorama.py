import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_match_features(img1, img2):
    """
    1. SIFT Feature Extraction
    2. Feature Correspondence (KNN Match)
    """
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Detect Keypoints and Compute Descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match features using FLANN (Fast Library for Approximate Nearest Neighbors)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's Ratio Test to keep only "good" matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches

def stitch_two_images(img_base, img_warp):
    """
    Stitches img_warp ONTO img_base.
    """
    # 1. Find features
    kp_base, kp_warp, matches = detect_and_match_features(img_base, img_warp)

    if len(matches) < 4:
        print("Not enough matches found.")
        return None

    # 2. Extract location of good matches
    src_pts = np.float32([kp_warp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_base[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 3. Geometric Transformation: Find Homography (H) using RANSAC
    # H maps points from 'img_warp' to 'img_base'
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 4. Image Warping
    # Get dimensions of both images
    h1, w1 = img_base.shape[:2]
    h2, w2 = img_warp.shape[:2]

    # Get the corners of the warping image to find the size of the final canvas
    pts_warp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts_warp_transformed = cv2.perspectiveTransform(pts_warp, H)

    # Combine with base image corners
    pts_base = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    all_pts = np.concatenate((pts_base, pts_warp_transformed), axis=0)

    # Find new canvas dimensions
    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift the image if coordinates are negative
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])

    # Warp the image
    output_img = cv2.warpPerspective(img_warp, H_translation.dot(H), (xmax-xmin, ymax-ymin))

    # 5. Artifact Mitigation & Blending
    # Paste the base image onto the canvas
    # We use a simple mask to blend overlaps
    base_transformed = np.zeros_like(output_img)
    base_transformed[translation_dist[1]:h1+translation_dist[1],
                     translation_dist[0]:w1+translation_dist[0]] = img_base

    # Create masks
    mask_warp = (cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY) > 0)
    mask_base = (cv2.cvtColor(base_transformed, cv2.COLOR_BGR2GRAY) > 0)

    # Where both exist, blend them (Simple average for assignment purposes)
    overlap = mask_warp & mask_base
    output_img[overlap] = cv2.addWeighted(output_img[overlap], 0.5, base_transformed[overlap], 0.5, 0)

    # Where only base exists, copy it over
    output_img[~overlap & mask_base] = base_transformed[~overlap & mask_base]

    return output_img

def crop_black_borders(img):
    """
    6. Final Cleanup: Crop to remove black edges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return img[y:y+h, x:x+w]
    return img

def create_naive_stitch(img1, img2, img3):
    """
    Creates a simple horizontal concatenation to represent a 'naive' approach.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]

    # Ensure all images are the same height for the naive strip
    min_h = min(h1, h2, h3)
    img1_r = cv2.resize(img1, (int(w1 * min_h / h1), min_h))
    img2_r = cv2.resize(img2, (int(w2 * min_h / h2), min_h))
    img3_r = cv2.resize(img3, (int(w3 * min_h / h3), min_h))

    return np.hstack((img1_r, img2_r, img3_r))


# --- MAIN PIPELINE ---
# Load images
img_l = cv2.imread('left.jpg')
img_c = cv2.imread('center.jpg')
img_r = cv2.imread('right.jpg')

# Resize to speed up processing (optional, but recommended for phone photos)
scale_percent = 50
width = int(img_l.shape[1] * scale_percent / 100)
height = int(img_l.shape[0] * scale_percent / 100)
dim = (width, height)

img_l = cv2.resize(img_l, dim)
img_c = cv2.resize(img_c, dim)
img_r = cv2.resize(img_r, dim)

# naive stitch for these three images
print("Creating naive stitch...")
naive_result = create_naive_stitch(img_l, img_c, img_r)
#save naive stitch
cv2.imwrite('naive_stitch.jpg', naive_result)

print("Stitching Right to Center...")
# Step 1: Stitch Right image onto Center image
temp_result = stitch_two_images(img_c, img_r)

print("Stitching Left to Result...")
# Step 2: Stitch Left image onto the Result of step 1
final_result = stitch_two_images(temp_result, img_l)

# Step 3: Cleanup
final_result = crop_black_borders(final_result)

# Save and Show
cv2.imwrite('final_panorama.jpg', final_result)



plt.figure(figsize=(10, 5))
plt.title("Naive Stitch")
plt.imshow(cv2.cvtColor(naive_result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Final Panorama")
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# 1. SIFT FEATURE EXTRACTION:
#     -We initialize the SIFT algorithm by calling cv2.SIFT_create().
#     -The algorithm scans the images to find distinctive keypoints (corners and blobs) and computes descriptors (mathematical vectors representing the texture), which remain invariant even if the camera moves or zooms.

# 2. FEATURE CORRESPONDENCE:
#     -We will be utilizing the FLANN Matcher (Fast Library for Approximate Nearest Neighbors) to efficiently find matches between corresponding features in the two images.
#     -To reject poor matches, we will apply Lowe's Ratio Test with a threshold set to 0.75.

# 3. ROBUST ESTIMATION (RANSAC) & HOMOGRAPHY:
#     -We will be utilizing the matched points to compute a 3x3 Homography Matrix (H).
#     -To account for possible errors in matching the features, we will be utilizing RANSAC (Random Sample Consensus) with a threshold set to 5.0.
#     -The RANSAC algorithm randomly selects a set of matches and uses it to compute a transformation that best matches most points, ignoring the "outliers" (noise).
# 4. IMAGE WARPING & CANVAS SIZING:
#     -We can calculate the size of the final stitched image by transforming the corners of the source image with the Homography matrix.
#     -If the warped image is found to be projecting into negative coordinates, we calculate a Translation Matrix.
#     -We then use cv2.warpPerspective to project the image onto a new, larger canvas that fits both images.

# 5. ARTIFACT MITIGATION (WEIGHTED BLENDING):
#     -If we were to directly overlap the two images, there would be a "cut" between the two images.
#     -To avoid this, our pipeline has identified the region where the two images overlap (i.e., where there are pixels from both).
#     -We then use Weighted Averaging over this region with cv2.addWeighted. We blend 50% of the base image with 50% of the warped image, which smooths over exposure differences and eliminates the edge.

# 6. ITERATIVE PROCESSING:
#     -We process the two images iteratively to create the complete panoramic view.
#     -We start by stitching the Right image with the Center image, which gives us an intermediate result.
#     -We then stitch the Left image with this intermediate result to get the complete panoramic view.

#
