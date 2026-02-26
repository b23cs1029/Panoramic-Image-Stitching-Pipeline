# Panoramic Image Stitching Pipeline

This repository contains a simple image‑stitching pipeline implemented in Python using OpenCV. It was developed as part of a computer vision assignment and demonstrates the basic steps required to create a panorama from three overlapping photographs.

## Project Structure

```
assignment1/
├── Panorama.py          # main pipeline implementation
├── left.jpg             # sample input (not included)
├── center.jpg           # sample input (not included)
├── right.jpg            # sample input (not included)
├── naive_stitch.jpg     # generated naive concatenation
└── final_panorama.jpg   # output of the algorithm
```

> **Note:** The input images (`left.jpg`, `center.jpg`, `right.jpg`) should be placed alongside `Panorama.py`.

## ⚙️ Requirements

- Python 3.7 or newer
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib (optional, for display)

Install dependencies with:

```bash
pip install opencv-python numpy matplotlib
```

## 🚀 Usage

1. **Place your images** named `left.jpg`, `center.jpg`, and `right.jpg` in the same directory as `Panorama.py`.
2. **Run the script**:

   ```bash
   python Panorama.py
   ```

   The script will:
   - Create a naive horizontally concatenated image (`naive_stitch.jpg`).
   - Detect and match SIFT features between images.
   - Compute homographies with RANSAC and warp images.
   - Blend overlapping regions and crop black borders.
   - Save the final panorama as `final_panorama.jpg` and display both results.

## 📌 How it Works

The pipeline follows these high‑level steps:

1. **Feature Extraction and Matching** using SIFT + FLANN with Lowe's ratio test.
2. **Robust Homography Estimation** via RANSAC to align image pairs.
3. **Warping and Canvas Calculation** to accommodate both images.
4. **Blending** overlapping regions with simple averaging.
5. **Cropping** black borders from the stitched result.
6. **Iterative Stitching**: stitch right to center, then left to the result.

## 🧪 Improving the Pipeline

- Swap `cv2.SIFT_create()` with other detectors/descriptors (ORB, SURF).
- Use multi‑band or gradient‑based blending to reduce visible seams.
- Add exposure compensation and seam optimization.
- Extend the code to handle more than three images or automate image ordering.

