# HDR Imaging & Bayer Demosaicing (FAU Computer Vision Project – Exercise 2)

This repository contains the full implementation of **Exercise 2: Demosaicing & High Dynamic Range (HDR)** 

All algorithms (demosaicing, HDR merging, tone mapping, response curve estimation) are implemented **from scratch**, following the methods taught in the lecture.

---

## 📌 Overview of Tasks

### 1️⃣ Investigate Bayer Patterns
- Inspect raw sensor data (`.CR3` → `.raw_image_visible`) to detect the Bayer pattern.

### 2️⃣ Simple Demosaicing

### 3️⃣ Improve Luminosity

### 4️⃣ White Balance (Gray World)
- Implemented gray-world algorithm with clipping for high dynamic values.

### 5️⃣ Show Sensor Linearity
- Produced a plot verifying linearity.

### 6️⃣ HDR Merging (Lecture Method)
- Combined differently exposed RAW frames (00.CR3–10.CR3).
- Applied weighted replacement method (lecture slides): brighter image replaces saturated pixels.
- Demosaiced and white-balanced *after* HDR merging.
- Tone mapping performed with logarithmic compression.

### 7️⃣ iCAM06 Implementation

### 8️⃣ `process_raw()` Function
- A function that loads a `.CR3` file and outputs a high-quality JPG (quality=99).
- Includes demosaicing, white balance, and gamma correction.

### 9️⃣ Individual Exercise — Response Curve Estimation
- Estimated the camera response curve `g(z)` using least-squares (not OpenCV’s Debevec).
- Computed linearized radiance values and produced HDR from JPG input as required.

---

