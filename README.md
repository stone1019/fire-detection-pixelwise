# Pixel-Wise Fire Detection based on Fuzzy Inference System

This repo is an implementation of Pixel-Wise Fire Detection method which is based on Fuzzy Inference System.

<p align="center">
  <img src=assets/video2.gif?style=centerme>
</p>

## **Approach:**
 ### WorkFlow of Fire Detection
 
* **RGB2YCrCb** Convert image to YCrCb Color Space.
* **Fuzzify input:** feed $Y - Cb$ and $Cr - Cb$
  <p align="center">
    <img src=assets/rule1.png>
  </p>
  
* **Apply rule table**
  <p align="center">
    <img src=assets/rule_table.png>
  </p>
  
* **Defuzzify output:** output probability
  <p align="center">
    <img src=assets/rule2.png>
  </p>
  
* **Reference:**
  This implementation is based on the method described below, but different rule table and implication operator are used.
  ```
  Çelik T, Özkaramanlı H, Demirel H. Fire and smoke detection without sensors: Image processing based approach[C]//2007 15th European Signal Processing Conference. IEEE, 2007: 1794-1798.
  ```
  
## **Demo**

  ```
  mkdir build
  cmake ..
  make
  ./fireDetection 'i' /path/to/image
  ```
