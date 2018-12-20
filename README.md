# Mask-RCNN Implementation in Python using OpenCV


This deep learning architecture is used for instance segmentation. Mask RCNN combines the two networks — Faster RCNN and FCN in one mega architecture.

[Mask-RCNN paper](https://arxiv.org/pdf/1703.06870.pdf)

### Running this repo

* Requires OpenCV 3.4.3 or higher
* Download this repo and run the following command

```
python3 mask_rcnn.py --image=img.jpg
```
This implementation has been done following the tutorial on learnopencv.com. Please note that this implementation is only for forward pass computation and no training can be done for a custom dataset. 