# YOLO-V4
Object detection in this repository is done using ready weights. If you want to build your own weight, take a look at this 
# <! buraya yolo-cloud repositorymi paylaştığım zaman linkini koyacağım>

### Object Detection
<p align="center"><img src="https://user-images.githubusercontent.com/77979910/157029804-76e5e460-da5c-494d-b14f-75d4e90b8d83.gif" width="550"\></p>

## Getting Started
### You need to install CUDA for GPU.
### Conda 

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

## USE MY LICENSE PLATE TRAINED CUSTOM WEIGHTS
   https://drive.google.com/file/d/15i6Y3SvCT088KyLewoMxhJwSBbIhEKeq/view?usp=sharing
 

Copy and paste your yolov4-obj_best.weights file into the 'data' folder and copy and paste your obj.names into the 'data/classes/' folder.
The only change within the code you need to make in order for your custom model to work is on line 14 of 'core/config.py' file.
Update the code to point at your custom .names file as seen below. (my custom .names file is called custom.names but yours might be named differently)
<p align="center"><img src="https://user-images.githubusercontent.com/77979910/157033464-9dda3814-bffa-4009-b3b8-f5235b0c67fb.png" width="640"\></p>

<strong>Note:</strong> If you are using the pre-trained yolov4 then make sure that line 14 remains <strong>coco.names</strong>.

## YOLOv4 Using Tensorflow
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.
```bash
# Convert darknet weights to tensorflow
## yolov4
python save_model.py --weights ./data/yolov4-obj_best.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

# Run yolov4 tensorflow model
python detect.py --weights ./checkpoints/yolov4-obj_best-416 --size 416 --model yolov4 --images ./data/images/kite.jpg

# Run yolov4 on video
python detect_video.py --weights ./checkpoints/yolov4-obj_best-416 --size 416 --model yolov4 --video ./data/video/video.mp4 --output ./detections/results.avi

# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/yolov4-obj_best-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi
```
If you want to run yolov3 or yolov3-tiny change ``--model yolov3`` and .weights file in above commands.

<strong>Note:</strong> You can also run the detector on multiple images at once by changing the --images flag like such ``--images "./data/images/kite.jpg, ./data/images/dog.jpg"``


## Custom YOLOv4 Using TensorFlow
The following commands will allow you to run your custom yolov4 model. (video and webcam commands work as well)
```
# custom yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Run custom yolov4 tensorflow model
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car.jpg
```

#### Custom YOLOv4 Model Example
<p align="center"><img src="https://user-images.githubusercontent.com/77979910/157036815-6dc24df7-294c-4017-8e43-f4886006404c.png" width="640"\></p>
