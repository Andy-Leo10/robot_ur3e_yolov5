# robot_ur3e_yolov5

```
git clone https://github.com/Andy-Leo10/robot_ur3e_yolov5.git
```

- [robot\_ur3e\_yolov5](#robot_ur3e_yolov5)
  - [Motivation](#motivation)
  - [Optional setup](#optional-setup)
    - [tensorboard](#tensorboard)
    - [pip in venv](#pip-in-venv)
  - [trains](#trains)
    - [1. e=10](#1-e10)
    - [2. e=100](#2-e100)
    - [3. e=300](#3-e300)
    - [4. e=y](#4-ey)
  - [tests](#tests)
    - [1. d=0](#1-d0)
    - [2. d=0.86](#2-d086)
    - [3. d=0.94](#3-d094)
    - [4. e=y](#4-ey-1)

---

<details>
<summary><b>Motivation</b></summary>

## Motivation
When working on a real environment it is difficult to tune parameters in a program based on digital image processing. For that reason, a good approach is to use a CNN.
![alt text](assets/motivation.png)

</details>

---

<details>
<summary><b>Optional setup</b></summary>

## Optional setup
### tensorboard
```
cd yolov5
tensorboard --logdir runs\train
```
### pip in venv
```
.venv\Scripts\python.exe -m pip install <LIBRARY>
```

</details>

---

<details>
<summary><b>TRAINING</b></summary>

## trains

### 1. e=10
```
python train.py --img 424 --batch 16 --epochs 10 --data customData.yaml --weights yolov5n.pt --cache
```

### 2. e=100
```
python train.py --img 424 --batch 16 --epochs 100 --data customData.yaml --weights yolov5n.pt --cache
```

### 3. e=300
```
python train.py --img 424 --batch 16 --epochs 300 --data customData.yaml --weights yolov5n.pt --cache
```

### 4. e=y
```

```

</details>

---

<details>
<summary><b>TESTING</b></summary>

## tests

### 1. d=0
```
python detect.py --weights C:\Users\Leo\Downloads\MachineLearning\yolov5\runs\train\exp\weights\best.pt --img 424 --conf 0.25 --source C:\Users\Leo\Downloads\MachineLearning\photo_1718317391.jpg
```

### 2. d=0.86
```
python detect.py --weights C:\Users\Leo\Downloads\MachineLearning\yolov5\runs\train\exp1\weights\best.pt --img 424 --conf 0.25 --source C:\Users\Leo\Downloads\MachineLearning\photo_1718317391.jpg
```

### 3. d=0.94
```
python detect.py --weights C:\Users\Leo\Downloads\MachineLearning\yolov5\runs\train\exp2\weights\best.pt --img 424 --conf 0.25 --source C:\Users\Leo\Downloads\MachineLearning\photo_1718317391.jpg
```

### 4. e=y
```

```

</details>

---
