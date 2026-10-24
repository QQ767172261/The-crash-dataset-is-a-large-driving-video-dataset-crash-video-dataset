车祸数据集是一个大型驾驶视频数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c0accbf06175424ba41fd880435ca41e.png)

数量：500 多个各种驾驶场景的高分辨率视频，每个视频持续约几十秒
类别数量：这些视频是从不同的地点、天气条件和照明条件拍摄的
类别类型：这些视频在不同类型的汽车、行人和骑自行车的人等物体周围加上边界框，以及交通标志和红绿灯
用途：目标检测、自动驾驶、无人驾驶


# 车祸/自动驾驶驾驶视频数据集（使用全教程：YOLOv8训练+视频转图片+标注+部署）
我给你**最完整、可直接落地**的使用方案，专门针对**500+ 驾驶场景视频数据集**，用于**无人驾驶、目标检测、车祸检测、自动驾驶感知系统**。

---

## 一、数据集信息描述（表格版）
| 项目 | 内容 |
|------|------|
| 数据集名称 | 自动驾驶驾驶场景 / 车祸检测视频数据集 |
| 数据格式 | 高分辨率驾驶视频（几十秒/个） |
| 数量 | 500+ 个视频 |
| 场景 | 城市道路、高速、乡村、各种天气/光照 |
| 标注内容 | 车辆、行人、骑行者、交通标志、红绿灯 |
| 任务类型 | 目标检测 / 视频分析 / 自动驾驶感知 / 车祸识别 |
| 输出格式 | 可转 YOLO 格式训练 |

---

# 二、最重要：视频数据集 → 训练必须这样用
## 核心步骤（你必须按这个流程）
1. **视频抽帧**（把视频转成图片）
2. **数据清洗**（筛选有效帧）
3. **标注 / 使用已有标注**（转 YOLO 格式）
4. **训练 YOLOv8 / YOLOv11 目标检测模型**
5. **构建自动驾驶感知系统 / 车祸检测系统**

---

# 三、我直接给你全套可运行代码

## 1. 视频批量抽帧代码（把500个视频转图片）
`video2images.py`
```python
import os
import cv2

# 视频文件夹
video_dir = "videos/"
save_dir = "dataset/images/"
os.makedirs(save_dir, exist_ok=True)

# 每10帧取1张（可调）
step = 10

idx = 0
for video_name in os.listdir(video_dir):
    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_cnt % step == 0:
            cv2.imwrite(f"{save_dir}/{idx}.jpg", frame)
            idx += 1
        frame_cnt += 1
    cap.release()

print(f"抽取完成，共 {idx} 张图片")
```

---

## 2. 数据集目录结构（YOLO格式）
```
autopilot/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── autopilot.yaml
```

---

## 3. 数据集配置文件 `autopilot.yaml`
```yaml
train: ./images/train
val: ./images/val

nc: 5  # 根据你的标注类别调整
names:
  0: car
  1: person
  2: bicycle
  3: traffic_sign
  4: traffic_light
```

---

## 4. YOLOv8 训练代码（自动驾驶目标检测）
`train_autopilot.py`
```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8m.pt")

# 训练
model.train(
    data="autopilot.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=15,
    project="autopilot_detect",
    name="yolov8_autopilot",
    mosaic=0.5,
    fliplr=0.0,  # 驾驶画面不建议左右翻转
    hsv_h=0.3,
    hsv_s=0.5,
)

# 测试
model.val()
```

---

## 5. 视频实时检测推理代码（自动驾驶演示）
`infer_video.py`
```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

# 视频实时检测
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame)
    frame = results[0].plot()
    
    cv2.imshow("Autopilot Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---


