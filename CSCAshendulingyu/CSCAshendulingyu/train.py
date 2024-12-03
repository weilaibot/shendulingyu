import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
     model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=64,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD',
                # patience=0,
                # resume=True,
                # amp=False,
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )