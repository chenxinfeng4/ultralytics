#%%
#python -m yolov8_det_640x640_ball $modelfile
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import os.path as osp
import argparse
#%%
def train(modelfile):
    modelfile = "/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_ball/weights/last.pt"
    model = YOLO(model=modelfile)  
    DEFAULT_CFG.save_dir = osp.join(osp.dirname(__file__), 'work_dirs', osp.splitext(osp.basename(__file__))[0])
    model.train(data="/home/liying_lab/chenxinfeng/DATA/ultralytics/data/ball_det/YOLODataset/dataset.yaml", epochs=50, 
            imgsz=640, scale=0.2, degrees=10, device=[0], rect=True, augment=False)
    path = model.export(format="onnx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelfile', type=str, help='load from modelfile')
    args = parser.parse_args()
    train(args)

# %%
#convert
#predict
#modelfile = '/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_ball/weights/last.onnx'