# python yolov8n_seg_640_ratbw_metric.py
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import os.path as osp

# 【手动修改】每个批次需要修改的参数
data = "/home/liying_lab/chenxinfeng/DATA/ultralytics/data/rats_metric/dataset_extra.yaml"
epochs = 40

# 【不要变更】 载入模型
DEFAULT_CFG.save_dir = osp.join(osp.dirname(__file__), 'work_dirs', osp.splitext(osp.basename(__file__))[0])
modelfile_last = osp.join(DEFAULT_CFG.save_dir, 'weights/last.pt')
modelfile_load = '/home/liying_lab/chenxinfeng/DATA/ultralytics/checkpoints/yolov8n-seg.pt' if not osp.exists(modelfile_last) else modelfile_last
print('====load model:', modelfile_load, '======\n\n')
model = YOLO(model=modelfile_load)  # load a model file

# 【不要变更】 训练模型
DEFAULT_CFG.save_dir = osp.join(osp.dirname(__file__), 'work_dirs', osp.splitext(osp.basename(__file__))[0])
model.train(data=data, epochs=epochs, imgsz=640, scale=0.2, degrees=10, device=[0], rect=True, augment=False)

# 【不要变更】 导出模型
# model = YOLO(model=modelfile_last)
# path = model.export(format="onnx")
