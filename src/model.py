from ultralytics.nn.tasks import DetectionModel
from loss import YOCOLoss
from torch import nn

class YOCOModel(DetectionModel):
    def __init__(self, 
                 cfg, 
                 ch=3, 
                 nc=None, 
                 verbose=True, 
                 yoco=False,
                 args=None,
                 cmd_args=None):
        super().__init__(cfg=cfg, ch=ch, nc=nc)
        self.yoco = yoco
        self.args = args
        self.cmd_args = cmd_args
    
    # Custom YOCO loss
    def init_criterion(self):
        return YOCOLoss(self, self.yoco, self.args, self.cmd_args)