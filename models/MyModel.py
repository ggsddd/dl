from torch import nn
import torch as t
import time
class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = str(type(self))

    def load(self,path):
        self.load_state_dict(t.load(path))

    def save(self,name):
        """
        模型保存至checkpoints目录中，默认命名方式为模型名称+时间
        """
        if name is not None:
            prefix = 'checkpoints/' + name
        else:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(),name)
        return name

    def forward(self, x):
        pass
