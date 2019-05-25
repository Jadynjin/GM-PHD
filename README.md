# GM-PHD for multi-target tracking

GM-PHD方法出自 B.-N. VO, MA W-K. The Gaussian Mixture Probability Hypothesis Density Filter[J]. IEEE Transactions on Signal Processing, 2006, 54(11): 4091–4104.

## 内容

`gmphd.py`: GM-PHD的Python实现。源自[隔壁](https://github.com/danstowell/gmphd)。

改动：
- 使用Python 3
- 改为用于非线性系统的EKF
- 排序的bug（未修正）

`co-seek.py`: 文献中的例子？

`my-seek.py`: 2d cv model

`matlab/`: GM-PHD的MATLAB实现。源自文献作者写的[rfs_tracking_toolbox](http://ba-tuong.vo-au.com/codes.html)。改为3d cv model，内含多种黑魔法。不想维护了。