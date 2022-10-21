# An example of a deep learning framework - AEDLF
## AEDLF : 一个深度学习框架基础示例
目前已经实现MLP网络的搭建和训练，bin文件夹中提供的是linux环境下的预编译程序           
具体实现在aedlf.cpp中，目前因时间安排问题，暂无编写相关说明的意向，等到TODO适当完结后进行相关补充                
核心代码均在include文件夹中，src文件夹中可能残留了开发早期代码，但无法使用              
### 支持以下功能
* 静态计算图自动搭建
* 自动求导
* 简单的数据载入（依靠numpy）
### TODO
* 调试CV相关算子
* 编写优化器相关代码
* 优化计算过程
* 切换到动态图
* 增加CUDA加速相关功能
* 支持计算图剪枝优化等操作
* 支持模型并行、数据并行
### 本项目参考资料（及鸣谢）
* [Caffe](https://github.com/BVLC/caffe)
* [Pytorch DOC](https://pytorch.org/docs/stable/index.html)
* [ai-edu(Microsoft)](https://microsoft.github.io/ai-edu/)
* [cppreference.com](https://en.cppreference.com/w/)