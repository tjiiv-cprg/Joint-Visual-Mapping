# Joint-Visual-Mapping

Paper: **Vision-based mapping of lane semantics and topology for intelligent vehicles**. *International Journal of Applied Earth Observation and Geoinformation  July 2022, 102851*

Wei Tian, Xiaozhou Ren, Xianwang Yu, Mingzhi Wu, Wenbo Zhao and Qiaosen Li. [[PDF](https://www.sciencedirect.com/science/article/pii/S156984322200053X)]

This repository is the PyTorch implementation for the framework of Joint-Visual-Mapping.

![Map](./demo/Map.png)

<div  align="center">   
<img src="./demo/normal.gif" width="200px"/><img src="./demo/fork.gif" width="300px"/>
</div>


## Abstract

High-definition map is an essential tool for route measurement, planning and navigation of intelligent vehicles. Yet its creation is still a persisting challenge, especially in creating the semantic and topology layer of the map based on visual sensing. However, current semantic mapping approaches do not consider the map applicability in navigation tasks while the topology mapping approaches face the issues of limited location accuracy or expensive hardware cost. In this paper, we propose a joint mapping framework for both semantic and topology layers, which are learned in a lane-level and based on a monocular camera sensor and an on-board GPS positioning device. A map management approach “RoadSegDict” is also proposed to support the efficient updating of semantic map in a crowdsourced manner. Moreover, a new dataset is proposed, which includes a variety of lane structures with detailed semantic and topology annotations.

## Framework

![Framework](./demo/Framework.png)

## Experimental results

### Mapping on Carla and test field

<div  align="center">  
    <img src="./demo/testMap.png" style="zoom:100%;" />
</div>

### Mapping with RoadSegDict

<div  align="center">  
    <img src="./demo/Map_RoadSegDict.png" style="zoom:100%;" />
</div>

### Examples of node position prediction and node state classification

<div  align="center">
    <img src="./demo/node_position.png" style="zoom:100%;" />
</div>

<div  align="center">
    <img src="./demo/node_state.png" style="zoom:100%;" />
</div>

## Proposed Mapping Dataset

Comming soon...

## Citation

If you find this project useful in your research, please consider citing us.  

```
@article{tian2022vision,
  title={Vision-based mapping of lane semantics and topology for intelligent vehicles},
  author={Tian, Wei and Ren, Xiaozhou and Yu, Xianwang and Wu, Mingzhi and Zhao, Wenbo and Li, Qiaosen},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={111},
  pages={102851},
  year={2022},
  publisher={Elsevier}
}
```







