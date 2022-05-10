# face_recognition
Python implement of face recognition methods: 1. Eigenface; 2. Fisherface; 3. LBPH

## 数据集
链接: https://pan.baidu.com/s/1D1nlIiWQ4WbnWWhu0dFo1w 提取码: 5ewc 

## 对比实验

> 图像尺寸：`(20, 20)`；
>
> 特征脸个数：`20`；

### 准确率

> 正确率：acc

| 算法\数据集 | faces94            | faces95             | faces96            | grimace |
| ----------- | ------------------ | ------------------- | ------------------ | ------- |
| Eigenface   | 0.9526143790849673 | 0.8125              | 0.8211920529801324 | 1.0     |
| Fisherface  | 0.9686985172981878 | 0.49473684210526314 | 0.9499165275459098 | 1.0     |
| LBP         | 0.9803921568627451 | 0.875               | 0.9536423841059603 | 1.0     |

### 耗时

> 单位：秒

| 算法\样本数 | faces94            | faces95            | faces96            | grimace           |
| ----------- | ------------------ | ------------------ | ------------------ | ----------------- |
| Eigenface   | 26.522611141204834 | 7.438950538635254  | 25.5718777179718   | 2.532238006591797 |
| Fisherface  | 15.553928852081299 | 5.8003761768341064 | 15.759870529174805 | 2.809000015258789 |
| LBP         | 8731.06625032425   | 1951.6611280441284 | 8556.96798324585   | 128.836838722229  |



------

## 单项实验

### Eigenface

> 图像尺寸：`(64,64)`

|         | 平均脸                                                       | 特征脸                                                       | 特征脸                                                       | 特征脸                                                       | 特征脸                                                       |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| faces94 | ![eigen_mean](Results.assets/eigen_mean-165208075584118.jpg) | ![eigen_face0](Results.assets/eigen_face0-165208074064114.jpg) | ![eigen_face1](Results.assets/eigen_face1-165208074465615.jpg) | ![eigen_face2](Results.assets/eigen_face2-165208074751616.jpg) | ![eigen_face3](Results.assets/eigen_face3-165208075001917.jpg) |
| faces95 | ![eigen_mean](Results.assets/eigen_mean-16520804572289.jpg)  | ![eigen_face0](Results.assets/eigen_face0-165208046299910.jpg) | ![eigen_face1](Results.assets/eigen_face1-165208046704211.jpg) | ![eigen_face2](Results.assets/eigen_face2-165208047056712.jpg) | ![eigen_face3](Results.assets/eigen_face3-165208047328613.jpg) |
| faces96 | ![eigen_mean](Results.assets/eigen_mean-16520804452468.jpg)  | ![eigen_face0](Results.assets/eigen_face0-16520803012583.jpg) | ![eigen_face1](Results.assets/eigen_face1-16520803069384.jpg) | ![eigen_face2](Results.assets/eigen_face2-16520803093875.jpg) | ![eigen_face3](Results.assets/eigen_face3-16520803118366.jpg) |
| grimace | ![eigen_mean](Results.assets/eigen_mean-16520801357751.jpg)  | ![eigen_face0](Results.assets/eigen_face0.jpg)               | ![eigen_face1](Results.assets/eigen_face1.jpg)               | ![eigen_face2](Results.assets/eigen_face2.jpg)               | ![eigen_face3](Results.assets/eigen_face3.jpg)               |

### Fisher

> 图像尺寸：`(64,64)`

|         | 平均脸                                                       | 特征脸                                                       | 特征脸                                                       | 特征脸                                                       | 特征脸                                                       |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| faces94 | ![mean_face_64x64](Results.assets/mean_face_64x64-165210641982645.jpg) | ![fisher_face_64x64_0](Results.assets/fisher_face_64x64_0-165210640348041.jpg) | ![fisher_face_64x64_1](Results.assets/fisher_face_64x64_1-165210640587642.jpg) | ![fisher_face_64x64_2](Results.assets/fisher_face_64x64_2-165210641141543.jpg) | ![fisher_face_64x64_3](Results.assets/fisher_face_64x64_3-165210641587944.jpg) |
| faces95 | ![mean_face_64x64](Results.assets/mean_face_64x64-165210630894140.jpg) | ![fisher_face_64x64_0](Results.assets/fisher_face_64x64_0-165210626204136.jpg) | ![fisher_face_64x64_1](Results.assets/fisher_face_64x64_1-165210626486337.jpg) | ![fisher_face_64x64_2](Results.assets/fisher_face_64x64_2-165210626761938.jpg) | ![fisher_face_64x64_3](Results.assets/fisher_face_64x64_3-165210626983139.jpg) |
| faces96 | ![mean_face_64x64](Results.assets/mean_face_64x64-165210607288031.jpg) | ![fisher_face_64x64_0](Results.assets/fisher_face_64x64_0-165210609212232.jpg) | ![fisher_face_64x64_1](Results.assets/fisher_face_64x64_1-165210610180633.jpg) | ![fisher_face_64x64_2](Results.assets/fisher_face_64x64_2-165210610387234.jpg) | ![fisher_face_64x64_3](Results.assets/fisher_face_64x64_3-165210610645835.jpg) |
| grimace | ![mean_face_64x64](Results.assets/mean_face_64x64.jpg)       | ![fisher_face_64x64_0](Results.assets/fisher_face_64x64_0.jpg) | ![fisher_face_64x64_1](Results.assets/fisher_face_64x64_1.jpg) | ![fisher_face_64x64_2](Results.assets/fisher_face_64x64_2.jpg) | ![fisher_face_64x64_3](Results.assets/fisher_face_64x64_3.jpg) |
