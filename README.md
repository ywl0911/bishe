---
date: 2017-02-10 22:24
status: public
tags: 'dl, tensorflow, theano'
title: Ubuntu下深度学习库tensorflow和theano环境的配置
---

近几年关于深度学习甚是火热，自己也尝试着搭建了一下常用的环境。实验室主要的硬件配置如下：
显卡：GTX 980、Tesla K40
cpu：Intel Xeon
内存：32G
OS：Ubuntu 14.04 LTS
##1、改源
装完Ubuntu系统后要要用装一些包和依赖，首先改源，是下载的速度变快，这里改科大的源。
###改Ubuntu的源：
```python
sudo sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list 
```
###改pip的源：
```shell
mkdir ~/.pip
vi ~/.pip/pip.conf
[global]
index-url = https://mirrors.ustc.edu.cn/pypi/web/simple
format = columns
```
最后更新一下软件列表
```shell
sudo apt-get update
```
##2、安装NumPy和SciPy
在Ubuntu中这些软件包必须在安装Numpy和Scipy之前安装，否则numpy和scipy就会安装失败。
###安装BLAS, LAPACK, ATLAS
```shell
sudo apt-get install -y libopenblas-dev liblapack-dev libatlas-base-dev gcc g++ git gfortran
```
###安装 python-dev python-pip python-nose
```shell
sudo apt-get install python3-dev python3-pip python3-nose 
```
###安装Numpy
```shell
sudo pip install numpy
```
安装完成之后进行测试
```shell
python -c 'import numpy; numpy.test()'
```
###安装Scipy
```shell
sudo pip install scipy
```
安装完成之后进行测试
```shell
python -c 'import scipy; scipy.test()'
```
##3、安装Cuda
cuda可以根据系统的版本在[官网](https://developer.nvidia.com/cuda-downloads)下载。这里下载的是deb版的。
下载完后解压并安装
```shell
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64.deb
sudo apt-get update    
sudo apt-get install cuda  
```
安装完后更改环境变量
```shell
gedit ~/.bashrc
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"
export PATH=$CUDA_HOME/bin:$PATH
```
##4、安装cudnn
若只需要theano环境，可以只装cuda，若需要tensorflow还需要装cudnn。cudnn可以根据系统的版本在[官网](https://developer.nvidia.com/cudnn)下载，下载前需要先注册。
安装cudnn
```shell
tar zxvf cudnn-8.0-linux-x64-v5.1.solitairetheme8
cd cuda
sudo cp -P include/cudnn.h /usr/include
sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
sudo ldconfig
```
##5、安装theano
安装
```shell
sudo pip install theano
```
测试
```shell
python -c 'import theano; theano.test()'
```
配置gpu环境
```shell
gedit  .theanorc 
[global]
floatX=float32
device=gpu
[cuda]
root=/usr/local/cuda-8.0
```
##6、安装tensorflow
Ubuntu14.04的wheel较老，需要先更新
在[官网](https://pypi.python.org/pypi/wheel#downloads)下载wheel，安装
```shell
sudo apt-get remove python-wheel
sudo pip install wheel-0.30.0a0-py2.py3-none-any.whl
```
安装tensorflow，安装gpu版本的，python2和3不一样
```shell
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl
sudo pip3 install --upgrade $TF_BINARY_URL

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
```
测试
```shell
python3 -c 'import tensorflow'
python -c 'import tensorflow'
```

至此深度学习库tensorflow和theano在Ubuntu下面的安装都已经完成了~~
