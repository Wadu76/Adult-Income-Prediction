# Adult-Income-Prediction

本指南主要介绍如何使用Anaconda虚拟环境配置并运行成年人收入预测代码。

## 环境准备

### 步骤1：创建虚拟环境

打开Anaconda Prompt并输入以下命令创建Python 3.13的虚拟环境：

```bash
conda create -n adult-env python=3.13
```

根据提示输入`y`确认安装。

### 步骤2：激活虚拟环境

```bash
conda activate adult-env
```

### 步骤3：配置国内镜像源（推荐）

为加速包的下载，可以配置清华大学镜像源：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

### 步骤4：安装依赖包

```bash
conda install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost
```

## 运行代码

### 方法一：直接运行Python脚本

定位到代码文件夹并运行脚本：

```bash
# 将路径替换为您的实际代码路径
cd My_User/adult-income-project/scripts
python train.py
```

### 方法二：使用Jupyter Notebook

1. 定位到代码文件夹后启动Jupyter Notebook：

```bash
jupyter notebook
```

2. 在Jupyter Notebook界面中：
   - 点击右上角的"New"，选择"Python 3"创建新笔记本
   - 或点击"Upload"上传现有的.ipynb文件
   - 或将代码复制到笔记本单元格中运行

## 注意事项

- 本指南适用于Anaconda虚拟环境和Jupyter Notebook，当然你也可以不用虚拟环境就按本文的依赖库下载方法安装，但推荐使用虚拟环境以避免依赖冲突。
- 请根据实际情况调整文件路径！