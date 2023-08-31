# Few-Shot

The aim for this repository is to record an Pytorch Implementation of the classic work of the few shot learning.

This code include the implementation of 
- Matching Network
- Prototypical Network
- MAML

You can test the different methods by running the 'main.py ' in the corresponding folder

This project is written in Python 3.9 and Pytorch and assumes you have a GPU.

This Project is based on this repo: https://github.com/oscarknagg/few-shot

## DataSet

Edit the DATA_PATH variable in config.py to the location where you store the Omniglot and miniImagenet datasets.

After acquiring the data and running the setup scripts your folder structure should look like
```
DATA_PATH/
    Omniglot/
        images_background/
        images_evaluation/
    miniImageNet/
        images_background/
        images_evaluation/
```
Omniglot dataset. Download from https://github.com/brendenlake/omniglot/tree/master/python, place the extracted files into DATA_PATH/Omniglot_Raw and run scripts/prepare_omniglot.py

miniImageNet dataset. Download files from https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view, place in data/miniImageNet/images and run scripts/prepare_mini_imagenet.py

最后附一张元学习任务的数据包含关系图：

![Pasted image 20230820141116](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/newBlog/Pasted%20image%2020230820141116.png)