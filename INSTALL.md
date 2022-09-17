Последовательность установки зависимостей.

Будет изменена на более интуитивную в будущем (после избавления от зависимости `sg_benchmark`).

Требуется версия CUDA 11.0, либо другая совместимая с собранным pytorch 1.7.1 
в канале conda pytorch. В противном случае нужно собирать pytorch вручную.

```sh
conda create -n kvqa python=3.7 ipython
```

```sh
pip install ninja yacs cython matplotlib tqdm opencv-python numpy && \
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 \
-c pytorch -c conda-forge && \
conda install -c conda-forge timm einops pycocotools && \
pip install cityscapesscripts
```

```sh
git clone https://github.com/ZONT3/kvqa --recursive && \
cd kvqa/scene_graph_benchmark && \
pip install .&& \
cd ../transformers && \
pip install .
```
