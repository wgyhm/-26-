source /mnt/nas/gezuhao/zhengyuqian/starvla/.venv/bin/activate

python -m jupyterlab \
  /mnt/nas/gezuhao/zhengyuqian/computer_vision/assignment1/Assignment1_image_classifier/image-classifier.ipynb \
  --allow-root --no-browse

| 网络 | 单分支参数量 | 公开准确率参考 | 适合参考什么 |
|---|---:|---:|---|
| WRN-28-2 | 约 1.5M | 约 74.2% | 残差结构，容易理解和改造 |
| DenseNet-BC-100，增长率 12 | 约 0.8M | 77.73% | 参数利用率高，但特征拼接较多 |
| CIFAR 版 MobileNetV2 | 约 2.35M | 74.34% | 轻量卷积结构 |