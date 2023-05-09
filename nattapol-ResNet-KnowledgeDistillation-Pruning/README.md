# ResNet + Tolerant after Talented
- Knowledge Distillation
- Cosine Similarity Pruning

## Requirements
see ```requirements.txt```

## File directories
```bash
|-- README.md
|-- blazeface
|   |-- __init__.py
|   |-- anchors.npy
|   |-- anchorsback.npy
|   |-- blazeface.pth
|   |-- blazeface.py
|   `-- blazefaceback.pth
|-- dataset
|   |-- Columbia Gaze Data Set (to be generated)
|   `-- MPIIFaceGaze.h5 (to be generated)
|-- pretrained
|   |-- ResNet10+.pt
|   |-- ResNet10+P.pt
|   |-- ResNet10.pt
|   |-- ResNet18+.pt
|   |-- ResNet18+P.pt
|   `-- ResNet18.pt
|-- results (to be generated)
|-- setup.sh
|-- train+prune.py
|-- train.py
|-- train.sh
`-- utils
    |-- __init__.py
    |-- augmentation.py
    |-- dataset.py
    |-- loss
    |   |-- __init__.py
    |   |-- __pycache__
    |   |-- angular.py
    |   |-- distillation.py
    |   `-- feature.py
    |-- models
    |   |-- __init__.py
    |   |-- resnet.py
    |   `-- simvit.py
    |-- preprocess_mpiifacegaze.py
    |-- testhelper.py
    |-- trainhelper.py
    `-- weight_prune.py
```

## Load Dataset and Train the model
```bash
chmod +x setup.sh
chmod +x train.sh
./setup
./train
```

## Results
### Benchmark
|            Model             |      Columbia Gaze     | MPIIGaze |
|:----------------------------:|:----------------------:|:--------:|
| ResNet10 (wide)              |          4.00          |   4.97   |
| ResNet10 (wide) + KD         |          4.02          |   4.87   |
| ResNet10 (wide) + KD + Prune |          3.98          |   4.81   |
| ResNet18                     |          4.11          |   5.26   |
| ResNet18 + KD                |          4.10          |   5.26   |
| ResNet18 + KD + Prune        |          3.85          |   4.64   |

### Cross-dataset Performance - [train/test]
| Model                        | [Columbia Gaze/ MPIIGaze] | [MPIIGaze/Columbia Gaze] |
|------------------------------|---------------------------|--------------------------|
| ResNet10 (wide)              | 11.31                     | 12.96                    |
| ResNet10 (wide) + KD         | 11.2                      | 12.01                    |
| ResNet10 (wide) + KD + Prune | 11.41                     | 12.71                    |
| ResNet18                     | 11.22                     | 12.55                    |
| ResNet18 + KD                | 10.87                     | 12.26                    |
| ResNet18 + KD + Prune        | 11.14                     | 12.74                    |

### Inference Speed
|      Model      |  Params |  FLOPS  | Throughput (frames per second) |
|:---------------:|:-------:|:-------:|:------------------------------:|
| ResNet10 (wide) |  3.19 M | 32.45 G |              14.6              |
|     ResNet18    | 11.17 M |  1.75 G |              13.2              |

*tested on M1 chip CPU*

## Video test
```bash
python video_test.py --proj_path=$PWD --model=resnet18 --kd --prune
```

## References
- Guo, T., Liu, Y., Zhang, H., Liu, X., Kwak, Y., In Yoo, B. & Choi, C. (2019). [A generalized and robust method towards practical gaze estimation on smart phone. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops](http://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/Guo_A_Generalized_and_Robust_Method_Towards_Practical_Gaze_Estimation_on_ICCVW_2019_paper.html).
- Furlanello, T., Lipton, Z., Tschannen, M., Itti, L., & Anandkumar, A. (2018, July). [Born again neural networks](http://proceedings.mlr.press/v80/furlanello18a.html). In International Conference on Machine Learning (pp. 1607-1616). PMLR
- Zhang, L., & Ma, K. (2021, May). [Improve object detection with feature-based knowledge distillation: Towards accurate and efficient detectors](https://openreview.net/forum?id=uKhGRvM8QNH). In International Conference on Learning Representations.
