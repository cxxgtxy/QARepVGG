
<h1 align="center">
Make RepVGG Greater Again: A Quantization-aware Approach （AAAI2024）
</h1>

<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-MTGV%20HuggingFace-blue.svg)](https://huggingface.co/mtgv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/Arxiv-2402.03766-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2212.01593)

</h5>



## 📸 Release

* ⏳ QARepVGG training code. Note that the implementation is already provided in [YOLOv6](https://github.com/meituan/YOLOv6) and used in [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md). Both are well known objector detectors.

## 🦙 Model Zoo


🔔 **Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses. This project is licensed permissively under the MIT license and does not impose any additional constraints. 


## 🛠️ Install

1. Clone this repository and navigate to MobileVLM folder
   ```bash
   git clone  https://github.com/cxxgtxy/QARepVGG.git
   cd QARepVGG
   ```

2. Install Package
    ```Shell
    conda create -n  QARepVGG python=3.10 -y
    conda activate QARepVGG
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## 🗝️ Quick Start
    QARepVGGBlockV2 is the default implementation, and we also provide other variants(for ablation and not recommended for use)
    We use B0 for example, which is trained for 120 epochs on ImageNet 1k dataset.
    ```Shell
    sh train_QAV2_B0.sh
    ```



## 🤝 Acknowledgments

- [RepVGG](https://github.com/DingXiaoH/RepVGG): the codebase we built upon. Thanks for their wonderful work! 👏
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation): the great open-sourced framework for segmentation! 👏

## ✏️ Reference

If you find MobileVLM or MobileLLaMA useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@inproceedings{chu2023make,
  title={Make RepVGG Greater Again: A Quantization-aware Approach},
  author={Chu, Xiangxiang and Li, Liang and Zhang, Bo},
  booktitle={AAAI},
  year={2024}
}
```




