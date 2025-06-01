# Accelerated Proton Resonance Frequency-Based Magnetic Resonance Thermometry by Optimized Deep Learning Method

[![arXiv](https://img.shields.io/badge/arXiv-2407.03308-b31b1b.svg)](https://arxiv.org/abs/2407.03308)
[![GitHub](https://img.shields.io/badge/GitHub-minipuding/FastMRT-181717.svg)](https://github.com/minipuding/FastMRT)
[![Website](https://img.shields.io/badge/Website-FastMRT-blue.svg)](https://fastmrt.github.io/)
[![Journal](https://img.shields.io/badge/Journal-Medical_Physics-0080FF.svg)](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.17909)

Sijie Xu*, Shenyan Zong*, Chang-Sheng Mei, Guofeng Shen, Yueran Zhao, He Wang
> \* *Equal contribution*

## FastMRT

In the realm of high-intensity focused ultrasound (HIFU) therapy, magnetic resonance imaging (MRI) is an indispensable tool for monitoring the temperature changes in biological tissue. However, the relatively slow imaging speed of MRI not only exacerbates patient discomfort, but also introduces additional risks during the course of treatment.

Accelerating Magnetic Resonance Imaging (MRI) by acquiring fewer measurements has the potential to reduce medical costs, minimize stress to patients and make MR imaging possible in applications where it is currently prohibitively slow or expensive.

This repository contains convenient PyTorch data loaders, subsampling functions, evaluation metrics, and reference implementations of simple baseline methods.

## Installing from Source

### Prerequisites

What things you need to install the software and how to install them

```bash
git clone https://github.com/minipuding/FastMRT.git  
cd FastMRT  
```

### Installing

A step by step series of examples that tell you how to get a development env running

```bash
pip install -e .  
```

## Running

Run `main.py` can launch the training or testing processing.

```bash
python main.py --net $NET --stage $STAGE --gpus $GPUS  
```

where `$NET` is the net names choose from {`zf`, `casnet`, `cunet`, `swtnet`, `runet`, `resunet`}, `$STAGE` is processing stage choosing from {`train`, `train-test`, `test`}, and `$GPUS` is the GPU ranks like `--gpus 0` or using multiple GPUs as `--gpus 0,1`.

You can also use `-os` to use the source dataset only without using the dataset generated from diffusion model. Use `-nul` to disable the logger (we only support Weights & Biases (`wandb`) logger in this version).

Other hyperparameters about models and datasets/dataloader can be modified at `./configs`.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* [FastMRI](https://fastmri.med.nyu.edu/)
* [SwinIR](https://github.com/JingyunLiang/SwinIR)


## Citation

If you use this code or dataset, please cite our work:

```bibtex
@article{xu2025accelerated,  
  title={Accelerated Proton Resonance Frequency-Based Magnetic Resonance Thermometry by Optimized Deep Learning Method},  
  author={Sijie Xu, Shenyan Zong, Chang-Sheng Mei, Guofeng Shen, Yueran Zhao, He Wang},  
  journal={Medical Physics},  
  year={2025},  
  publisher={Wiley}  
}  
```
