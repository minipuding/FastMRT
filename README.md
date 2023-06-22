## FastMRT

In the realm of high-intensity focused ultrasound (HIFU) therapy, magnetic resonance imaging (MRI) is an indispensable tool for monitoring the temperature changes in biological tissue. However, the relatively slow imaging speed of MRI not only exacerbates patient discomfort, but also introduces additional risks during the course of treatment.

Accelerating Magnetic Resonance Imaging (MRI) by acquiring fewer measurements has the potential to reduce medical costs, minimize stress to patients and make MR imaging possible in applications where it is currently prohibitively slow or expensive.

This repository contains convenient PyTorch data loaders, subsampling functions, evaluation metrics, and reference implementations of simple baseline methods.

## Installing from Source

### Prerequisites

What things you need to install the software and how to install them

```
git clone https://github.com/minipuding/FastMRT.git
cd FastMRT
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
pip install -e .
```

## Running

Run `main.py` can laugch the training or testing processing.

```
python main.py --net $NET --stage $STAGE --gpus $GPUS 
```

where the \$NET is the net names choose from {` zf`,  `casnet`, `cunet`, `swtnet`, `runet`, `resunet`}, \$STAGE is processing stage choosing from {`train`, `train-test`, `test`}, and \$GPUS is the gpu ranks like `--gpus 0` or using multiple gpus as `--gpus 0,1`.

You can also use `-os` to use the source dataset only without using the dataset generate from diffusion model. And use `-nul` to assign not using logger. We only support wandb logger at this version.

Other hyperparameters about models and datasets/dataloader can be modified at `./configs`.

## Authors

* **Sijie Xu** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
