# Compressed Sensing with Total Variation Minimization

We implenment the compressed sensing as fastmri do. The implementation uses the BART toolkit.

## Install BART for Linux

prepare source code:

```
git clone https://github.com/mrirecon/bart.git
```

compile source code using cmake:

```
sudo apt-get install make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev
sudo chmod -R 775 /your/bart/path
cd /your/bart/path
make
```

waitting for a while, a `bart` file would be compiled at your bart path.

## Acknowledgements

This directory contains code for compressed sensing baselines. The baselines are based on the following paper:

[ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA (M. Uecker et al., 2013)](https://doi.org/10.1002/mrm.24751)

which was used as a baseline model in

[fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ({J. Zbontar*, F. Knoll*, A. Sriram*} et al., 2018)](https://arxiv.org/abs/1811.08839)
