# Dual Branch Network for RGBW Fusion and Denoising

This is the codebase for RGBW fusion and denoising challenge of MIPI22 of team jzsherlock.

## Brief Summary of Codebase

This codebase contains a `BasicSR` folder which is modified from the original [BasicSR](https://github.com/XPixelGroup/BasicSR) project. This codebase is mainly based on BasicSR framework. Folder `codebase_local` contains the necessary source code to re-run the training and inference code, including the archs of network, training settings and configures, and other utiliy codes. `experiments` is the place for final model and corresponding .log file which can be checked. The `requirements.txt` contains the possible required packages to be installed and their versions. Note that the precise versions given here means they are tested and can work well, packages with similar but not the same version may work well too if you cannot install the same version.

## Installation and Preparation

First, install the required packages in `requirements.txt` by:

```shell
pip install -r requirements.txt
```

Then install BasicSR from the local `BasicSR` folder, using command:

```shell
cd BasicSR
python setup.py develop
```

note that the BasicSR must be installed using the local source code, instead of using pip or original github source, because there are some modifications made in the local version. 

After the installation, please assure the torch and basicsr can be normally imported in python, and torch can use cuda. Then you can re-implement the result given in `experiments` folder.

## Re-run Training or Inference

In order to re-implement the training, you can use the `start_train.sh` with the arguments for config (.yml format) and gpu id:
(need to modify the .yml in `option/train` to make the dataroot comply with your dataset location)

```shell
cd codebase_local
sh start_train.sh options/train/012_clipnorm_bslndualv2_nobn_bs32ps160_msteplr_l1loss_trainall.yml 0  # re-run the config for final model in GPU 0
```

to inference for the test dataset with the final model without training (or after training), run the following command (modify test .yml dataroot as stated above):

```shell
python test_rgbw.py -opt options/test/finaltest_012_clipnorm_bslndualv2_nobn_bs32ps160_msteplr_l1loss_trainall_tta.yml
```

The processed results are saved `results` folder in the parent dir of `codebase_local`. in the subfolder with the name same as written in the final test configure: `012_clipnorm_bslndualv2_nobn_bs32ps160_msteplr_l1loss_trainall`.


## Contact
The results can be correctly re-produced or re-trained following the above steps for environment setups and bash commands. If there are some problems or questions about the re-implementation, please contact: jzsherlock@163.com. Thanks~

