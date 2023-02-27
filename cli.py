from argparse import ArgumentParser


class FastmrtCLI:

    @staticmethod
    def dir_cli(parser: ArgumentParser, dir_cfg: dict):
        parser.add_argument('--data_dir', type=str, default=dir_cfg["DATA_DIR"],
                            help="(str optional) Root path of dataset")
        parser.add_argument('--pt_data_dir', type=str, default=dir_cfg["PT_DATA_DIR"],
                            help="(str optional) Root path of pre-train dataset")
        parser.add_argument('--log_dir', type=str, default=dir_cfg["LOG_DIR"],
                            help="(str optional) Path of log saving, default is '../log'")
        parser.add_argument('--prf_config_dir', type=str, default=dir_cfg["PRF_CONFIG_DIR"],
                            help="(str optional) Path of prf header saving, default is './configs/prf.yaml'")
        parser.add_argument('--runet_config_dir', type=str, default=dir_cfg["RUNET_CONFIG_DIR"],
                            help="(str optional) Path of r-unet config saving, default is './configs/runet.yaml'")
        parser.add_argument('--casnet_config_dir', type=str, default=dir_cfg["CASNET_CONFIG_DIR"],
                            help="(str optional) Path of casnet config saving, default is './configs/casnet.yaml'")
        parser.add_argument('--rftnet_config_dir', type=str, default=dir_cfg["RFTNET_CONFIG_DIR"],
                            help="(str optional) Path of rftnet config saving, default is './configs/rftnet.yaml'")
        parser.add_argument('--kdnet_config_dir', type=str, default=dir_cfg["KDNET_CONFIG_DIR"],
                            help="(str optional) Path of rftnet config saving, default is './configs/kdnet.yaml'")
        return parser


    @staticmethod
    def prf_cli(parser: ArgumentParser, prf_cfg: dict):
        parser.add_argument('--b0', type=float, default=prf_cfg["B0"],
                            help="(float optional) Main magnetic field strength (T), default is 3")
        parser.add_argument('--gamma', type=float, required=False, default=prf_cfg["GAMMA"],
                            help="(float fixed) Gyromagnetic ratio of hydrogen atom, 42.576MHz/T. DON'T MODIFY IT!")
        parser.add_argument('--alpha', type=float, default=prf_cfg["ALPHA"])
        parser.add_argument('--te', type=float, default=prf_cfg["TE"],
                            help="(float optional) Echo time of MRI, default is 12ms.")
        return parser

    @staticmethod
    def runet_cli(parser: ArgumentParser, config: dict):
        # obtain sub configs
        sf_cfg = None
        if parser.parse_args().stage == "train" or parser.parse_args().stage == "fine-tune":
            sub_config = config["DIRECT"]
            ft_config = config["FINE_TUNE"]["MODEL"]
        elif parser.parse_args().stage == "pre-train":
            sub_config = config["PRETRAIN"]
            sf_cfg = sub_config["SF"]
        else:
            raise ValueError("stage must be one of ``train``, ``pre-train``, ``fine-tune``, ``test``")
        data_cfg = sub_config["DATA"]
        model_cfg = sub_config["MODEL"]
        log_cfg = sub_config["LOG"]
        augs_cfg = sub_config["AUGS"]


        # dataset configs
        parser.add_argument('--data_format', type=str, default=data_cfg["DATA_FORMAT"],
                            help="(str optional) One of 'CF'(Complex Float), 'RF'(Real Float), 'TM'(Temperature Map)"
                                 " and 'AP'(Amplitude & Phase), default is 'RF'")
        parser.add_argument('--batch_size', type=int, default=data_cfg["BATCH_SIZE"],
                            help="(int optional) Batch size of dataset")
        parser.add_argument('--sampling_mode', type=str, default=data_cfg["SAMPLING_MODE"],
                            help="(str optional) Sampling mode is one of ``RANDOM`` and ``EQUISQUARE``,"
                                 " default is ``RANDOM``")
        parser.add_argument('--acceleration', type=int, default=data_cfg["ACCELERATION"],
                            help="(int request) acceleration of fastMRT")
        parser.add_argument('--center_fraction', type=float, default=data_cfg["CENTER_FRACTION"],
                            help="(float request) center fraction of mask")
        parser.add_argument('--resize_size', type=int, default=data_cfg["RESIZE_SIZE"], nargs='+',
                            help="(tuple optional) Resize size of input image, default is (256, 256)")
        parser.add_argument('--resize_mode', type=str, default=data_cfg["RESIZE_MODE"],
                            help="(str optional) Resize mode is one of ``on_image`` and ``on_kspace``,"
                                 "default is ``on_kspace``")
        # model configs
        parser.add_argument('--in_channels', type=int, default=model_cfg["IN_CHANNELS"],
                            help="(int optional) Input channels of unet model, default is 2")
        parser.add_argument('--out_channels', type=int, default=model_cfg["OUT_CHANNELS"],
                            help="(int optional) Output channels of unet model, default is 2")
        parser.add_argument('--base_channels', type=int, default=model_cfg["BASE_CHANNELS"],
                            help="(int optional) Base channels of unet model, which doubles channels base on it,"
                                 " default is 32")
        parser.add_argument('--level_num', type=int, default=model_cfg["LEVEL_NUM"],
                            help="(int, optional) The level num (depth) of unet model, default is 4")
        parser.add_argument('--drop_prob', type=float, default=model_cfg["DROP_PROB"],
                            help="(float, optional) Probability of dropout method, default is 0.0")
        parser.add_argument('--leakyrelu_slope', type=float, default=model_cfg["LEAKYRELU_SLOPE"],
                            help="(float, optional) leaky relu slope, default is 0.4")
        parser.add_argument('--last_layer_with_act', type=bool, default=model_cfg["LAST_LAYER_WITH_ACT"],
                            help="(bool, optional) last layer use activation function, default is False")
        if parser.parse_args().stage == "fine-tune":
            parser.add_argument('--lr', type=float, default=ft_config["LR"],
                                help="(float, optional) Learning rate for fine-tune, default is 1e-7")
            parser.add_argument('--model_dir', type=str, default=ft_config["MODEL_DIR"],
                                help="(str, optional) Pre-trained model path")
        else:
            parser.add_argument('--lr', type=float, default=model_cfg["LR"],
                                help="(float, optional) Learning rate, default is 1e-3")
        parser.add_argument('--lr_step_size', type=int, default=model_cfg["LR_STEP_SIZE"],
                            help="(int optional) Learning rate step size, default is 40")
        parser.add_argument('--lr_gamma', type=float, default=model_cfg["LR_GAMMA"],
                            help="(float optional) Learning rate gamma decay, default is 0.1")
        parser.add_argument('--weight_decay', type=float, default=model_cfg["WEIGHT_DECAY"],
                            help="(float optional) Parameter for penalizing weights norm. Defaults is 0.0")
        parser.add_argument('--max_epochs', type=float, default=model_cfg["MAX_EPOCHS"],
                            help="(int optional) The max epoch for training, default is 150")
        # log_configs
        parser.add_argument('--log_name', type=str, default=log_cfg["LOG_NAME"],
                            help="(str optional) Directory name of log, default is empty")
        parser.add_argument('--log_images_frame_idx', type=int, default=log_cfg["LOG_IMAGES_FRAME_IDX"],
                            help="(int, optional) Images and tmaps with this frame index for saving, default is 6")
        parser.add_argument('--log_images_freq', type=int, default=log_cfg["LOG_IMAGES_FREQ"],
                            help="(int, optional) To control the frequency of saving images' log")
        parser.add_argument('--tmap_patch_rate', type=int, default=log_cfg["TMAP_PATCH_RATE"],
                            help="(int optional) The patch size is one over ``tmap_patch_rate`` of source image, "
                                 "default is 12")
        parser.add_argument('--tmap_max_temp_thresh', type=int, default=log_cfg["TMAP_MAX_TEMP_THRESH"],
                            help="(int optional) When max temperature of tmap patch is over ``tmap_max_temp_thresh``,"
                                 " we recode the max temperature error, default is 45(℃)")
        parser.add_argument('--tmap_ablation_thresh', type=int, default=log_cfg["TMAP_ABLATION_THRESH"],
                            help="(int optional) When temperature is over ``TMAP_ABLATION_THRESH``,"
                                 " we regard the issue is ablated, default is 57(℃)")
        # augs config
        parser.add_argument('--ap_shuffle', type=bool, default=augs_cfg["AP_SHUFFLE"],
                            help="")
        parser.add_argument('--union', type=bool, default=augs_cfg["UNION"],
                            help="")
        parser.add_argument('--objs', type=str, default=augs_cfg["OBJS"], nargs='+',
                            help="")
        parser.add_argument('--ap_logic', type=str, default=augs_cfg["AP_LOGIC"],
                            help="")
        parser.add_argument('--augs_list', type=str, default=augs_cfg["AUGS_LIST"], nargs='+',
                            help="")
        parser.add_argument('--compose_num', type=int, default=augs_cfg["COMPOSE_NUM"],
                            help="")
        if sf_cfg is not None:
            parser.add_argument('--sf_type', type=str, default=sf_cfg["SF_TYPE"],
                                help="(str, optional) Simulated focus type, one of ``gaussian`` and ``kwave``")
            parser.add_argument('--sf_frame_num', type=int, default=sf_cfg["SF_FRAME_NUM"],
                                help="(int, optional) Simulated focus frame number")
            parser.add_argument('--sf_cooling_time_rate', type=float, default=sf_cfg["SF_COOLING_TIME_RATE"],
                                help="(float optional) Cooling time over simulated sequence time")
            parser.add_argument('--sf_center_crop_size', type=int, default=sf_cfg["SF_CENTER_CROP_SIZE"],
                                help="center crop size")
            parser.add_argument('--sf_random_crop_size', type=int, default=sf_cfg["SF_RANDOM_CROP_SIZE"],
                                help="random crop size")
            parser.add_argument('--sf_max_delta_temp', type=float, default=sf_cfg["SF_MAX_DELTA_TEMP"],
                                help="max delta temperature")
        return parser

    @staticmethod
    def casnet_cli(parser: ArgumentParser, config: dict):

        # obtain sub configs
        data_cfg = config["DATA"]
        model_cfg = config["MODEL"]
        log_cfg = config["LOG"]

        # dataset configs
        parser.add_argument('--data_format', type=str, default=data_cfg["DATA_FORMAT"],
                            help="(str optional) One of 'CF'(Complex Float), 'RF'(Real Float), 'TM'(Temperature Map)"
                                 " and 'AP'(Amplitude & Phase), default is 'RF'")
        parser.add_argument('--batch_size', type=int, default=data_cfg["BATCH_SIZE"],
                            help="(int optional) Batch size of dataset")
        parser.add_argument('--sampling_mode', type=str, default=data_cfg["SAMPLING_MODE"],
                            help="(str optional) Sampling mode is one of ``RANDOM`` and ``EQUISQUARE``,"
                                 " default is ``RANDOM``")
        parser.add_argument('--acceleration', type=int, default=data_cfg["ACCELERATION"],
                            help="(int request) acceleration of fastMRT")
        parser.add_argument('--center_fraction', type=float, default=data_cfg["CENTER_FRACTION"],
                            help="(float request) center fraction of mask")
        parser.add_argument('--resize_size', type=tuple, default=data_cfg["RESIZE_SIZE"],
                            help="(tuple optional) Resize size of input image, default is (256, 256)")
        parser.add_argument('--resize_mode', type=str, default=data_cfg["RESIZE_MODE"],
                            help="(str optional) Resize mode is one of ``on_image`` and ``on_kspace``,"
                                 "default is ``on_kspace``")
        # model configs
        parser.add_argument('--in_channels', type=int, default=model_cfg["IN_CHANNELS"],
                            help="(int optional) Input channels of unet model, default is 2")
        parser.add_argument('--out_channels', type=int, default=model_cfg["OUT_CHANNELS"],
                            help="(int optional) Output channels of unet model, default is 2")
        parser.add_argument('--base_channels', type=int, default=model_cfg["BASE_CHANNELS"],
                            help="(int optional) Base channels of unet model, which doubles channels base on it,"
                                 " default is 32")
        parser.add_argument('--res_block_num', type=int, default=model_cfg["RES_BLOCK_NUM"],
                            help="(int, optional) The number of ResBlock, default is 5")
        parser.add_argument('--res_conv_ksize', type=int, default=model_cfg["RES_CONV_KSIZE"],
                            help="(int, optional) The kernel size of convolution layer in ResBlock, default is 3")
        parser.add_argument('--res_conv_num', type=int, default=model_cfg["RES_CONV_NUM"],
                            help="(int, optional) The number of convolution layers in each ResBlock, default is 5")
        parser.add_argument('--drop_prob', type=float, default=model_cfg["DROP_PROB"],
                            help="(float, optional) Probability of dropout method, default is 0.0")
        parser.add_argument('--leakyrelu_slope', type=float, default=model_cfg["LEAKYRELU_SLOPE"],
                            help="(float, optional) leaky relu slope, default is 0.4")
        parser.add_argument('--lr', type=float, default=model_cfg["LR"],
                            help="(float, optional) Learning rate, default is 1e-3")
        parser.add_argument('--lr_step_size', type=int, default=model_cfg["LR_STEP_SIZE"],
                            help="(int optional) Learning rate step size, default is 40")
        parser.add_argument('--lr_gamma', type=float, default=model_cfg["LR_GAMMA"],
                            help="(float optional) Learning rate gamma decay, default is 0.1")
        parser.add_argument('--weight_decay', type=float, default=model_cfg["WEIGHT_DECAY"],
                            help="(float optional) Parameter for penalizing weights norm. Defaults is 0.0")
        parser.add_argument('--max_epochs', type=float, default=model_cfg["MAX_EPOCHS"],
                            help="(int optional) The max epoch for training, default is 150")
        # log_configs
        parser.add_argument('--log_name', type=str, default=log_cfg["LOG_NAME"],
                            help="(str optional) Directory name of log, default is empty")
        parser.add_argument('--log_images_frame_idx', type=int, default=log_cfg["LOG_IMAGES_FRAME_IDX"],
                            help="(int, optional) Images and tmaps with this frame index for saving, default is 6")
        parser.add_argument('--log_images_freq', type=int, default=log_cfg["LOG_IMAGES_FREQ"],
                            help="(int, optional) To control the frequency of saving images' log")
        parser.add_argument('--tmap_patch_rate', type=int, default=log_cfg["TMAP_PATCH_RATE"],
                            help="(int optional) The patch size is one over ``tmap_patch_rate`` of source image, "
                                 "default is 12")
        parser.add_argument('--tmap_max_temp_thresh', type=int, default=log_cfg["TMAP_MAX_TEMP_THRESH"],
                            help="(int optional) When max temperature of tmap patch is over ``tmap_max_temp_thresh``,"
                                 " we recode the max temperature error, default is 45(℃)")
        parser.add_argument('--tmap_ablation_thresh', type=int, default=log_cfg["TMAP_ABLATION_THRESH"],
                            help="(int optional) When temperature is over ``TMAP_ABLATION_THRESH``,"
                                 " we regard the issue is ablated, default is 57(℃)")

        return parser

    @staticmethod
    def rftnet_cli(parser: ArgumentParser, config: dict):
        # obtain sub configs
        data_cfg = config["DATA"]
        model_cfg = config["MODEL"]
        log_cfg = config["LOG"]

        # dataset configs
        parser.add_argument('--data_format', type=str, default=data_cfg["DATA_FORMAT"],
                            help="(str optional) One of 'CF'(Complex Float), 'RF'(Real Float), 'TM'(Temperature Map)"
                                 " and 'AP'(Amplitude & Phase), default is 'RF'")
        parser.add_argument('--batch_size', type=int, default=data_cfg["BATCH_SIZE"],
                            help="(int optional) Batch size of dataset")
        parser.add_argument('--sampling_mode', type=str, default=data_cfg["SAMPLING_MODE"],
                            help="(str optional) Sampling mode is one of ``RANDOM`` and ``EQUISQUARE``,"
                                 " default is ``RANDOM``")
        parser.add_argument('--acceleration', type=int, default=data_cfg["ACCELERATION"],
                            help="(int request) acceleration of fastMRT")
        parser.add_argument('--center_fraction', type=float, default=data_cfg["CENTER_FRACTION"],
                            help="(float request) center fraction of mask")
        parser.add_argument('--resize_size', type=tuple, default=data_cfg["RESIZE_SIZE"],
                            help="(tuple optional) Resize size of input image, default is (256, 256)")
        parser.add_argument('--resize_mode', type=str, default=data_cfg["RESIZE_MODE"],
                            help="(str optional) Resize mode is one of ``on_image`` and ``on_kspace``,"
                                 "default is ``on_kspace``")
        # model configs
        parser.add_argument('--in_channels', type=int, default=model_cfg["IN_CHANNELS"], nargs='+',
                            help="(int optional) Input channels of refnet, recnet and tnet, respectively,"
                                 " default is (2, 2, 2)")
        parser.add_argument('--out_channels', type=int, default=model_cfg["OUT_CHANNELS"], nargs='+',
                            help="(int optional) Output channels of refnet, recnet and tnet, respectively,"
                                 " default is (2, 2, 1)")
        parser.add_argument('--base_channels', type=int, default=model_cfg["BASE_CHANNELS"], nargs='+',
                            help="(int optional) Base channels of refnet, recnet and tnet, respectively, which doubles channels base on it,"
                                 " default is (32, 32, 32)")
        parser.add_argument('--level_num', type=int, default=model_cfg["LEVEL_NUM"], nargs='+',
                            help="(int, optional) The level num (depth) of refnet, recnet and tnet, respectively,"
                                 " default is (4, 4, 4)")
        parser.add_argument('--drop_prob', type=float, default=model_cfg["DROP_PROB"], nargs='+',
                            help="(float, optional) Probability of dropout method of refnet, recnet and tnet, respectively,"
                                 " default is (0.0, 0.0, 0.0)")
        parser.add_argument('--leakyrelu_slope', type=float, default=model_cfg["LEAKYRELU_SLOPE"], nargs='+',
                            help="(float, optional) leaky relu slope of refnet, recnet and tnet, respectively,"
                                 " default is (0.1, 0.1, 0.1)")
        parser.add_argument('--last_layer_with_act', type=bool, default=model_cfg["LAST_LAYER_WITH_ACT"],
                            help="(bool, optional) last layer use activation function, default is False")
        parser.add_argument('--lr', type=float, default=model_cfg["LR"],
                            help="(float, optional) Learning rate, default is 1e-3")
        parser.add_argument('--lr_step_size', type=int, default=model_cfg["LR_STEP_SIZE"],
                            help="(int optional) Learning rate step size, default is 40")
        parser.add_argument('--lr_gamma', type=float, default=model_cfg["LR_GAMMA"],
                            help="(float optional) Learning rate gamma decay, default is 0.1")
        parser.add_argument('--weight_decay', type=float, default=model_cfg["WEIGHT_DECAY"],
                            help="(float optional) Parameter for penalizing weights norm. Defaults is 0.0")
        parser.add_argument('--max_epochs', type=float, default=model_cfg["MAX_EPOCHS"],
                            help="(int optional) The max epoch for training, default is 150")
        # log_configs
        parser.add_argument('--log_name', type=str, default=log_cfg["LOG_NAME"],
                            help="(str optional) Directory name of log, default is empty")
        parser.add_argument('--log_images_frame_idx', type=int, default=log_cfg["LOG_IMAGES_FRAME_IDX"],
                            help="(int, optional) Images and tmaps with this frame index for saving, default is 6")
        parser.add_argument('--log_images_freq', type=int, default=log_cfg["LOG_IMAGES_FREQ"],
                            help="(int, optional) To control the frequency of saving images' log")
        parser.add_argument('--tmap_patch_rate', type=int, default=log_cfg["TMAP_PATCH_RATE"],
                            help="(int optional) The patch size is one over ``tmap_patch_rate`` of source image, "
                                 "default is 12")
        parser.add_argument('--tmap_max_temp_thresh', type=int, default=log_cfg["TMAP_MAX_TEMP_THRESH"],
                            help="(int optional) When max temperature of tmap patch is over ``tmap_max_temp_thresh``,"
                                 " we recode the max temperature error, default is 45(℃)")
        parser.add_argument('--tmap_ablation_thresh', type=int, default=log_cfg["TMAP_ABLATION_THRESH"],
                            help="(int optional) When temperature is over ``TMAP_ABLATION_THRESH``,"
                                 " we regard the issue is ablated, default is 57(℃)")

        return parser

    @staticmethod
    def kdnet_cli(parser: ArgumentParser, config: dict):
        # obtain sub configs
        public_cfg = config["PUBLIC"]
        data_tea_cfg = config["DATA_TEA"]
        data_stu_cfg = config["DATA_STU"]
        model_tea_cfg = config["MODEL_TEA"]
        model_stu_cfg = config["MODEL_STU"]
        log_cfg = config["LOG"]
        augs_cfg = config["AUGS"]

        # public configs
        parser.add_argument('--data_format', type=str, default=public_cfg["DATA_FORMAT"],
                            help="(str optional) One of 'CF'(Complex Float), 'RF'(Real Float), 'TM'(Temperature Map)"
                                 " and 'AP'(Amplitude & Phase), default is 'RF'")
        parser.add_argument('--batch_size', type=int, default=public_cfg["BATCH_SIZE"],
                            help="(int optional) Batch size of dataset")
        parser.add_argument('--sampling_mode', type=str, default=public_cfg["SAMPLING_MODE"],
                            help="(str optional) Sampling mode is one of ``RANDOM`` and ``EQUISQUARE``,"
                                 " default is ``RANDOM``")
        parser.add_argument('--in_channels', type=int, default=public_cfg["IN_CHANNELS"],
                            help="(int optional) Input channels of unet model, default is 2")
        parser.add_argument('--out_channels', type=int, default=public_cfg["OUT_CHANNELS"],
                            help="(int optional) Output channels of unet model, default is 2")
        parser.add_argument('--max_epochs', type=float, default=public_cfg["MAX_EPOCHS"],
                            help="(int optional) The max epoch for training, default is 150")
        parser.add_argument('--use_ema', type=bool, default=public_cfg["USE_EMA"],
                            help="(bool optional) ")
        parser.add_argument('--soft_label_weight', type=float, default=public_cfg["SOFT_LABEL_WEIGHT"],
                            help="(float optional)")

        # teacher dataset configs
        parser.add_argument('--acceleration_tea', type=int, default=data_tea_cfg["ACCELERATION"],
                            help="(int request) acceleration of fastMRT")
        parser.add_argument('--center_fraction_tea', type=float, default=data_tea_cfg["CENTER_FRACTION"],
                            help="(float request) center fraction of mask")
        # student dataset configs
        parser.add_argument('--acceleration_stu', type=int, default=data_stu_cfg["ACCELERATION"],
                            help="(int request) acceleration of fastMRT")
        parser.add_argument('--center_fraction_stu', type=float, default=data_stu_cfg["CENTER_FRACTION"],
                            help="(float request) center fraction of mask")

        # teacher model configs
        parser.add_argument('--base_channels_tea', type=int, default=model_tea_cfg["BASE_CHANNELS"],
                            help="(int optional) Base channels of unet model, which doubles channels base on it,"
                                 " default is 32")
        parser.add_argument('--level_num_tea', type=int, default=model_tea_cfg["LEVEL_NUM"],
                            help="(int, optional) The level num (depth) of unet model, default is 4")
        parser.add_argument('--drop_prob_tea', type=float, default=model_tea_cfg["DROP_PROB"],
                            help="(float, optional) Probability of dropout method, default is 0.0")
        parser.add_argument('--leakyrelu_slope_tea', type=float, default=model_tea_cfg["LEAKYRELU_SLOPE"],
                            help="(float, optional) leaky relu slope, default is 0.4")
        parser.add_argument('--last_layer_with_act_tea', type=bool, default=model_tea_cfg["LAST_LAYER_WITH_ACT"],
                            help="(bool, optional) last layer use activation function, default is False")
        parser.add_argument('--lr_tea', type=float, default=model_tea_cfg["LR"],
                            help="(float, optional) Learning rate, default is 1e-3")
        parser.add_argument('--weight_decay_tea', type=float, default=model_tea_cfg["WEIGHT_DECAY"],
                            help="(float optional) Parameter for penalizing weights norm. Defaults is 0.0")
        # student model configs
        parser.add_argument('--base_channels_stu', type=int, default=model_stu_cfg["BASE_CHANNELS"],
                            help="(int optional) Base channels of unet model, which doubles channels base on it,"
                                 " default is 32")
        parser.add_argument('--level_num_stu', type=int, default=model_stu_cfg["LEVEL_NUM"],
                            help="(int, optional) The level num (depth) of unet model, default is 4")
        parser.add_argument('--drop_prob_stu', type=float, default=model_stu_cfg["DROP_PROB"],
                            help="(float, optional) Probability of dropout method, default is 0.0")
        parser.add_argument('--leakyrelu_slope_stu', type=float, default=model_stu_cfg["LEAKYRELU_SLOPE"],
                            help="(float, optional) leaky relu slope, default is 0.4")
        parser.add_argument('--last_layer_with_act_stu', type=bool, default=model_stu_cfg["LAST_LAYER_WITH_ACT"],
                            help="(bool, optional) last layer use activation function, default is False")
        parser.add_argument('--lr_stu', type=float, default=model_stu_cfg["LR"],
                            help="(float, optional) Learning rate, default is 1e-3")
        parser.add_argument('--weight_decay_stu', type=float, default=model_stu_cfg["WEIGHT_DECAY"],
                            help="(float optional) Parameter for penalizing weights norm. Defaults is 0.0")
        # log_configs
        parser.add_argument('--log_name', type=str, default=log_cfg["LOG_NAME"],
                            help="(str optional) Directory name of log, default is empty")
        parser.add_argument('--log_images_frame_idx', type=int, default=log_cfg["LOG_IMAGES_FRAME_IDX"],
                            help="(int, optional) Images and tmaps with this frame index for saving, default is 6")
        parser.add_argument('--log_images_freq', type=int, default=log_cfg["LOG_IMAGES_FREQ"],
                            help="(int, optional) To control the frequency of saving images' log")
        parser.add_argument('--tmap_patch_rate', type=int, default=log_cfg["TMAP_PATCH_RATE"],
                            help="(int optional) The patch size is one over ``tmap_patch_rate`` of source image, "
                                 "default is 12")
        parser.add_argument('--tmap_max_temp_thresh', type=int, default=log_cfg["TMAP_MAX_TEMP_THRESH"],
                            help="(int optional) When max temperature of tmap patch is over ``tmap_max_temp_thresh``,"
                                 " we recode the max temperature error, default is 45(℃)")
        parser.add_argument('--tmap_ablation_thresh', type=int, default=log_cfg["TMAP_ABLATION_THRESH"],
                            help="(int optional) When temperature is over ``TMAP_ABLATION_THRESH``,"
                                 " we regard the issue is ablated, default is 57(℃)")
        # augs config
        parser.add_argument('--ap_shuffle', type=bool, default=augs_cfg["AP_SHUFFLE"],
                            help="")
        parser.add_argument('--union', type=bool, default=augs_cfg["UNION"],
                            help="")
        parser.add_argument('--objs', type=str, default=augs_cfg["OBJS"], nargs='+',
                            help="")
        parser.add_argument('--ap_logic', type=str, default=augs_cfg["AP_LOGIC"],
                            help="")
        parser.add_argument('--augs_list', type=str, default=augs_cfg["AUGS_LIST"], nargs='+',
                            help="")
        parser.add_argument('--compose_num', type=int, default=augs_cfg["COMPOSE_NUM"],
                            help="")
        return parser