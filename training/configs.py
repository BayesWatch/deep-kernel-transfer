class Config:
    def __init__(self, args):
        self.kernel_type = args.kernel_type  # spectral' #'bncossim' #linear, rbf, spectral (regression only), matern, poli1, poli2, cossim, bncossim
        self.data_dir = {}
        self.data_dir['miniImagenet'] = './filelists/miniImagenet/'
        self.data_dir['omniglot'] = './filelists/omniglot/'
        self.data_dir['emnist'] = './filelists/emnist/'
        self.data_dir['nasdaq'] = './filelists/Nasdaq_100/nasdaq100_padding.csv'
        self.save_dir = args.save_dir

        if self.kernel_type == "nn":
            self.nn_config = {}
            if args.dataset == "sines":
                self.nn_config["input_dim"] = args.output_dim
            elif args.dataset == "nasdaq":
                self.nn_config["input_dim"] = args.output_dim
            elif args.dataset == "QMUL":
                self.nn_config["input_dim"] = 2916
            elif args.dataset == "CUB":
                self.nn_config["input_dim"] = 1600
            else:
                raise ValueError("input dim for nn kernel not known for value {}".format(args.dataset))
            self.nn_config["hidden_dim"] = 16
            self.nn_config["output_dim"] = 16
            self.nn_config["num_layers"] = 1
