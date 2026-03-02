def get_dataset(args, **kwargs):
    """Function to get the specific dataset

    Args:
        args (argparse.Namespace): Arguments from the command line.

    Returns:
        Dataset: The dataset object.
    """    
    if args.dimension == "2d":
        if args.dataset == "amos":
            from .dim2.dataset_amos import AMOSDataset
            return AMOSDataset(args)
    else:
        if args.dataset == "btcv":
            from .dim3.dataset_btcv import BTCVDataset
            return BTCVDataset(args)