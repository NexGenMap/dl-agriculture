import argparse, sys

parser = argparse.ArgumentParser(
	description="U-net: A convolutional neural network to semantic segmentation",
	epilog="@NextGenMap",
	formatter_class=argparse.RawDescriptionHelpFormatter,
	add_help=True,
)


parser_mode = argparse.ArgumentParser(add_help=False)

parser.add_argument("--mode", help="selecte mode", type=str, choices=["generate", "train", "evaluate", "predict"])
parser_mode.add_argument("--mode", help="selecte mode", type=str, choices=["generate", "train", "evaluate", "predict"])

parser_generate = argparse.ArgumentParser(parents=[parser_mode])
parser_generate.add_argument("-i", "--image", type=str)
parser_generate.add_argument("-l", "--labels",type=str)
parser_generate.add_argument("-o", "--output", type=str)
#optional
parser_generate.add_argument("-cs", "--chip_size", type=int, default=256)
parser_generate.add_argument("-ch", "--channels",  type=int, default=4)
parser_generate.add_argument("-gr", "--grids",  type=int, default=1, choices=[1, 2, 3])
parser_generate.add_argument("-r", "--rotate", type=bool, default=False)
parser_generate.add_argument("-f", "--flip", type=bool, default=False)

parser_train = argparse.ArgumentParser(parents=[parser_mode])
parser_train.add_argument("-ta", "--train", type=str)
parser_train.add_argument("-te", "--test", type=str)
parser_train.add_argument("-e", "--epochs", type=int, default=5)
parser_train.add_argument("-b", "--batch_size", type=int, default=5)
parser_train.add_argument("-md", "--model_dir", type=str, default="logs")
parser_train.add_argument("-m", "--model", type=str, default="unet")
parser_train.add_argument("-c", "--classes", type=int, default=2)

parser_evaluate = argparse.ArgumentParser(parents=[parser_mode])
parser_evaluate.add_argument("-e", "--evaluate", type=str)
parser_evaluate.add_argument("-b", "--batch_size", type=int, default=5)
parser_evaluate.add_argument("-md", "--model_dir", type=str, default="logs")
parser_evaluate.add_argument("-m", "--model", type=str, default="unet")
parser_evaluate.add_argument("-c", "--classes", type=int, default=2)

parser_predict = argparse.ArgumentParser(parents=[parser_mode])
parser_predict.add_argument("-i", "--input", type=str)
parser_predict.add_argument("-o", "--output", type=str)
parser_predict.add_argument("-cs", "--chip_size", type=int, default=256)
parser_predict.add_argument("-ch", "--channels", type=int, default=4)
parser_predict.add_argument("-gr", "--grids", type=int, default=1, choices=[1, 2, 3])
parser_predict.add_argument("-b", "--batch_size", type=int, default=5)
parser_predict.add_argument("-md", "--model_dir", type=str, default="logs")
parser_predict.add_argument("-m", "--model", type=str, default="unet")
parser_predict.add_argument("-c", "--classes", type=int, default=2)
