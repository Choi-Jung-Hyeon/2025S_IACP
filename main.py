import models
import datasets
import argparse

def main(args):
	'''
	model = models.load_model(args.model)
	dataset = datasets.load_dataset(args.dataset)
	dataloader = ...
	optimizer = nn.optim.SGD(model.parameters(), lr=1e-3)
	for batch in dataloader:
		train()
	'''
	print("Hello World")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="resnet34")
    argparser.add_argument("--dataset", type=str, default="cifar10")
    argparser.add_argument("--epoch", type=int, default="100")
    argparser.add_argument("--batch", type=int, default="64")
    args = argparser.parse_args()
    main(args)
