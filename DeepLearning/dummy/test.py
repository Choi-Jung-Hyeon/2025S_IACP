# commandline
# python main.py --model resnet34 --dataset cifar100 --num_epochs 100 --batch_size 64

# file structure
'''
main.py
datasets.py
models/
  __init__.py
  resnet.py
  desnetnet.py
  fractalnet.py
'''
  
# models/__init__.py
def load_model(model_name):
    if model_name == "resnet34":
        return resnet.resnet34()

# main.py
import models
import datasets

def main(args):
    model = models.load_model(args.model)
    dataset = datasets.load_dataset(args.dataset)
    dataloader = ...
    optimizer = nn.optim.SGD(model.parameters(), lr=1e-3)
    for batch in dataloader:
        train()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--dataset", type=str, default="cifar10")
    args = argparser.parse_args()
    main(args)