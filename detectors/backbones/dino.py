from urllib.request import urlopen
from PIL import Image
import timm

class Dino:
	def __init__(self, model="dino", device):

		self.device = device
		
		self.model = timm.create_model(
			'vit_base_patch14_dinov2.lvd142m',
			pretrained=True,
			num_classes=0,  # remove classifier nn.Linear
		)
	def embed(self, img):
		self.model = self.model.to(self.device)	
		self.model = self.model.eval()

		# get model specific transforms (normalization, resize)
		data_config = timm.data.resolve_model_data_config(self.model)
		transforms = timm.data.create_transform(**data_config, is_training=False)

		output = self.model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

		# or equivalently (without needing to set num_classes=0)

		output = self.model.forward_features(transforms(img).unsqueeze(0))
		# output is unpooled, a (1, 1370, 768) shaped tensor

		output = self.model.forward_head(output, pre_logits=True)
		# output is a (1, num_features) shaped tensor

		return output

