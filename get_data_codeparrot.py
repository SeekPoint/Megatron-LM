from datasets import load_dataset
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")

# train_data = load_dataset('/share/codeparrot-clean-train', split='train')
train_data = load_dataset(r'C:/codeparrot-clean-train', split='train')
train_data.to_json("codeparrot_data.json", lines=True)

