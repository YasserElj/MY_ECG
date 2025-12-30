from datasets import load_dataset

dataset = load_dataset("ILSVRC/imagenet-1k", revision="4603483700ee984ea9debe3ddbfdeae86f6489eb", trust_remote_code=True)

dataset.save_to_disk("data/imagenet-1k-hf")