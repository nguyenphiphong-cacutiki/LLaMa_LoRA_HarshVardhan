from datasets import load_dataset

dataset_name = "mlabonne/guanaco-llama2-1k"
train_dataset = load_dataset(dataset_name,split = "train")
print(train_dataset)

# load from csv
val_dataset = load_dataset('csv', data_files='data/demo2.csv', split="train")
print(val_dataset)
# for i in range(10):
#     print(train_dataset[i])