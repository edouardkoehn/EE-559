import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
import json


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# loading dataset
tweets = pd.read_csv("data/MMHS150K/MMHS150K_with_img_text.csv")
data = tweets[["tweet_text_clean", "binary_hate", "split", "index"]]
data = data.rename(
    columns={
        "tweet_text_clean": "text",
        "binary_hate": "labels",
        "split": "split",
        "index": "index",
    }
)

# Convert the pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(data[data["split"] == "train"])
test_dataset = Dataset.from_pandas(data[data["split"] == "test"])
val_dataset = Dataset.from_pandas(data[data["split"] == "val"])

# Create a DatasetDict
dataset_dict = DatasetDict(
    {"train": train_dataset, "test": test_dataset, "validation": val_dataset}
)

# map for expected ids to labels
id2label = {0: "Not hateful", 1: "Hateful"}
label2id = {"Not hateful": 0, "Hateful": 1}

# load bert
model_name = "bert-base-cased"  #'google/bert_uncased_L-4_H-128_A-2'
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

if torch.cuda.is_available():
    model = model.to(device)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_data = dataset_dict.map(tokenize_function, batched=True)

# Remove the text column because the model does not accept raw text as an input
tokenized_data = tokenized_data.remove_columns(["text", "__index_level_0__", "split"])

# Set the format of the dataset to return PyTorch tensors instead of lists
tokenized_data.set_format("torch")

# create a DataLoader for your training and test datasets so you can iterate over batches of data:
train_dataloader = DataLoader(tokenized_data["train"], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_data["test"], batch_size=8)

optimizer = AdamW(model.parameters(), lr=4e-5)

num_epochs = 2
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

# training the model
model.train()

# iterate over epochs
for epoch in range(num_epochs):
    # iterate over batches in training set
    for batch in train_dataloader:
        batch.pop("index")  # remove 'index' from batch
        batch = {k: v.to(device) for k, v in batch.items()}
        # **kwargs is a common idiom to allow an arbitrary number of arguments to functions
        # The **kwargs will give you all keyword arguments as a dictionary
        # https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters
        #

        outputs = model(**batch)
        # Note that Transformers models all have a default task-relevant loss function,
        # so you don’t need to specify one unless you want to

        # get the loss form the outputs
        # in this example, the outputs are instances of subclasses of ModelOutput
        # https://huggingface.co/transformers/v4.3.0/main_classes/output.html
        # Those are data structures containing all the information returned by
        # the model, but that can also be used as tuples or dictionaries.

        # the outputs object has a loss and logits attribute
        # You can access each attribute as you would usually do,
        # and if that attribute has not been returned by the model, you will get None.
        # for instance, outputs.loss is the loss computed by the model

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        lr_scheduler.step()

        optimizer.zero_grad()

        progress_bar.update(1)

# evaluate model
# define the metric you want to use to evaluate your model
metric = evaluate.load("accuracy")
progress_bar = tqdm(range(len(eval_dataloader)))

predictions_bin_dict = {}
predictions_prob_dict = {}

# put the model in eval mode
model.eval()
# iterate over batches of evaluation dataset
for batch in eval_dataloader:
    indices = batch.pop("index")  # remove 'index' from batch and store it
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        # pass the batches to the model and get the outputs
        outputs = model(**batch)

    # get the logits from the outputs
    logits = outputs.logits

    # use softmax to get the predicted probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # add the probabilities of the positive class to the dictionary
    for i, probability in zip(indices, probabilities[:, 0]):
        predictions_prob_dict[
            i.item()
        ] = probability.item()  # assuming that the positive class is the second one

    # use argmax to get the predicted class
    predictions = torch.argmax(logits, dim=-1)

    # add the predictions to the dictionary
    for i, prediction in zip(indices, predictions):
        predictions_bin_dict[i.item()] = prediction.item()

    # metric.add_batch() adds a batch of predictions and references
    # Metric.add_batch() by passing it your model predictions, and the references
    # the model predictions should be evaluated against
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)
# calculate a metric by  calling metric.compute()
metric.compute()

# save the predictions to a JSON file
with open("predictions_bin_bert.json", "w") as f:
    json.dump(predictions_bin_dict, f)

with open("predictions_prob_bert.json", "w") as f:
    json.dump(predictions_prob_dict, f)
