import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# loading dataset
tweets = pd.read_csv("data/MMHS150K/MMHS150K_with_img_text.csv")
data = tweets[["tweet_text_clean", "binary_hate", "split"]]
data = data.rename(
    columns={"tweet_text_clean": "text", "binary_hate": "labels", "split": "split"}
)

# Convert the pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(data[data["split"] == "train"])
test_dataset = Dataset.from_pandas(data[data["split"] == "test"])
val_dataset = Dataset.from_pandas(data[data["split"] == "val"])

# Create a DatasetDict
dataset_dict = DatasetDict(
    {"train": train_dataset, "test": test_dataset, "validation": val_dataset}
)

# load model and tokenizer
def load_pretrained(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to(device)
    return tokenizer, model


# load bert
tokenizer, model = load_pretrained("distilbert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_data = dataset_dict.map(tokenize_function, batched=True)

# Set the format of the dataset to return PyTorch tensors instead of lists
tokenized_data.set_format("torch")

train_dataloader = DataLoader(tokenized_data["train"], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_data["test"], batch_size=8)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 2
num_training_steps = num_epochs * len(train_dataloader)
# feel free to experiment with different num_warmup_steps
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

# put the model in train mode
model.train()

# iterate over epochs
for epoch in range(num_epochs):
    # iterate over batches in training set
    for batch in train_dataloader:
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

# define the metric you want to use to evaluate your model
metric = evaluate.load("accuracy")
progress_bar = tqdm(range(len(eval_dataloader)))

# put the model in eval mode
model.eval()
# iterate over batches of evaluation dataset
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        # pass the batches to the model and get the outputs
        outputs = model(**batch)

    # get the logits from the outputs
    logits = outputs.logits

    # use argmax to get the predicted class
    predictions = torch.argmax(logits, dim=-1)

    # metric.add_batch() adds a batch of predictions and references
    # Metric.add_batch() by passing it your model predictions, and the references
    # the model predictions should be evaluated against
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)
# calculate a metric by  calling metric.compute()
metric.compute()
