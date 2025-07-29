from datasets import load_dataset

dataset = load_dataset("json",data_files="economics.jsonl" , split="train")

def format_for_t5(example):
  instruction = example["instruction"]
  input_text = example.get("input","")
  return {
      "prompt": f"{instruction}\n{input_text}".strip(),
      "labels" : example["output"]
  }

formatted = dataset.map(format_for_t5)

model_name = "HuggingFaceTB/SmolLM2-360M"

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    inputs = tokenizer(
        example["prompt"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    outputs = tokenizer(
        example["labels"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"]
    }

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

tokenized_dataset = formatted.map(tokenize)

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

train = tokenized_dataset["train"]
test = tokenized_dataset["test"]

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./checkpoints-smollm",
    num_train_epochs = 5,
    per_device_train_batch_size=2,
    learning_rate=5e-4,
    weight_decay=0.01,
    save_steps=10,
    logging_dir=".logs",
    logging_steps=1,
    report_to='none'
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=eval,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("my-smollm")
tokenizer.save_pretrained("my-smollm")

