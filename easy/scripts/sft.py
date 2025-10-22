import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


model_id = "/fsx/benjamin_burtenshaw/transformers/nanochat-d32"
max_new_tokens = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=False, dtype=torch.bfloat16
).to(device)


print("=" * 80)
print("TEST 1: Simplified NanoChat template")
print("="*80)
conversation = [
    {"role": "user", "content": "What is the capital of France?"},
]

inputs = tokenizer.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(device)

print(f"Formatted prompt: {tokenizer.decode(inputs['input_ids'][0])}")
print(f"Input IDs: {inputs['input_ids'][0].tolist()}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]
print(f"\nGenerated: {tokenizer.decode(generated_tokens)}")
print("=" * 80)

# -----------------------------------------------------------------------------
# Supervised finetuning setup (single GPU, minimal example)

tokenizer.pad_token = tokenizer.eos_token

raw_ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split="table_gpt_no_think")
splits = raw_ds.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
val_ds = splits["test"]

max_train_samples = None
max_eval_samples = None
if max_train_samples is not None:
    train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
if max_eval_samples is not None:
    val_ds = val_ds.select(range(min(max_eval_samples, len(val_ds))))

max_length = 2048
train_batch_size = 2
eval_batch_size = 2
num_epochs = 1
gradient_accumulation_steps = 4
learning_rate = 1e-5
weight_decay = 0.0
warmup_ratio = 0.03
logging_frequency = 10


def format_example(example):
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        "input_ids": formatted["input_ids"][0].tolist(),
        "attention_mask": formatted["attention_mask"][0].tolist(),
    }


train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names)


def collate_fn(batch):
    batch_dict = {
        "input_ids": [example["input_ids"] for example in batch],
        "attention_mask": [example["attention_mask"] for example in batch],
    }
    padded = tokenizer.pad(batch_dict, padding=True, return_tensors="pt")
    labels = padded["input_ids"].clone()
    labels[padded["attention_mask"] == 0] = -100
    padded["labels"] = labels
    return padded


train_dataloader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
eval_dataloader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)

num_update_steps_per_epoch = max(len(train_dataloader) // gradient_accumulation_steps, 1)
total_update_steps = num_epochs * num_update_steps_per_epoch
warmup_steps = int(total_update_steps * warmup_ratio)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)


def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            losses.append(loss.float().item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


model.train()
global_step = 0
running_loss = 0.0
running_steps = 0
block_loss = 0.0
block_steps = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for step, batch in enumerate(train_dataloader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss_value = outputs.loss.float().item()
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        block_loss += loss_value
        block_steps += 1

        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            averaged_loss = block_loss / block_steps
            running_loss += averaged_loss
            running_steps += 1

            if global_step % logging_frequency == 0:
                mean_recent_loss = running_loss / running_steps
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"step={global_step:05d} | loss={mean_recent_loss:.4f} | lr={current_lr:.2e}"
                )
                running_loss = 0.0
                running_steps = 0

            block_loss = 0.0
            block_steps = 0

    val_loss = evaluate()
    print(f"Validation loss after epoch {epoch + 1}: {val_loss:.4f}")

print("Training complete.")

