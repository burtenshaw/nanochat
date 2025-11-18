import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


model_id = "/fsx/benjamin_burtenshaw/nanochat_checkpoints/mid_d20-converted"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device)
tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id


print("=" * 80)
print("TEST 1: Plain Autoregressive Prompt")
print("=" * 80)
prompt = "The Eiffel Tower stands in Paris and"
test_inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"Prompt tokens: {tokenizer.convert_ids_to_tokens(test_inputs['input_ids'][0])}")

with torch.no_grad():
    test_outputs = model.generate(
        **test_inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_tokens = test_outputs[0, test_inputs["input_ids"].shape[1] :]
print(f"\nGenerated: {tokenizer.decode(generated_tokens, skip_special_tokens=True)}")
print("=" * 80)


# -----------------------------------------------------------------------------
# Minimal mid-training loop (single GPU)


raw_dataset = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split="table_gpt_no_think")
splits = raw_dataset.train_test_split(test_size=0.05, seed=13)
train_dataset = splits["train"]
eval_dataset = splits["test"]


max_length = 2048
train_batch_size = 1
eval_batch_size = 1
num_epochs = 1
gradient_accumulation_steps = 16
learning_rate = 6e-5
weight_decay = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.95
warmup_ratio = 0.02
max_train_examples = None
max_eval_examples = 128
logging_frequency = 20


def flatten_messages(messages):
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role:
            lines.append(f"{role.upper()}: {content}")
        else:
            lines.append(content)
    return "\n".join(lines)


def tokenize_example(example):
    text = flatten_messages(example["messages"])
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


train_dataset = train_dataset.map(tokenize_example, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_example, remove_columns=eval_dataset.column_names)


if max_train_examples is not None:
    train_dataset = train_dataset.select(range(min(len(train_dataset), max_train_examples)))
if max_eval_examples is not None:
    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), max_eval_examples)))


def collate_fn(batch):
    batch_dict = {
        "input_ids": [record["input_ids"] for record in batch],
        "attention_mask": [record["attention_mask"] for record in batch],
    }
    padded = tokenizer.pad(batch_dict, padding=True, return_tensors="pt")
    labels = padded["input_ids"].clone()
    labels[padded["attention_mask"] == 0] = -100
    padded["labels"] = labels
    return padded


TrainLoader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
EvalLoader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(adam_beta1, adam_beta2),
    weight_decay=weight_decay,
)

num_update_steps_per_epoch = max(len(TrainLoader) // gradient_accumulation_steps, 1)
max_train_steps = num_epochs * num_update_steps_per_epoch
warmup_steps = max(1, int(max_train_steps * warmup_ratio))
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_train_steps)


def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in EvalLoader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            losses.append(loss.float().item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


model.train()
global_step = 0
running_loss = 0.0
running_steps = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(TrainLoader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        running_loss += outputs.loss.float().item()
        running_steps += 1

        if step % gradient_accumulation_steps == 0 or step == len(TrainLoader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % logging_frequency == 0:
                current_lr = scheduler.get_last_lr()[0]
                mean_loss = running_loss / running_steps
                print(f"step={global_step:05d} | loss={mean_loss:.4f} | lr={current_lr:.2e}")
                running_loss = 0.0
                running_steps = 0

    val_loss = evaluate()
    print(f"Validation loss after epoch {epoch + 1}: {val_loss:.4f}")

print("Training complete.")

