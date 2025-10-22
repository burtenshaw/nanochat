import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


model_id = "/fsx/benjamin_burtenshaw/transformers/nanochat-d32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id


print("=" * 80)
print("TEST 1: Simplified NanoChat template")
print("=" * 80)
conversation = [
    {"role": "user", "content": "What is the capital of France?"},
]


inputs = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

print(f"Formatted prompt: {tokenizer.decode(inputs['input_ids'][0])}")
print(f"Input IDs: {inputs['input_ids'][0].tolist()}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]
print(f"\nGenerated: {tokenizer.decode(generated_tokens)}")
print("=" * 80)


# -----------------------------------------------------------------------------
# Minimal GRPO-style fine-tuning


raw_ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split="table_gpt_no_think")
train_val = raw_ds.train_test_split(test_size=0.1, seed=42)
train_ds = train_val["train"]
eval_ds = train_val["test"]


max_train_steps = 50
prompt_batch_size = 1
num_generations = 4
max_new_tokens = 128
temperature = 1.0
top_k = 50
learning_rate = 5e-6
weight_decay = 0.0
epsilon = 0.2
gradient_accumulation_steps = 1
warmup_ratio = 0.1
logging_frequency = 5


def reward_dots(texts):
    return [text.count(".") for text in texts]


def per_token_log_probs(logits, labels):
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def prepare_prompt(example):
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        add_generation_prompt=True,
        truncation=True,
        max_length=2048,
        padding=False,
        return_dict=True,
        return_tensors="pt",
    )
    return formatted["input_ids"], formatted["attention_mask"]


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_update_steps = max_train_steps // gradient_accumulation_steps
warmup_steps = max(1, int(total_update_steps * warmup_ratio))
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)


if device.type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    autocast_ctx = nullcontext()


def evaluate_sample():
    model.eval()
    example = eval_ds[0]
    prompt_ids, prompt_mask = prepare_prompt(example)
    with torch.no_grad():
        sequences = model.generate(
            input_ids=prompt_ids.to(device),
            attention_mask=prompt_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    model.train()
    completion = sequences[0, prompt_ids.shape[1] :]
    print("Sample eval completion:", tokenizer.decode(completion, skip_special_tokens=True))


model.train()
evaluate_sample()


train_index = 0
global_step = 0
running_reward = 0.0

for step in range(1, max_train_steps + 1):
    example = train_ds[train_index % len(train_ds)]
    train_index += 1

    prompt_ids, prompt_mask = prepare_prompt(example)
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    prompt_length = prompt_ids.shape[1]

    prompt_repeat = prompt_ids.repeat(num_generations, 1)
    mask_repeat = prompt_mask.repeat(num_generations, 1)

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_repeat,
            attention_mask=mask_repeat,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
        )
    model.train()

    sequences = generated
    attention_mask = (sequences != tokenizer.pad_token_id).long()
    completion_mask = attention_mask.clone()
    completion_mask[:, :prompt_length] = 0

    completion_tokens = sequences[:, prompt_length:]
    completion_texts = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
    rewards = torch.tensor(reward_dots(completion_texts), dtype=torch.float32, device=device)
    running_reward += rewards.mean().item()

    rewards_view = rewards.view(prompt_batch_size, num_generations)
    mean_rewards = rewards_view.mean(dim=1, keepdim=True)
    std_rewards = rewards_view.std(dim=1, keepdim=True)
    std_rewards = torch.where(std_rewards > 0, std_rewards, torch.ones_like(std_rewards))
    advantages = ((rewards_view - mean_rewards) / std_rewards).view(-1)

    labels = sequences[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = tokenizer.pad_token_id

    with torch.no_grad():
        with (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()):
            old_outputs = model(
                input_ids=sequences,
                attention_mask=attention_mask,
                use_cache=False,
            )
        old_log_probs = per_token_log_probs(old_outputs.logits[:, :-1], labels)

    valid_mask = (completion_mask[:, 1:] == 1) & (labels != tokenizer.pad_token_id)

    optimizer.zero_grad(set_to_none=True)
    with (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()):
        outputs = model(
            input_ids=sequences,
            attention_mask=attention_mask,
            use_cache=False,
        )
        log_probs = per_token_log_probs(outputs.logits[:, :-1], labels)

    ratio = (log_probs - old_log_probs).exp()
    ratio = torch.where(valid_mask, ratio, torch.ones_like(ratio))
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)

    adv = advantages.unsqueeze(1)
    loss_unclipped = ratio * adv
    loss_clipped = clipped_ratio * adv
    per_token_loss = -torch.min(loss_unclipped, loss_clipped)
    per_token_loss = torch.where(valid_mask, per_token_loss, torch.zeros_like(per_token_loss))

    denom = valid_mask.sum().clamp(min=1)
    loss = per_token_loss.sum() / denom

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    global_step += 1

    if step % logging_frequency == 0:
        avg_reward = running_reward / logging_frequency
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"step={step:04d} | loss={loss.item():.4f} | avg_reward={avg_reward:.4f} | lr={current_lr:.2e}"
        )
        running_reward = 0.0

evaluate_sample()
print("Training complete.")




