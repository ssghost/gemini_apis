from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import FastLanguageModel, is_bfloat16_supported  
from datasets import load_dataset
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

max_seq_length = 2048  
load_in_4bit = True
def main() -> None:
  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-4",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit
  )
  
  dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
  
  model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None
  )
  
  tokenizer = get_chat_template(
      tokenizer,
      chat_template = "phi-4",
  )
  
  dataset = standardize_sharegpt(dataset)
  dataset = dataset.map(
      formatting_prompts_func,
      batched=True,
  )
  
  trainer = SFTTrainer(
  model = model,
  tokenizer = tokenizer,
  train_dataset = dataset,
  dataset_text_field = "text",
  max_seq_length = max_seq_length,
  data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
  dataset_num_proc = 2,
  packing = False, 
  args = TrainingArguments(
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 4,
      warmup_steps = 5,
      max_steps = 30,
      learning_rate = 2e-4,
      fp16 = not is_bfloat16_supported(),
      bf16 = is_bfloat16_supported(),
      logging_steps = 1,
      optim = "adamw_8bit",
      weight_decay = 0.01,
      lr_scheduler_type = "linear",
      seed = 3407,
      output_dir = "outputs",
      report_to = "none"
      )
  )
  
  trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>"
  )

  space = tokenizer(" ", add_special_tokens = False).input_ids[0]
  tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
  trainer_stats = trainer.train()
  model.save_pretrained("lora_model")  
  tokenizer.save_pretrained("lora_model")


def formatting_prompts_func(examples):
  convos = examples["conversations"]
  texts = [
    tokenizer.apply_chat_template(
        convo, tokenize = False, add_generation_prompt = False
    )
    for convo in convos
  ]
  return { "text" : texts, } pass

if __name__ == '__main__':
  main()

