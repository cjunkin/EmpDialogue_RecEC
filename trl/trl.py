# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""
from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl.trl import ModelConfig, SFTTrainer, get_kbit_device_map, get_peft_config, get_quantization_config

import data_utils
from main import ModelWrapper
import metric_utils
from misc_utils import init_logger, logger


tqdm.pandas()


class TRLModelWrapper(PreTrainedModel):
    def __init__(self, tokenizer, generate_net: Transformer, emotion_net: EmotionNet, beam_width: int):
        """
        Initialize the wrapper with our model and tokenizer.
        :param model: The custom model instance.
        :param tokenizer: An instance of PreTrainedTokenizer or similar, compatible with model.
        """
        super.__init__(generate_net, emotion_net, beam_width)
        self.tokenizer = tokenizer

    def forward(self, input):
        return 
    
    def predict(self, input):
        return


    def generate(self, input, **kwargs):
        """
        Generate responses for the given input texts using the model.
        :param input_texts: A list of input texts to generate responses for.
        :param kwargs: Additional keyword arguments to pass to the model's generate method.
        :return: A list of generated responses.
        """
        # Tokenize the input texts
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Move inputs to the same device as the model
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        # Generate responses. Adapt this part based on your model's method for generating text.
        # This example assumes your model has a `predict` method that returns indices of generated tokens.
        with torch.no_grad():
            output_indices = self.model.predict(inputs)

        # Decode the generated indices to text
        responses = [self.tokenizer.decode(generated, skip_special_tokens=True) for generated in output_indices]

        return responses


@dataclass
class ScriptArguments:
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_text_field: str = field(default="text", metadata={"help": "the text field of the dataset"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)



from trl.trl import PPOConfig, PPOTrainer
from transformers import AutoTokenizer

################
# Model Loading
################

# Assuming your model is compatible with the Hugging Face interface
model_name_or_path = "path/to/your/custom/model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# PPO Configuration
ppo_config = PPOConfig(
    model_name=model_name_or_path,
    batch_size=64,  # Adapt based on your GPU capacity
    forward_batch_size=16,  # Forward pass batch size
    learning_rate=1.41e-5,
    num_train_epochs=3,
    max_grad_norm=1.0,
    # Add other PPOConfig parameters as needed
)

# Ensure your model has a compatible interface
model = ... # Your model loading mechanism here

################
# Dataset
################

from datasets import load_dataset

dataset_name = "your_dataset_name_here"
raw_datasets = load_dataset(dataset_name)

# Preprocess and prepare your dataset here

################
# PPO Training Loop
################
# Initialize the PPO Trainer
trainer = PPOTrainer(
    model=model,
    args=ppo_config,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["validation"],
    tokenizer=tokenizer,
    # Specify additional arguments as necessary
)

# Start training
trainer.train()
