
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, DataCollatorForSeq2Seq, PreTrainedTokenizer, TrainingArguments, pipeline
from datasets import load_dataset, Dataset
import evaluate
from dotenv import load_dotenv

import os

load_dotenv()

os.environ["HF_HOME"] = os.getenv("HF_HOME")
token = os.getenv("HF_TOKEN")
save_dir = os.getenv("SAVE_MODEL_DIR")


def train_model(checkpoint):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ds = load_dataset('atenglens/taiwanese_english_translation')

    ds = ds['train'].train_test_split(test_size=0.2)
    print(ds['train'][0])

    source = 'en'
    target = 'tw'
    prefix = "translate from english to taiwanese: "

    def preprocess_func(sentence):
        inputs = [prefix + sentence[source] for sentence in sentence['translation']]
        targets = [sentence[target] for sentence in sentence['translation']]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    tokenized_text = ds.map(preprocess_func, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    evaluate.load("sacrebleu")

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=0.01,
        fp16=True,
        fp16_full_eval=True,
        logging_steps=1,
        output_dir='outputs',
        num_train_epochs=5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_text['train'],
    )

    # Train the model
    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


def translate(model, tokenizer, prompt: str) -> None:
    input = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**input)
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated



if __name__ == "__main__":
    train_model("google-t5/t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    prompt = "How are you?"
    print(translate(model, tokenizer, prompt))
