# coding=utf-8
import logging
import datasets
import numpy as np
import lawrouge
import rouge
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
from transformers import LEDForConditionalGeneration, BertTokenizer

from datasets import load_dataset

logger = logging.getLogger("longformer-chinese")
logging.basicConfig(level=logging.INFO)

dataset = load_dataset('json', data_files=r'E:\chz1\pythonRep\transformers\src\transformers\models\led\data\nlpcc_data.json',
                       field='data')  # 加载自己的长文本摘要数据集
dataset = dataset.shuffle(seeds=42)  # shuffle

model_name = r'E:\chz1\pythonRep\transformers\src\transformers\models\led\model\led_init'
tokenizer = BertTokenizer.from_pretrained(model_name)
# tokenizer = BertTokenizer.from_pretrained(r'D:\nlp\project\long-document\bert-base-chinese')  # 加载bert tokenizer
model = LEDForConditionalGeneration.from_pretrained(model_name)  # 加载Longformer


# model.resize_token_embeddings(tokenizer.vocab_size) # 补充词表 21128--->50000

def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
    }


dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])  # , remove_columns=["title", "content"]

max_input_length = 8192  # 4096 or others ，不能超过我们转换的最大长度8192
max_target_length = 1024  # summary, target text


def preprocess_function(examples):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dataset = dataset.shuffle()

train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).values()
tokenized_datasets = datasets.DatasetDict({"train": train_data_txt, "validation": validation_data_txt}).map(
    preprocess_function, batched=True)

batch_size = 4  # ==>穷人
args = Seq2SeqTrainingArguments(
    fp16=True,
    output_dir="results_log",
    num_train_epochs=50,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,  # demo
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-04,
    warmup_steps=1000,
    weight_decay=0.0001,
    label_smoothing_factor=0.15,
    predict_with_generate=True,
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=1,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    gradient_accumulation_steps=1,
    generation_max_length=64,
    generation_num_beams=1,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]
    # Rouge with jieba cut
    # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
    # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]

    labels_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in labels]

    for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        if pred == "":
            decoded_preds[i] = "decoding error,skipping..."

    # print(decoded_preds)
    # print(decoded_labels)
    # rouge = lawrouge.Rouge()
    rouge = rouge.Rouge()
    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    # print(result)
    print(result)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}

    result = {key: value * 100 for key, value in result.items()}
    result["gen_len"] = np.mean(labels_lens)
    return result


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 保存模型即训练数据
train_result = trainer.train()
print(train_result)
trainer.save_model()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


