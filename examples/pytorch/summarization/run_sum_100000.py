import copy
import lawrouge
import numpy as np
import torch
import random
from datasets import load_dataset, concatenate_datasets

from transformers import (
    BartConfig,
    # BartForConditionalGeneration,
    AutoTokenizer,
    set_seed,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, AutoModelForSeq2SeqLM,
)

# # 设置随机数种子
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# 加载数据集
from transformers.models.bart.modeling_bart_seqLoss import BartForConditionalGeneration

train_dataset = load_dataset('json', data_files='./data/train_subsets.json')
test_dataset = load_dataset('json', data_files='./data/test_public.json')
valid_dataset = load_dataset('json', data_files='./data/valid.json')

model_name = "IDEA-CCNL/Randeng-BART-139M"
# model_name = "IDEA-CCNL/Randeng-BART-139M-SUMMARY"
# model_name = "beyond/genius-base-chinese"
# model_name = "./output/original_results/checkpoint-15500"
# model_name = "./output/lcstsm/version3/sentenceLoss/checkpoint-15500"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

max_input_length = 1024
max_target_length = 1024

batch_size = 128
args = Seq2SeqTrainingArguments(
    output_dir="output/lcstsm/version3/sentenceLoss",
    num_train_epochs=20,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=500,
    evaluation_strategy="epoch",  ## epoch
    save_total_limit=3,
    # generation_max_length最大生成长度，系统默认20 generation_num_beams=1表示贪心解码，大于1为树搜索
    generation_max_length=1024,
    generation_num_beams=1,
)

def preprocess_function001(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function002(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["n-summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function1(examples):
    inputs1 = [doc for doc in examples["train"]["text"]]
    inputs2 = [doc for doc in examples["train"]["text"]]

    model_inputs1 = tokenizer(inputs1, max_length=max_input_length, truncation=True)
    model_inputs2 = tokenizer(inputs2, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels1 = tokenizer(examples["train"]["summary"], max_length=max_target_length, truncation=True)
        labels2 = tokenizer(examples["train"]["n-summary"], max_length=max_target_length, truncation=True)

    model_inputs1["labels"] = labels1["input_ids"]
    model_inputs2["labels"] = labels2["input_ids"]

    model_inputs = copy.deepcopy(model_inputs1)
    model_inputs['input_ids'].extend(model_inputs2['input_ids'])
    model_inputs["labels"].extend(model_inputs2["labels"])
    model_inputs['attention_mask'].extend(model_inputs2['attention_mask'])

    return model_inputs

def main():

    train_data_txt, validation_data_txt, test_data_txt = train_dataset, valid_dataset, test_dataset

    tokenized_train_datasets = train_dataset.map(preprocess_function001, batched=True)
    tokenized_eval_datasets = validation_data_txt.map(preprocess_function001, batched=True)

    # trainer = SFTTrainer(
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_eval_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # neftune_noise_alpha=5  ############SFTTrainer
    )

    # 评估时：
    # eval_result = trainer.evaluate()
    # print(eval_result)

    # 训练时：
    # train_result = trainer.train(resume_from_checkpoint=True)
    train_result = trainer.train()
    print(train_result)

    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# 这里用的是中文lawrouge 至于字符级还是词级计算看自己调整 这里是字符级
def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 保存修改后的数据回JSON文件
    # import json
    # with open('./data/segment_model_output2.json', 'w', encoding='utf-8') as file:
    #     for i, entry in enumerate(decoded_preds):
    #         # 将JSON对象转化为字符串，并逐行写入文件
    #         result = {
    #             "summary": decoded_labels[i],
    #             "summary2": entry
    #         }
    #         json_str = json.dumps(result, ensure_ascii=False)
    #         file.write(json_str + '\n')

    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    # reserve_decoded_preds = [pred[::-1] for pred in decoded_preds]
    decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]

    rouge = lawrouge.Rouge()

    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    print(result)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}

    result = {key: value * 100 for key, value in result.items()}
    return result

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)     # 开启异常检测
    main()
