# coding=utf-8
import jieba
from pkuseg import pkuseg
from transformers import BertConfig, BertForMaskedLM, DataCollatorForWholeWordMask, \
    BertTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BartForCausalLM, \
    LEDForConditionalGeneration, LineByLineTextDataset
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class pretrain_dataset(Dataset):

    def __init__(self, path, tokenizer, dup_factor=5, max_length=512):  # dup_factor : dynamic mask for 5 times
        self.examples = []
        with open(path, 'r', encoding='utf-8') as f:
            total_data = f.readlines()
            with tqdm(total_data * dup_factor) as loader:
                for data in loader:
                    # clean data
                    data = data.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '').replace('　', '')
                    # chinese_ref = self.get_new_segment(data)
                    input_ids = tokenizer.encode_plus(data, truncation=True, max_length=max_length).input_ids
                    dict_data = {'input_ids': input_ids}  # , 'chinese_ref' : chinese_ref
                    self.examples.append(dict_data)
                    loader.set_description(f'loading data')

    def get_new_segment(self, segment):
        """
            使用分词工具获取 whole word mask
            用于wwm预训练
            e.g [喜,欢]-> [喜，##欢]
        """
        seq_cws = jieba.cut("".join(segment))  # 利用jieba分词
        # seq_cws = segment
        chinese_ref = []
        index = 1
        for seq in seq_cws:
            for i, word in enumerate(seq):
                if i > 0:
                    chinese_ref.append(index)
                index += 1
        return chinese_ref

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    # configuration
    epoch = 100
    batch_size = 4
    pretrian_model = r'...'
    train_file = r'train_nlpcc.txt'
    test_file = r'test_nlpcc.txt'
    save_epoch = 1  # every 10 epoch save checkpoint

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = BertConfig.from_pretrained(pretrian_model)
    tokenizer = BertTokenizer.from_pretrained(pretrian_model)

    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_file,  # mention train text file here
        block_size=512)

    # train_dataset = pretrain_dataset(train_file,tokenizer)

    test_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=test_file,  # mention train text file here
        block_size=512,
    )
    # test_dataset = pretrain_dataset(test_file, tokenizer)
    # model = BertForMaskedLM(config)
    model = LEDForConditionalGeneration.from_pretrained(pretrian_model).to(device)  # BartForCausalLM
    print('No of parameters: ', model.num_parameters())

    data_collator = DataCollatorForLanguageModeling(  # DataCollatorForWholeWordMask DataCollatorForLanguageModeling
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    print('No. of lines: ', len(train_dataset))
    save_step = len(train_dataset) * save_epoch
    tot_step = int(len(train_dataset) / batch_size * epoch)
    print(f'\n\t***** Running training *****\n'
          f'\tNum examples = {len(train_dataset)}\n'
          f'\tNum Epochs = {epoch}\n'
          f'\tBatch size = {batch_size}\n'
          f'\tTotal optimization steps = {tot_step}\n')

    # official training
    training_args = TrainingArguments(
        output_dir=r'nlpcc',
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=16,
        learning_rate=2e-05,
        warmup_steps=100,
        weight_decay=0,
        # save_steps=save_step,
        logging_dir="../logs",
        logging_strategy="steps",
        logging_steps=1,
        save_total_limit=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    trainer.save_model(r'/mlm_ouputs/led/nlpcc')

