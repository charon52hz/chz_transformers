
import torch
from transformers import (
    AutoTokenizer, AutoModel,
)

# 加载数据集
from transformers.models.bart.modeling_bart_seqLoss import BartForConditionalGeneration

model_name = "IDEA-CCNL/Randeng-BART-139M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

label_means_5000 = torch.load(r"E:\chz1\pythonRep\transformers\examples\pytorch\summarization\labels_means5000-1.pt")

for i in range(label_means_5000.size(0)):
    token = f"token_{i}"
    tokenizer.add_tokens(token, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        model.embeddings.word_embeddings.weight[-1:, :] = label_means_5000[i]
print(model.embeddings.word_embeddings.weight.size())

print(model.embeddings.word_embeddings.weight[-2:, :])