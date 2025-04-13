import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, column_name, max_length=512):
        self.data = data[column_name].dropna().reset_index(drop=True)  # 只处理指定列，并移除缺失值
        self.tokenizer = tokenizer
        self.max_length = max_length  # 最大文本长度

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定列的文本
        text = self.data.iloc[idx]

        # 使用 BERT Tokenizer 处理文本
        inputs = self.tokenizer(text,
                                padding='max_length',  # 填充到最大长度
                                truncation=True,  # 截断超过最大长度的文本
                                max_length=self.max_length,
                                return_tensors='pt')  # 返回 PyTorch 张量

        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # (1, seq_len) -> (seq_len)
            'attention_mask': inputs['attention_mask'].squeeze(0)  # (1, seq_len) -> (seq_len)
        }


# 加载 CSV 数据
def load_data(csv_file, column_name):
    # 读取 CSV 文件，并检查指定列是否存在
    data = pd.read_csv(csv_file)
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")
    return data


# 加载 BERT 模型
def load_bert_model(model_path='./chinese-roberta-ext-large/'):
    model = BertModel.from_pretrained(model_path)
    model.eval()  # 切换到评估模式
    return model


# 生成 BERT 词向量

def get_bert_embeddings(dataloader, model, device='cuda'):
    embeddings = []

    model.to(device)

    for batch in tqdm(dataloader, desc="Processing batches"):
        torch.cuda.empty_cache()  # 清空缓存，释放内存

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 使用自动混合精度
        with torch.no_grad(), autocast():
            outputs = model(input_ids, attention_mask=attention_mask)

        # 提取 [CLS] token 的词向量，作为句子的表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        embeddings.append(cls_embedding)

        # 清理中间变量，防止内存泄漏
        del outputs, cls_embedding
        torch.cuda.empty_cache()  # 清空缓存

        # 定期拼接并清空中间缓存
        if len(embeddings) >= 100:  # 每处理100个批次就拼接一次
            embeddings_batch = torch.cat(embeddings, dim=0)
            embeddings = [embeddings_batch]  # 重置embeddings，只保留当前拼接的数据
            torch.cuda.empty_cache()  # 清理缓存，减少内存占用

    # 最终拼接所有批次的词向量
    embeddings = torch.cat(embeddings, dim=0)  # (num_samples, hidden_size)
    return embeddings
