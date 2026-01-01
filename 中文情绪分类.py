import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import ast

# data = {
#     'text': [
#         '这个电影太好看了，强烈推荐！',
#         '服务态度很差，不会再来了。',
#         '产品质量不错，价格也合理。',
#         '物流太慢了，等了一个星期。',
#         '客服很耐心，问题解决了。',
#         '包装破损，商品有瑕疵。',
#         '性价比很高，物超所值。',
#         '广告和实际不符，失望。',
#         '操作简单，容易上手。',
#         '说明书不清楚，很难用。'
#     ],
#     'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1:正面, 0:负面
# }

 # 1. 读取txt文件内容
with open("comments.txt", 'r', encoding='utf-8') as f:
    content = f.read()
    
    # 2. 清理干扰内容（如注释、多余换行，可选）
    # 移除单行注释（#开头的行）
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        # 去掉行内的注释（如 # 正样本（1-100））
        line = line.split('#')[0].strip()
        if line:  # 保留非空行
            cleaned_lines.append(line)
    cleaned_content = '\n'.join(cleaned_lines)
    
    # 3. 解析为Python字典（核心：ast.literal_eval安全解析字符串为字典）
    try:
        data = ast.literal_eval(cleaned_content)
    except SyntaxError as e:
        raise ValueError(f"txt文件内容语法错误：{e}")
    
    # 4. 验证格式（确保text和label是列表，且长度匹配）
    if not isinstance(data, dict):
        raise TypeError("解析结果不是字典")
    if 'text' not in data or 'label' not in data:
        raise KeyError("字典缺少text或label键")
    if not isinstance(data['text'], list) or not isinstance(data['label'], list):
        raise TypeError("text或label不是列表")
    if len(data['text']) != len(data['label']):
        raise ValueError(f"text长度({len(data['text'])})和label长度({len(data['label'])})不匹配")


df = pd.DataFrame(data)
print(df,type(df))

# 数据集处理类
class SentimentDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }   #返回输入词向量，注意力掩码和标签
    
# 初始化模型
MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42 #随机种子，确保可重复性
)

print(len(train_texts),len(val_texts))

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

BATCH_SIZE = 2
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

EPOCHS = 30
LEARNING_RATE = 2e-5

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, # 预热步数 
    num_training_steps=total_steps
)

def training(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for batch in data_loader:
        # 清空梯度
        optimizer.zero_grad()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)
    
def eval_model(model, data_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
    
    accuracy = correct_predictions.double() / total_predictions
    return accuracy.item()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)
    train_loss = training(model, train_loader, optimizer, scheduler)
    val_accuracy = eval_model(model, val_loader)
    print(f"Train loss: {train_loss:.4f}")
    print(f"Val accuracy: {val_accuracy:.4f}")

torch.save(model.state_dict(), 'sentiment_model.pt')

def predict(text, model, tokenizer):
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # logits = outputs.logits
        # _, preds = torch.max(logits, dim=1)
        # return preds.item()
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label].item()
    
    sentiment = "正面" if pred_label == 1 else "负面"
    
    return sentiment, confidence, probs.tolist()[0]

predict_text = ""

print("输入文字序列：")
while True:
    sentiment, confidence, probs = predict(predict_text, model, tokenizer)
    print(f"情感分类结果：{sentiment}，置信度：{confidence:.4f}")
    print(f"正面情感概率：{probs[1]:.4f}，负面情感概率：{probs[0]:.4f}")
    print("\n请输入下一个文字序列（输入空行结束）：")
    predict_text = input()
    if predict_text == "exit":
        break