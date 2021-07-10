# summrization
This is some summary code and model


# dataset download
链接：https://pan.baidu.com/s/1DyK59idec4VJCaf3J5ssog 
提取码：1234


# Bart model download
链接：https://pan.baidu.com/s/17gmfchFWl7NMmKfnfjA7gA 
提取码：1234

# T5 model
代码框架基于huggingface中的transformer包

设定checkpoint,可从huggingface个人仓库中读取模型

```python 
model_checkpoint = 'sunhao666/chi-sum2'
```
读取分词器
```python
from transformers import AutoTokenizer
# 分词器采样的是T5的分词器
tokenizer = AutoTokenizer.from_pretrained('uer/t5-base-chinese-cluecorpussmall')
```
读取模型
```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

进行文本摘要
```python
sentence = 'you input sentence for summarization'

input = tokenizer(sentence, max_length=max_target_length, truncation=True, return_tensors='pt')  # 对句子进行编码
del input['token_type_ids']
output = model.generate(
    **input,
    do_sample=True,  # 是否抽样
    num_beams=3,   # beam search
    bad_words_ids=[[101], [100]],  # bad_word_ids表示这些词不会出现在结果中
    # length_penalty=100,   # 长度的惩罚项
    max_length=100,     # 输出的最大长度
    repetition_penalty=5.0   # 重复的惩罚
)
summary = tokenizer.decode(output[0])
```
最终得到结果summary
