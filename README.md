# summrization

这是一个中文自动摘要代码仓库，我们使用了基于IT-IDF的传统方法进行图摘要抽取；基于Transformer的两个模型分别是：基于T5模型的自动摘要；基于bart模型的自动摘要。生成式摘要的一大问题是找不到合适的停止位置，这也是后两个模型面对的问题。

如果想要了解BART模型，可以参考我的[CSDN博客](https://blog.csdn.net/weixin_43718786/article/details/119741580?spm=1001.2014.3001.5501)

This is a Chinese automatic summarization code repository. We use the traditional method based on IT-IDF and graph to extract to do summarization. Two models based on Transformer are also used. Automatic summary based on T5 model; Automatic summarization based on BART(not BERT) model. One problem with generative summaries, as with the latter two models, is that it is difficult to find a suitable place to stop.

If you want to know BART model, you can refer to my [CSDN blog] (https://blog.csdn.net/weixin_43718786/article/details/119741580?spm=1001.2014.3001.5501)

# 资源下载 Download
## training dataset download
链接：https://pan.baidu.com/s/1DyK59idec4VJCaf3J5ssog 
提取码 keyword：1234


## Bart model download
链接：https://pan.baidu.com/s/17gmfchFWl7NMmKfnfjA7gA 
提取码 keyword：1234

## T5 model
代码框架基于huggingface中的transformer包 The code framework is based on the Transformer package in HuggingFace

The code framework is based on the Transformer package in HuggingFace
 
设定checkpoint,可从huggingface个人仓库中读取模型

Set checkpoint to read models from HuggingFace's repository

```python 
model_checkpoint = 'sunhao666/chi-sum2'
```
读取分词器 word segment
```python
from transformers import AutoTokenizer
# 分词器采样的是T5的分词器
tokenizer = AutoTokenizer.from_pretrained('uer/t5-base-chinese-cluecorpussmall')
```
读取模型 load the model
```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

进行文本摘要 summarization
```python
sentence = 'you input sentence for summarization'

input = tokenizer(sentence, max_length=128, truncation=True, return_tensors='pt')  # 对句子进行编码
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
summary = tokenizer.decode(output[0]).split('[SEP]')[0].replace('[CLS]', '').replace(' ', '')
```
最终得到结果summary


# 集成系统 How to use
运行runningExample.py文件

run runningExample.py
```
python runningExample.py
```
按提示进行输入

Enter as prompted
