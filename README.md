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


```python
model_checkpoint = 'sunhao666/chi-sum2'
from transformers import AutoTokenizer


from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained(model_self)
