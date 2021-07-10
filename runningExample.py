from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import warnings
from pathlib import Path
from typing import List, Tuple, Union
import torch
import fire
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.utils import logging
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator

def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]) -> None:
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())


LAYERS_TO_COPY = {
    # maps  num layers in teacher -> num_layers in student -> which teacher layers to copy.
    # 12: bart, 16: pegasus, 6: marian/Helsinki-NLP
    12: {
        1: [0],  # This says that if the teacher has 12 layers and the student has 1, copy layer 0 of the teacher
        2: [0, 6],
        3: [0, 6, 11],      # the first, 7th and 12th decode layers
        4: [0, 4, 8, 11],
        6: [0, 2, 4, 7, 9, 11],
        9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
        12: list(range(12)),
    },
    16: {  # maps  num layers in student -> which teacher layers to copy
        1: [0],
        2: [0, 15],
        3: [0, 8, 15], 
        4: [0, 5, 10, 15],
        6: [0, 3, 6, 9, 12, 15],
        8: [0, 2, 4, 6, 8, 10, 12, 15],
        9: [0, 1, 3, 5, 7, 9, 11, 13, 15],
        12: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        16: list(range(16)),
    },
    6: {1: [0], 2: [0, 5], 3: [0, 2, 5], 4: [0, 1, 3, 5], 6: list(range(6))},
}
LAYERS_TO_SUPERVISE = {
    # maps  num layers in student -> which teacher layers to copy.
    6: {1: [5], 2: [3, 5], 3: [1, 4, 5], 4: [1, 2, 4, 5]},
    12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
    16: {1: [15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15]},
}

def create_student_by_copying_alternating_layers(
    teacher: Union[str, PreTrainedModel],
    save_path: Union[str, Path] = "student",
    e: Union[int, None] = None,
    d: Union[int, None] = None,
    copy_first_teacher_layers=False,
    e_layers_to_copy=None,
    d_layers_to_copy=None,
    **extra_config_kwargs
) -> Tuple[PreTrainedModel, List[int], List[int]]:
    
    _msg = "encoder_layers and decoder_layers cannot be both None-- you would just have an identical teacher."
    assert (e is not None) or (d is not None), _msg
    if isinstance(teacher, str):
        AutoTokenizer.from_pretrained(teacher).save_pretrained(save_path)  # purely for convenience
        teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher).eval()
    else:

        assert isinstance(teacher, PreTrainedModel), f"teacher must be a model or string got type {type(teacher)}"
    init_kwargs = teacher.config.to_diff_dict()

    try:
        teacher_e, teacher_d = teacher.config.encoder_layers, teacher.config.decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"encoder_layers": e, "decoder_layers": d})
    except AttributeError:  # T5
        teacher_e, teacher_d = teacher.config.num_layers, teacher.config.num_decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"num_layers": e, "num_decoder_layers": d})

    # Kwargs to instantiate student: teacher kwargs with updated layer numbers + **extra_config_kwargs
    init_kwargs.update(extra_config_kwargs)

    # Copy weights
    student_cfg = teacher.config_class(**init_kwargs)
    student = AutoModelForSeq2SeqLM.from_config(student_cfg)
    # Start by copying the full teacher state dict this will copy the first N teacher layers to the student.
    info = student.load_state_dict(teacher.state_dict(), strict=False)
    assert info.missing_keys == [], info.missing_keys  # every student key should have a teacher keys.

    if copy_first_teacher_layers:  # Our copying is done. We just log and save
        e_layers_to_copy, d_layers_to_copy = list(range(e)), list(range(d))
        logger.info(
            f"Copied encoder layers {e_layers_to_copy} and decoder layers {d_layers_to_copy}. Saving them to {save_path}"
        )
        student.save_pretrained(save_path)
        return student, e_layers_to_copy, d_layers_to_copy

    # Decide which layers of the teacher to copy. Not exactly alternating -- we try to keep first and last layer.
    if e_layers_to_copy is None:
        e_layers_to_copy: List[int] = pick_layers_to_copy(e, teacher_e)
    if d_layers_to_copy is None:
        d_layers_to_copy: List[int] = pick_layers_to_copy(d, teacher_d)

    try:
        copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)
        copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
    except AttributeError:  # For t5, student.model.encoder.layers is called student.encoder.block
        copy_layers(teacher.encoder.block, student.encoder.block, e_layers_to_copy)
        copy_layers(teacher.decoder.block, student.decoder.block, d_layers_to_copy)
    logger = logging.get_logger(__name__)
    logger.info(
        f"Copied encoder layers {e_layers_to_copy} and decoder layers {d_layers_to_copy}. Saving them to {save_path}"
    )
    student.config.init_metadata = dict(
        teacher_type=teacher.config.model_type,
        copied_encoder_layers=e_layers_to_copy,
        copied_decoder_layers=d_layers_to_copy,
    )
    student.save_pretrained(save_path)
    # Save information about copying for easier reproducibility

    return student, e_layers_to_copy, d_layers_to_copy

def pick_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
            )
        return list(range(n_student))


def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str

def Bart(text, Bart_path):
    tokenModel = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(tokenModel)
    model_checkpoint = "facebook/bart-large-cnn"
    if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = "" # BART-12-3
    
    max_input_length = 1024 # input, source text
    max_target_length = 256 # summary, target text
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model, list_en, list_de = create_student_by_copying_alternating_layers(model, 'trian.pth', 12, 3)
    model.load_state_dict(torch.load(Bart_path))
    print(generate_summary(text, model))

def cleanData(name):    #句子切分
    setlast = jieba.cut(name, cut_all=False)
    seg_list = [i.lower() for i in setlast]
    return " ".join(seg_list)


def calculateSimilarity(sentence, doc):  # 根据句子和句子，句子和文档的余弦相似度
    if doc == []:
        return 0
    vocab = {}
    for word in sentence.split():
        vocab[word] = 0  # 生成所在句子的单词字典，值为0

    docInOneSentence = '';
    for t in doc:
        docInOneSentence += (t + ' ')  # 所有剩余句子合并
        for word in t.split():
            vocab[word] = 0  # 所有剩余句子的单词字典，值为0

    cv = CountVectorizer(vocabulary=vocab.keys())

    docVector = cv.fit_transform([docInOneSentence])
    sentenceVector = cv.fit_transform([sentence])
    return cosine_similarity(docVector, sentenceVector)[0][0]


def TextRank(sentence):
    texts = [sentence]  # 读行
    texts = [i[:-1] if i[-1] == '\n' else i for i in texts]

    sentences = []
    clean = []
    originalSentenceOf = {}

    # Data cleansing
    for line in texts:
        parts = line.split('。')[:-1]  # 句子拆分
        #   print (parts)
        for part in parts:
            cl = cleanData(part)  # 句子切分
            #       print (cl)
            sentences.append(part)  # 原本的句子
            clean.append(cl)  # 干净有重复的句子
            originalSentenceOf[cl] = part  # 字典格式
    setClean = set(clean)  # 干净无重复的句子

    # calculate Similarity score each sentence with whole documents
    scores = {}
    for data in clean:
        temp_doc = setClean - set([data])  # 在除了当前句子的剩余所有句子
        score = calculateSimilarity(data, list(temp_doc))  # 计算当前句子与剩余所有句子的相似度
        scores[data] = score  # 得到相似度的列表
        # print score

    # calculate MMR
    n = 10 * len(sentences) / 100  # 摘要的比例大小
    alpha = 0.7
    summarySet = []
    while n > 0:
        mmr = {}
        # kurangkan dengan set summary
        for sentence in scores.keys():
            if not sentence in summarySet:
                mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence,
                                                                                            summarySet)  # 公式
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        summarySet.append(selected)
        #   print (summarySet)
        n -= 1


    summary = ""
    summary+"sasa"
    for sentence in summarySet:
        summary = summary + originalSentenceOf[sentence].lstrip('')
    return summary

def T5(text):
    model_checkpoint = 'sunhao666/chi-sum2'
    # 分词器采样的是T5的分词器
    tokenizer = AutoTokenizer.from_pretrained('uer/t5-base-chinese-cluecorpussmall')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    sentence = text
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
    return summary




if __name__ == '__main__':
    text = input("请输入句子：")
    type = "2"
    type = input("1代表bart模型，2代表textrank方法，3代表t5模型，请输入(默认为2):")
    if type=="1":
        Bart_path = "D:\课程资料\文本数据挖掘\CodeAndModel\BartModel\BART.pth" # 默认路径
        Bart_path = input("请输入Bart模型参数保存路径与文件:")
        print(Bart(text, Bart_path))
    elif type=="2":
        print(TextRank(text))
    elif type=="3":
        print(T5(text))
    
