import time
import torch
import string
import pandas as pd
import speech_recognition as sr
import soundfile
import librosa.display
from IPython.display import display, Audio
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from transformers import AutoTokenizer
import argparse
import json
import logging
import glob
from transformers import AutoTokenizer
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from captum.attr import (
    visualization
)
from difflib import SequenceMatcher

model = BertForSequenceClassification.from_pretrained('../../../../Transformer-Explainability/modelss').to("cuda:0")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# initialize the explanations generator
explanations = Generator(model)
classificationsdict = {"0": "협박", "1": "일반대화", "2": "갈취및공갈", "3": "직장내괴롭힘", "4": "기타괴롭힘"}
classifications = ["0", "1", "2", "3", "4"]
r = sr.Recognizer()
with sr.AudioFile(f'./demo/t2_590.wav') as source:
    audio = r.record(source)  # 전체 audio file 읽기
asr = r.recognize_google(audio, language='ko')

print('*****************위협상황 분류 및 설명 가능 대화 시스템 데모*****************')
fileinput = input("BOT : 분석하실 파일명을 입력하세요. - ")
print(f'\n\n{fileinput}파일을 입력받았습니다.\n\n')
print('**********음성인식 진행 중 입니다.*********\n\n')
print(f'BOT : 음성인식 결과 입니다 : {asr}\n\n')
# i="어머님! 따님 데뷔는 저희만 믿으시라니까요! 저희 딸이 정말 연예인 되는 건가요? 물론이죠. 지금 연습생으로 열심히 연습 중이에요. 감사합니다. 그럼 잘 좀 부탁드리겠습니다. 제가 따님은 다른 연습생과는 다르게 직접 관리 중이에요. 신경 써주셔서 감사합니다. 데뷔하려면 돈 필요한 건 아시죠? 2천만 원 입금하시면 됩니다. 네? 돈이오? 따님 연습생 생활은 저희가 땅 파서 하는 게 아니잖아요. 직접 관리까지 하는데 그 정도 성의는 보이셔야 될 거 같은데요? 아무리 그래도.. 따님 데뷔하는 걸 많이 기대하고 있던데. 어머님 때문에 데뷔 못하는 걸 알면 어떨까요? 알겠어요.. 무슨 일이 있어도 준비해서 보내드릴게요."
encoding = tokenizer(asr, return_tensors='pt')
input_ids = encoding['input_ids'].to("cuda:0")
attention_mask = encoding['attention_mask'].to("cuda:0")

# true class is positive - 1
true_class = 1

# generate an explanation for the input
expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
# normalize scores
expl = (expl - expl.min()) / (expl.max() - expl.min())
tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
import re

position_id = [0]
init = 0
for i in range(1, len(tokens)):
    if '#' in tokens[i]:
        position_id.append(init)
    else:
        init += 1
        position_id.append(init)
num = 0
string = ''
wordlist = []
for i in range(len(position_id)):
    if position_id[i] == num:
        string += re.sub('#+', '', tokens[i])
    else:
        wordlist.append(string)
        num += 1
        string = ''
        string += re.sub('#+', '', tokens[i])
expllist = expl.cpu().detach()
num = 0
attentions = 0
attentionlist = []
for i in range(len(position_id)):
    if position_id[i] == num:
        attentions += expllist[i]
    else:
        attentionlist.append(attentions)
        num += 1
        attentions = 0
        attentions += expllist[i]

output = torch.nn.functional.softmax(model(input_ids=input_ids, attention_mask=attention_mask)[0], dim=-1)
classification = output.argmax(dim=-1).item()
# get class name
class_name = classificationsdict[str(classification)]
# if the classification is negative, higher explanation scores are more negative
# flip for visualization
# if class_name == "NEGATIVE":
# expl *= (-1)

# tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
init = 0
windowlist = []
for ww in range(len(wordlist) // 2):
    if init + 5 > len(wordlist):
        newindexlist = [i for i in range(init, len(wordlist))]
    else:
        newindexlist = [i for i in range(init, init + 5)]
    windowlist.append(newindexlist)
    init += 2
# print(windowlist)
attdict = {}
for index, j in enumerate(windowlist):
    attlist = [attentionlist[att].item() for att in j]
    attresult = sum(attlist)
    attdict[index] = attresult
# print(attdict)
finalresult = sorted(attdict.items(), key=lambda x: x[1], reverse=True)
evidencelist = []
for result in finalresult[:2]:
    ids, res = result
    evidence = ''
    for chars in windowlist[ids]:
        evidence += ' ' + wordlist[chars]
    evidence = evidence.strip()
    evidencelist.append(evidence)

if SequenceMatcher(None, evidencelist[0], evidencelist[1]).ratio() > 0.5:
    ids, res = finalresult[3]
    evidence = ''
    for chars in windowlist[ids]:
        evidence += ' ' + wordlist[chars]
    evidence = evidence.strip()
    evidencelist[1] = evidence

print('*****************설명 가능성 대화 시스템 답변 출력*****************\n')
print(f'BOT : 해당 대화는 <{class_name}>로 분류되었습니다.\n')
users = input(f'USER : ')
print(f'\nBOT : 해당 대화가 <{class_name}>로 분류된 이유는 대화속의"{evidencelist[0]}" 발화와 "{evidencelist[1]}"발화 때문으로 판단됩니다.')
