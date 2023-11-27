# XAI 예제 모음

각 기술 별 예제를 모아놓았습니다.

## XAI 공통

### 음성인식 예제

#### Google ASR

설치

```
pip install SpeechRecognition
```

실행

```
python transcribe_google.py --file sample.wav
```

#### Whisper ASR


```bash
python transcribe_whisper.py --file sample.wav
```

#### NeMo ASR

##### pip requirements

```
# NeMo
nemo-toolkit==1.11
torchmetrics==0.10.3
hydra-core
braceexpand
webdataset
youtokentome
inflect
editdistance
jiwer
pytorch_lightning==1.7
librosa
transformers
pandas
sentencepiece
pyannote.core
pyannote.metrics
ipywidgets
```

manifest.json 파일을 미리 준비해야 한다.

```
python transcribe_nemo.py
```



### 위협상황 XAI 분석

```bash
python demo_plz.py
```

## 의료 XAI

## 법률 XAI