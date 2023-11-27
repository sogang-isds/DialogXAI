import json
import logging
import time

import torch
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel

torch.set_num_threads(1)

logging.basicConfig(format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

batch_size = 1
model_path = 'model/Conformer-CTC-BPE-Medium--val_wer=0.1796-epoch=340.nemo'
# model_path = 'model/Squeezeformer-CTC-BPE--val_wer=0.1808-epoch=393.nemo'

manifest_filepath = 'sample_manifest.json'

filepaths = []
references = []
total_audio_duration = 0.0

with open(manifest_filepath, 'r', encoding='utf-8') as f:
    for line in f:
        elem = json.loads(line)
        filepaths.append(elem['audio_filepath'])
        references.append(elem['text'])
        total_audio_duration += elem['duration']


# filepaths = filepaths[:1]
# references = references[:1]

logging.info(f'Num of files: {len(filepaths)}')

asr_model = ASRModel.restore_from(restore_path=model_path)

# transcribe audio
start_time = time.time()

transcriptions = asr_model.transcribe(filepaths, batch_size=batch_size, num_workers=4)
print(transcriptions)
decode_time = time.time() - start_time

wer_cer = word_error_rate(hypotheses=transcriptions, references=references, use_cer=True)


logging.info(f"Total audio duration : {total_audio_duration:.2f}")
logging.info(f"Total decode time : {decode_time:.2f}")
logging.info(f"RTF : {decode_time / total_audio_duration:.4f}")
logging.info(f"files/s : {len(filepaths) / decode_time:.2f}")
logging.info(f"CER : {wer_cer * 100:.2f}")
