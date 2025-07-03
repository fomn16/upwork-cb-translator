import torch
import torchaudio
import numpy as np
import time
from cusom_implementation.configs.xtts_config import XttsConfig
from cusom_implementation.models.xtts import Xtts
from TTS.api import TTS

benchmark_strings = [
    "हर सुबह एक नया आशीर्वाद और एक नया अवसर लेकर आती है।",
    "ऐसा कोई नहीं है जो खुद दर्द को प्यार करता हो!",
    "जीवन में हर दिन एक नई शुरुआत का मौका होता है।",
    "सपने वो नहीं जो हम सोते वक्त देखते हैं, सपने वो हैं जो हमें सोने नहीं देते।",
    "सफलता की कुंजी है मेहनत और धैर्य।",
    "खुश रहना एक कला है, जो हर किसी को नहीं आती।",
    "हर मुश्किल के पीछे एक बड़ा अवसर छिपा होता है।",
    "सच्चा दोस्त वही है जो मुश्किल समय में साथ खड़ा रहे।",
    "जीवन में छोटी-छोटी खुशियों को नजरअंदाज मत करो।",
    "हर दिन को ऐसे जियो जैसे यह तुम्हारा आखिरी दिन हो।",
    "सपने देखने वालों के लिए दुनिया में कुछ भी असंभव नहीं।",
    "जो लोग अपने लक्ष्य पर ध्यान केंद्रित करते हैं, वही सफल होते हैं।",
    "हर इंसान के अंदर एक अद्भुत शक्ति छिपी होती है।",
    "सच्ची खुशी दूसरों को खुश करने में है।",
    "हर समस्या का समाधान होता है, बस उसे ढूंढने की जरूरत है।",
    "जीवन में असफलता ही सफलता का पहला कदम है।",
    "जो लोग मेहनत करते हैं, किस्मत भी उन्हीं का साथ देती है।",
    "हर दिन एक नई उम्मीद लेकर आता है।",
    "सच्चा प्यार वही है जो बिना शर्त के हो।",
    "जो लोग अपने सपनों को सच करना चाहते हैं, उन्हें मेहनत करनी पड़ती है।",
    "हर इंसान के पास अपनी कहानी होती है।",
    "सकारात्मक सोच से हर मुश्किल आसान हो जाती है।",
    "जो लोग दूसरों की मदद करते हैं, वे खुद भी खुश रहते हैं।",
    "हर दिन को एक नई शुरुआत के रूप में देखो।",
    "सपने देखना आसान है, लेकिन उन्हें पूरा करना मेहनत मांगता है।",
    "जो लोग अपने डर का सामना करते हैं, वही असली विजेता होते हैं।",
    "हर इंसान के अंदर कुछ खास होता है।",
    "सच्चा सुख दूसरों की भलाई में है।",
    "हर दिन को एक नई चुनौती के रूप में स्वीकार करो।",
    "जो लोग अपने लक्ष्य के प्रति समर्पित होते हैं, वही सफल होते हैं।",
    "हर इंसान के पास अपनी ताकत और कमजोरी होती है।",
    "सकारात्मक सोच से जीवन में चमत्कार हो सकते हैं।",
    "जो लोग अपने सपनों को सच करना चाहते हैं, उन्हें कभी हार नहीं माननी चाहिए।",
    "हर दिन एक नई शुरुआत का मौका होता है।",
    "सपने वो नहीं जो हम सोते वक्त देखते हैं, सपने वो हैं जो हमें सोने नहीं देते।",
    "सफलता की कुंजी है मेहनत और धैर्य।",
    "खुश रहना एक कला है, जो हर किसी को नहीं आती।",
    "हर मुश्किल के पीछे एक बड़ा अवसर छिपा होता है।",
    "सच्चा दोस्त वही है जो मुश्किल समय में साथ खड़ा रहे।",
    "जीवन में छोटी-छोटी खुशियों को नजरअंदाज मत करो।",
    "हर दिन को ऐसे जियो जैसे यह तुम्हारा आखिरी दिन हो।",
    "सपने देखने वालों के लिए दुनिया में कुछ भी असंभव नहीं।",
    "जो लोग अपने लक्ष्य पर ध्यान केंद्रित करते हैं, वही सफल होते हैं।",
    "हर इंसान के अंदर एक अद्भुत शक्ति छिपी होती है।",
    "सच्ची खुशी दूसरों को खुश करने में है।",
    "हर समस्या का समाधान होता है, बस उसे ढूंढने की जरूरत है।",
    "जीवन में असफलता ही सफलता का पहला कदम है।",
    "जो लोग मेहनत करते हैं, किस्मत भी उन्हीं का साथ देती है।",
    "हर दिन एक नई उम्मीद लेकर आता है।"
]

# taken from the coqui streaming example code
def postprocess(wav):
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav.tobytes()

print("Downloading model...")
_, _, _, _, model_dir = TTS().download_model_by_name(
    "tts_models/multilingual/multi-dataset/xtts_v2",
)
assert model_dir is not None
print(f"model downloaded to {model_dir}")

print("Loading model...")
config = XttsConfig()
config.load_json(model_dir + "/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_dir, use_deepspeed=True)
if torch.cuda.is_available():
    model.cuda()
print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["voice-profile.wav"])

print('Starting benchmark...')
total_time_start = time.perf_counter()
full_total = []
first_byte_total = []
for text in benchmark_strings:
    first = True
    wav = bytearray()
    full_time = first_byte_time = time.perf_counter()
    chunks = model.inference_stream(
        text,
        "hi",
        gpt_cond_latent,
        speaker_embedding
    )
    for chunk in chunks:
        if(first):
            first_byte_time=time.perf_counter()-first_byte_time
            first_byte_total.append(first_byte_time)
            first = False
        wav.extend(postprocess(chunk))
    full_time=time.perf_counter()-full_time
    full_total.append(full_time)
    print(f"generated for: ${text} in ${full_time} with first byte in ${first_byte_time}")
print(f'total:{time.perf_counter() - total_time_start}')
print(f"avrg: {np.average(full_total)}")
print(f"avrg first byte: {np.average(first_byte_total)}")
    
# saving last as example
audio_np = np.frombuffer(wav, dtype=np.int16)
audio_tensor = torch.from_numpy(audio_np).unsqueeze(0) # Unsqueeze to add channel dimension (1, N)
torchaudio.save(f"testOutputs/wav{time.time()}.wav", audio_tensor, sample_rate=24000)

#result in my device:
# total:39.3006980359969
# avrg: 0.8020276750002249