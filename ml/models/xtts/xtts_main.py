import torch
import torchaudio
import numpy as np
import time
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

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
model.load_checkpoint(config, checkpoint_dir=model_dir)
if torch.cuda.is_available():
    model.cuda()
print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["voice-profile.wav"])

print('Starting benchmark...')
total_time_start = time.perf_counter()
test_times = []
for str in benchmark_strings:
    test_time_start = time.perf_counter()
    wav = model.inference(
        benchmark_strings[0],
        "hi",
        gpt_cond_latent,
        speaker_embedding
    )
    test_times.append(time.perf_counter() - test_time_start)
    print("generated for: " + str)
print(f'total:{time.perf_counter() - total_time_start}')
print(f"avrg: {np.average(test_times)}")
    
# saving last as example
torchaudio.save(f"testOutputs/wav{time.time()}.wav", torch.tensor(wav["wav"]).unsqueeze(0), sample_rate=24000)

#result in my device:
# total:102.87182618300176
# avrg: 2.0994055654692994