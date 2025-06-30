import asyncio
import logging
import numpy as np
import time
from auralis import TTS, TTSRequest, TTSOutput
import torch
import cusom_implementation # This import must stay here even if not used directly in this file
from auralis.common.definitions.enhancer import AudioPreprocessingConfig

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
async def test_async(tts:TTS):
    print('Starting benchmark...')
    total_time_start = time.perf_counter()
    test_times = []
    test_times_beginning_audio = []
    
    conditioningRequest = TTSRequest(
        text="हर सुबह एक नया आशीर्वाद और एक नया अवसर लेकर आती है।",
        speaker_files=["voice-profile.wav"],
        stream=True,
        language='hi',
        enhance_speech=False,
        temperature=0.7,
        top_p=0.8,
        audio_config=AudioPreprocessingConfig(
            normalize=False,
            trim_silence=False,
            remove_noise=False,
            enhance_speech=False
        )
    )
    conditioning_partial = await tts.prepare_for_streaming_generation(conditioningRequest)

    print('finished conditioning')
    
    for str in benchmark_strings:
        outputs = []
        test_time_start = time.perf_counter()
        request = TTSRequest(
            text=str,
            speaker_files=["voice-profile.wav"],
            stream=True,
            context_partial_function=conditioning_partial,
            language='hi',
            enhance_speech=False,
            temperature=0.7,
            top_p=0.8,
            audio_config=AudioPreprocessingConfig(
                normalize=False,
                trim_silence=False,
                remove_noise=False,
                enhance_speech=False
            )
        )
        first = True
        start_time = time.perf_counter()
        
        res = await tts.generate_speech_async(request)
        async for output in res: # type: ignore
            if first:
                start_time = time.perf_counter() - start_time
                first = False
        output.to_bytes()
        outputs.append(output)
        
        test_times.append(time.perf_counter() - test_time_start)
        test_times_beginning_audio.append(start_time)
        print("generated for: " + str)
    print(f'total test time:{time.perf_counter() - total_time_start}')
    print(f"avrg time for complete audio: {np.average(test_times)}")
    print(f"avrg time for start of audio: {np.average(test_times_beginning_audio)}")
        
    # saving last as example
    TTSOutput.combine_outputs(outputs).save(f"./testOutputs/wav{time.time()}.wav")

tts = TTS().from_pretrained(
    "AstraMindAI/xttsv2",
    gpt_model='AstraMindAI/xtts2-gpt',
    torch_dtype=torch.float16)
asyncio.run(test_async(tts))
#result in my device:
# total test time: 85.24953444200003
# avrg time for complete audio: 1.7286368300000021
# avrg time for start of audio: 1.6205630161836713