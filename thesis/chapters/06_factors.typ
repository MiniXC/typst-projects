#import "../abbr.typ"
#import "../quote.typ": q

== Factors and representations of speech <06_factors>

#q(
  [James L. McClelland and Jeffrey L. Elman],
  [#emph[The TRACE Model of Speech Perception], 1986 @mcclelland_trace_1986],
  [… we could say that speech perception is the process of forming representations of the stimulus -- the speaker’s utterance -- at several levels of description.]
)

In speech technology the acoustic signals which make up speech are represented in a number of different ways. Due to the continuous, large and highly redundant nature of the "raw" signal in audio recordings, representations such as mel spectrograms or #abbr.pla[MFCC] were used from the onset, both to more closely align with human perception and to compress the signal @flanagan_speech_1971. Since then, representations at #emph[several levels of description] have been introduced -- often aiming to encode a particular aspect of speech. In this chapter, we discuss these representations and how they relate to human perception, as well as our own contribution of a self-supervised prosodic representation model. These representations are valuable for synthetic speech evaluation since they can help us quantify both synthetic and real speech across several dimensions or factors.

=== Self-supervised learning representations

A prolific field of study when it comes to speech representations is #abbr.a[SSL]. In this a model is trained to predict pseudo-labels created from the data itself. The first of these models were introduced for #abbr.a[NLP] in which a percentage of tokens in the original data are masked and said models are trained to predict the masked tokens using cross entropy loss @devlin_bert_2019.

This methodology was adapted to speech by later works @schneider_wav2vec_2019@baevski_wav2vec_2020@hsu_hubert_2021@chen_wavlm_2022. These approaches typically process speech into 20-millisecond chunks which are passed through a #abbr.a[CNN] and discretised. HuBERT @hsu_hubert_2021 and WavLM @chen_wavlm_2022 use iterative clustering, while wav2vec and wav2vec 2.0 use vector quantisation, with discretisation using Gumbel softmax in the latter. Due to this, the latter family of models relies on contrastive loss while the former can use categorical cross entropy as in BERT @devlin_bert_2019. The resulting representations achieve state-of-the-art results on several benchmarks and challenges @evain_lebenchmark_2021@yang_superb_2021@shi_mlsuperb_2023@tsai_superb-sg_2022@wang_fine-tuned_2021 -- they have also been expanded to cover many languages beyond English @conneau_unsupervised_2021@boito_mhubert-147_2024. However useful these models and their representations are for downstream tasks, it is not clear in which ways the latent spaces learned by these models correlate with human perception. For example, @millet_humanlike_2022 find that #abbr.pla[MFCC] predict systematic effects of contrasts between specific pairs of phones better than self-supervised models. Similar effects can be observed with tasks such as word segmentation @pasad_words_2024, and different layers of the model have been shown to correlate with different perceptual units @pasad_layer-wise_2021. It therefore stands to reason that while these representations are useful for a wide range of tasks, they do not intuitively correlate with human perception. In the following sections, we discuss more specific representations which more transparently correlate with different aspects of human perception.

=== Perceptually-grounded representations <06_perceptual>

In this work, we divide non-verbal perception of speech into three categories.
- *Prosody* relates to features the speaker can control independently of lexical/verbal content.
- *Speaker* relates to features in the speech that are present due to the speakers biology, in particular their vocal tract, but could also include any other characteristics the speaker does not consciously alter.
- *Ambient* relates to features that have to do with recording and environmental conditions.

We outline these in more detail in the following sections. There are also nuances in these definitions for which are outside the scope of this work -- for example, by our definition, changes to speech by impersonators altering their voice quality to imitate another speaker would be classed as prosody, while prosody is often defined as only including changes to the speech that carry non-verbal meaning, which does not include impersonation @jurafsky_slp_2008.

==== Prosody

Prosody does not have one universal definition, but we follow @jurafsky_slp_2008 in particularly focusing on features the speaker can vary independently of the phone string, such as F0, duration and energy. We now outline the most common ways these features are estimated from the raw waveform or spectrogram and which uses they have seen in the literature. The following is a description of common prosodic correlates -- they are #emph[correlates] since there is no easily defined ground truth -- as perception depends on the listener and can differ from person to person.

*Fundamental frequency (F0)* is the pitch at which the vocal folds vibrate. Both algorithmic and deep learning approaches have been proposed to estimate this. On the algorithmic side the DIO algorithm @morise_dio_2009 is commonly used. A deep-learned-based approach has recently gained popularity in pitch-estimating neural networks, however, if they universally generalise to unseen datasets is uncertain @morrison_penn_2023. These pitch values are commonly used as prosodic correlates for #abbr.a[TTS] @ren_fastspeech_2019@hayashi_espnet-tts_2020@ren_fastspeech_2021 -- outside of this, they are also useful for predicting other prosodic components such as prominences and boundaries @suni_hierarchical_2017.

*Duration*, refers to the duration of different units present in speech. A common value of interest is *speaking rate*, which can be defined as the rate of phones produced per unit of time. Some deep learning approaches have been tried to estimate this value @tomashenko_sr_2014, but state-of-the art works most commonly estimate the number of phones first and then divide by the length of the audio @lyth_parler_2024. This can be achieved by using #abbr.a[g2p] conversion if a transcript exists (or speech recognition can be performed with reasonable accuracy) or using a phone recogniser @li_allosaurus_2020. Duration is implicitly modeled by forced alignment models @mcauliffe_montreal_2017 -- their output in turn is used to explicitly predict duration in some #abbr.a[TTS] models @ren_fastspeech_2019@lancucki_fastpitch_2021@ren_fastspeech_2021. Recently it has also been used to condition TTS systems more generally on speaking rate using prompts @lyth_parler_2024.

*Energy*, also referred to as loudness or intensity, can be estimated is trivially -- it is the magnitude of the audio signal or spectrogram. It is rarely used outside of simple #abbr.a[VAD] methods and has proved less useful than pitch for expressive #abbr.a[TTS] @lancucki_fastpitch_2021 - however, both pitch and energy are necessary components for conveying emotion in speech @jurafsky_slp_2008@haque_energypitch_2017.

==== Speaker <06_speaker>

Speaker characteristics relate to the quality of a speakers voice uncontrollable by them, dictated by the morphology and motor control of their vocal tract. This can be modeled in several ways, based on the task at hand, which are mainly speaker recognition (#emph[Who is speaking?]) and speaker verification (#emph[Is this specific person speaking?]). However, we focus primarily on #abbr.a[DNN] representations which can be useful for either task in this work. These are referred to as speaker embeddings @snyder_x_2018@wan_generalized_2018@desplanques_ecapa_2020. As of the time of writing frequently used methods include x-vectors @snyder_x_2018, d-vectors @wan_generalized_2018 and ECAPA-TDNN @desplanques_ecapa_2020. For TTS conditioning, no large performance differences have been found across speaker embedding systems @stan_spkanalysis_2023 -- similarly, while some embeddings perform better for specific tasks, no clear best system has emerged @zhao_probing_2022.

==== Ambient

Lastly ambient acoustics or environmental effects are shaped by recording conditions more generally. They can include the microphone being used for recording the speech and its distance to the speaker, the acoustics of the space of recording (e.g. leading to reverberation) and background noise. These effects are sometimes overlooked in speech research, as a recent hallmark work on dysarthria detection has shown @schu_silence_2023: Higher performance was achieved when using non-speech segments than when using speech segments, indicating most previous methods had relied on ambient acoustics rather than speech characteristics for this task. Correlates for ambient acoustics include reverberation in the form of #abbr.a[SRMR] @kinoshita_reverb_2013 and more general estimates of the "quality" of the signal when, for example, transferred over telephone networks such as #abbr.a[PESQ] @rix_pesq_2001. Another measure is the #abbr.a[SNR] which can be estimated using the amplitude distribution using #abbr.a[WADA] @kim_wada_2008.

==== For synthetic speech

As explored in @04_correlates, prosodic correlates have been widely used in #abbr.a[TTS]. The same is true for speaker representations. Ambient measures are more commonly used for filtering the training data than in the architecture of generative models.