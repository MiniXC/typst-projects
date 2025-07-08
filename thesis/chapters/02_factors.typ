#import "../abbr.typ"
#import "../quote.typ": *

#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#import fletcher.shapes: house, hexagon
#let blob(pos, label, tint: white, width: 26mm, ..args) = node(
	pos, align(center, label),
	width: width,
	fill: tint.lighten(60%),
	stroke: 1pt + tint.darken(20%),
	corner-radius: 5pt,
	..args,
)

== Factors and representations of speech <02_factors>

#q(
  [#citep(<mcclelland_trace_1986>)],
  [#emph[The TRACE Model of Speech Perception]],
  [… we could say that speech perception is the process of forming representations of the stimulus -- the speaker’s utterance -- at several levels of description.]
)

In speech technology the acoustic signals which make up speech are represented in a number of different ways. Due to the continuous, large and highly redundant nature of the "raw" signal in audio recordings, representations such as mel spectrograms or #abbr.pla[MFCC] were used from the onset, both to more closely align with human perception and to compress the signal @flanagan_speech_1971. Since then, representations at #emph[several levels of description] have been introduced -- often aiming to encode a particular aspect of speech. In this chapter, we discuss these representations and how they relate to human perception, as well as our own contribution of a self-supervised prosodic representation model. These representations are valuable for synthetic speech evaluation since they can help us quantify both synthetic and real speech across several dimensions or factors.

#figure(
  diagram(
    spacing: 5pt,
    cell-size: (8mm, 10mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 70%,

    blob((0,0), align(center)[#text(size: 42pt, baseline: -5pt)[🗣️]#linebreak()#text(size: 21pt, baseline: 0pt)[Speech Signal]], width: 50mm, height: 23mm),
    
    // top
    edge((0,0), (-1, -1), "-|>", bend: 10deg),
    blob((-1,-1), [Audio Codecs], tint: orange, height: 11mm),
    edge((-1,-1), (-1.6,-2), "-|>", bend: -10deg, label: [Algorithmic], label-side: left, label-pos: 1),
    node((-1.7,-2.1), align(bottom)[`mp3`,`opus`,`wav`,#sym.dots]),
    edge((-1,-1), (-0.7,-2), "-|>", bend: 10deg, label: [Learned], label-side: right, label-pos: 1.4),
    node((-0.6,-2.4), align(top)[#text(baseline: 0pt, top-edge: 0pt, bottom-edge: 0pt)[
      #sym.dots
       
      EnCodec (#cite(<defossez_encodec_2023>, form: "year"))
      
      DAC (#cite(<kumar_dac_2023>, form: "year"))
    ]]),

    edge((0,0), (1, -1), "-|>", bend: -10deg),
    blob((1,-1), text(baseline: 4pt, top-edge: 0pt, bottom-edge: 0pt)[Learned
    
    Representations], tint: blue, width: 40mm, height: 11mm),
    edge((1,-1), (0.35,-1.9), "-|>", bend: 10deg, label: [Monolingual], label-side: left, label-pos: 0.6, snap-to: (auto, none)),
    node((0.35,-2.6), text(baseline: 0pt, top-edge: 0pt, bottom-edge: 0pt)[
      #sym.dots

      EAT (#cite(<chen_eat_2024>, form: "year"))
      
      HuBERT (#cite(<hsu_hubert_2021>, form: "year"))

      wav2vec 2.0 (#cite(<baevski_wav2vec_2020>, form: "year"))
    ]),
    
    edge((1.5,-1), (1.7,-2.4), "-|>", bend: -10deg, label: [Multilingual], label-side: right, label-pos: 0.6),
    node((1.7,-2.4), text(baseline: 0pt, top-edge: 0pt, bottom-edge: 0pt)[
      #sym.dots
      
      mHuBERT-147 (#cite(<boito_mhubert-147_2024>, form: "year"))

      XLS-R (#cite(<conneau_xlsr_2021>, form: "year"))
    ]),
    

    // prosody
    edge((0,0), (-1,1), "-|>", bend: -10deg),
    blob((-1,1), [Prosody], tint: green, width: auto),
    edge((-1,1), (-1.4,1.7), "-|>", bend: -10deg, label: [Algorithmic], label-side: right, label-pos: 0.85, snap-to: (auto, none)),
    node((-1.4,2.5), text(baseline: 0pt, top-edge: 0pt, bottom-edge: 0pt)[
      
      WORLD pitch (#cite(<morise_world_2016>, form: "year"))

      Energy

      #sym.dots
    ]),
    edge((-1,1), (-0.65,1.7), "-|>", bend: 10deg, label: [Learned], label-side: left, label-pos: .9, snap-to: (auto, auto)),
    node((-0.3,2.5), text(baseline: 0pt, top-edge: 0pt, bottom-edge: 0pt)[
      
      MPM (#cite(<wallbridge_mpm_2025>, form: "year"))

      ProsodyBERT (#cite(<hu_prosodybert_2023>, form: "year"))

      #sym.dots
    ]),

    // speaker
    edge((0,0), (0.5,1), "-|>", bend: -30deg),
    blob((0.5,1), [Speaker], tint: maroon, width: auto),
    node((0.6,2), text(baseline: 0pt, top-edge: 0pt, bottom-edge: 0pt)[
      
      ECAPA-TDNN (#cite(<desplanques_ecapa_2020>, form: "year"))

      d-vector (#cite(<wan_generalized_2018>, form: "year"))

      #sym.dots
    ]),

    // ambient
    edge((0.5,0), (1.4,0.3), "-|>", bend: 30deg),
    blob((1.3,0.3), [Ambient], tint: olive, width: auto),
    node((1.3,1), text(baseline: 0pt, top-edge: 0pt, bottom-edge: 0pt)[
      
      #abbr.s[SRMR] (#cite(<santos_improved_2014>, form: "year"))

      #abbr.s[WADA] #abbr.s[SNR] (#cite(<kim_wada_2008>, form: "year"))

      #sym.dots
    ]),
  ),
  placement: none,
  caption: "Overview of representations of speech",
) <representations>

=== Self-supervised learning representations <02_ssl>

A prolific field of study when it comes to speech representations is #abbr.a[SSL]. In this a model is trained to predict pseudo-labels created from the data itself. The first of these models were introduced for #abbr.a[NLP] in which a percentage of tokens in the original data are masked and said models are trained to predict the masked tokens using cross entropy loss @devlin_bert_2019.

This methodology was adapted to speech by later works @schneider_wav2vec_2019@baevski_wav2vec_2020@hsu_hubert_2021@chen_wavlm_2022. These approaches typically process speech into 20-millisecond chunks which are passed through a #abbr.a[CNN] and discretised. HuBERT @hsu_hubert_2021 and WavLM @chen_wavlm_2022 use iterative clustering, while wav2vec and wav2vec 2.0 use vector quantisation, with discretisation using Gumbel softmax in the latter. Due to this, the latter family of models relies on contrastive loss while the former can use categorical cross entropy as in BERT @devlin_bert_2019. The resulting representations achieve state-of-the-art results on several benchmarks and challenges @evain_lebenchmark_2021@yang_superb_2021@shi_mlsuperb_2023@tsai_superb-sg_2022@wang_fine-tuned_2021 -- they have also been expanded to cover many languages beyond English @conneau_xlsr_2021@boito_mhubert-147_2024. However useful these models and their representations are for downstream tasks, it is not clear in which ways the latent spaces learned by these models correlate with human perception. For example, @millet_humanlike_2022 find that #abbr.pla[MFCC] predict systematic effects of contrasts between specific pairs of phones better than self-supervised models. Similar effects can be observed with tasks such as word segmentation @pasad_words_2024, and different layers of the model have been shown to correlate with different perceptual units @pasad_layer-wise_2021. It therefore stands to reason that while these representations are useful for a wide range of tasks, they do not intuitively correlate with human perception. In the following sections, we discuss more specific representations which more transparently correlate with different aspects of human perception.

=== Audio Codecs

Audio codecs (short for coder-decoder) are designed to compress and decompress audio data streams. Their primary goal is to reduce the amount of data required to store or transmit an audio signal, making them essential for applications ranging from streaming music to telephony. Codecs can be broadly categorised as either lossless or lossy. Lossless codecs, such as #abbr.pla[FLAC], allow for the perfect reconstruction of the original digital audio signal, but offer limited compression. Lossy codecs, on the other hand, achieve much higher compression ratios by permanently discarding information that is deemed perceptually irrelevant to human listeners @brandenburg_mp3_1999. Given the large size of raw audio data, lossy codecs are of particular interest in speech technology. These can be further divided into two families: traditional algorithmic codecs and modern learned neural codecs.

Traditional codecs like MP3 and Opus rely on psychoacoustic models @brandenburg_mp3_1999. These models exploit the limitations of human auditory perception, such as frequency masking (a loud sound making a quieter, nearby frequency inaudible) and temporal masking (a sound being masked by another that occurs just before or after). By identifying and removing these perceptually masked components of the signal, these codecs can significantly reduce file size while maintaining high perceived quality. Opus, in particular, has become a standard for real-time interactive applications like video conferencing due to its low latency and high efficiency across a wide range of bitrates @valin_opus_2012.

More recently, a new class of learned or neural audio codecs has emerged, which use deep learning to achieve unprecedented compression. These models, such as EnCodec @defossez_encodec_2023 and DAC @kumar_dac_2023, typically employ an autoencoder architecture. An encoder network maps the raw audio waveform to a compact, quantised latent representation, and a decoder network reconstructs the audio from this representation. By training end-to-end to minimise the reconstruction error, these models learn to preserve the most perceptually salient information, achieving quality comparable or superior to traditional codecs at significantly lower bitrates. For example, EnCodec uses residual vector quantisation to create a hierarchical representation that can be streamed in real-time @defossez_encodec_2023, while DAC demonstrates high-fidelity reconstruction for general audio at bitrates as low as 1.5 kbps @kumar_dac_2023.

The discrete tokens produced by these neural codecs constitute a powerful representation of the speech signal. Unlike the self-supervised representations discussed in @02_ssl, which are optimised for downstream discriminative tasks, the goal of a codec's representation is faithful reconstruction. This property has made neural codec tokens a popular target for modern generative models. Instead of predicting complex spectrograms or waveforms directly, many state-of-the-art text-to-speech and voice conversion systems now generate these discrete audio tokens, which are then converted back to audio using a pre-trained neural codec decoder @wang_valle_2023@borsos_soundstorm_2023. This approach simplifies the generation task and has led to significant advances in the quality and controllability of synthetic speech. It can be applied to audio in general or speech alone @wu_codecsuperb_2024.

=== Perceptually-grounded representations <02_perceptual>

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

==== Speaker <02_speaker>

Speaker characteristics relate to the quality of a speakers voice uncontrollable by them, dictated by the morphology and motor control of their vocal tract. This can be modeled in several ways, based on the task at hand, which are mainly speaker recognition (#emph[Who is speaking?]) and speaker verification (#emph[Is this specific person speaking?]). However, we focus primarily on #abbr.a[DNN] representations which can be useful for either task in this work. These are referred to as speaker embeddings @snyder_x_2018@wan_generalized_2018@desplanques_ecapa_2020. As of the time of writing frequently used methods include x-vectors @snyder_x_2018, d-vectors @wan_generalized_2018 and ECAPA-TDNN @desplanques_ecapa_2020. For TTS conditioning, no large performance differences have been found across speaker embedding systems @stan_spkanalysis_2023 -- similarly, while some embeddings perform better for specific tasks, no clear best system has emerged @zhao_probing_2022.

==== Ambient

Lastly ambient acoustics or environmental effects are shaped by recording conditions more generally. They can include the microphone being used for recording the speech and its distance to the speaker, the acoustics of the space of recording (e.g. leading to reverberation) and background noise. These effects are sometimes overlooked in speech research, as a recent hallmark work on dysarthria detection has shown @schu_silence_2023: Higher performance was achieved when using non-speech segments than when using speech segments, indicating most previous methods had relied on ambient acoustics rather than speech characteristics for this task. Correlates for ambient acoustics include reverberation in the form of #abbr.a[SRMR] @kinoshita_reverb_2013 and more general estimates of the "quality" of the signal when, for example, transferred over telephone networks such as #abbr.a[PESQ] @rix_pesq_2001. Another measure is the #abbr.a[SNR] which can be estimated using the amplitude distribution using #abbr.a[WADA] @kim_wada_2008.

==== For synthetic speech

Prosodic correlates have been widely used in #abbr.a[TTS], as we outline in the next chapter. The same is true for speaker representations. Ambient measures are more commonly used for filtering the training data than in the architecture of generative models.