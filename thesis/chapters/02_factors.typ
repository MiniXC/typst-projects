#import "../abbr.typ"
#import "../comic.typ"
#import "../quote.typ": *
#import "../math.typ": *
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

In speech technology, the acoustic signals which make up speech, denoted as $S$, are represented in a number of different ways. Due to the continuous, large, and highly redundant nature of the raw waveform of $S$, various representations $cal(R)(S)$ have been used from the onset. These include reconstructible transformations like Mel spectrograms and reductive ones like #abbr.pla[MFCC], both to more closely align with human perception and to compress the signal @flanagan_speech_1971. Since then, representations at #emph[several levels of description] have been introduced -- often aiming to encode a particular aspect of speech. These refer to differing hierarchical abstractions of the speech signal, ranging from low-level acoustic features (e.g., raw waveforms) to mid-level perceptual correlates (e.g., pitch and energy) and high-level semantic or contextual embeddings (e.g., speaker identity or prosodic patterns). These levels allow models to capture different facets of speech, from physical properties to perceptual and linguistic interpretations, enabling tasks like synthesis and recognition. In this chapter, we discuss these representations and how they relate to human perception, as well as our own contribution of a self-supervised prosodic representation model. We first outline perceptually grounded representations, which directly correlate with human auditory processing, and then broader learned and codec-based approaches.

#figure(
  scale(x: 85%, y: 85%)[
  #diagram(
    spacing: 7pt,
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
  )],
  placement: top,
  caption: "Overview of representations of speech",
) <representations>

=== Perceptually-grounded representations <02_perceptual>

Perceptually grounded representations are designed to capture specific aspects of speech that align closely with human auditory processing. These representations provide intuitive, interpretable features that correlate directly with perceptual phenomena, serving as building blocks for more complex models. In this work, we divide reductive transformations of a speech signal $S$ into three perceptually-grounded categories, which can be used for conditioning, i.e., as part of a set $Z$.
- *Prosody* relates to features the speaker can control independently of lexical/verbal content.
- *Speaker* relates to features in the speech that are present due to the speakers biology, in particular their vocal tract, but could also include any other characteristics the speaker does not consciously alter.
- *Ambient* relates to features that have to do with recording and environmental conditions.

We outline these in more detail in the following sections, highlighting their perceptual relevance before transitioning to how self-supervised models build upon these foundations. There are also nuances in these definitions which are outside the scope of this work -- for example, by our definition, changes to a signal $S$ by impersonators altering their voice quality to imitate another speaker would be classed as prosody, while prosody is often defined as only including changes to the speech that carry non-verbal meaning, which does not include impersonation @jurafsky_slp_2008.

==== Prosody

Prosody does not have one universal definition, but we follow #citep(<jurafsky_slp_2008>) in particularly focusing on features the speaker can vary independently of the transcript $T$, such as F0, duration, and energy. We now outline how these features, which we can denote as numerical conditioning signals like $"F0"$, are estimated from a speech signal $S$. The following is a description of common prosodic correlates -- they are #emph[correlates] since there is no easily defined ground truth. This is due to perception depending on the listener and can differ from person to person.

*Fundamental frequency (F0)* is the pitch at which the vocal folds vibrate. Both algorithmic and deep learning approaches have been proposed to estimate this from $S$. On the algorithmic side the DIO algorithm @morise_dio_2009 is commonly used. A deep-learned-based approach has recently gained popularity in pitch-estimating neural networks, however, if they universally generalise to unseen datasets is uncertain @morrison_penn_2023. The estimated pitch values, which can be denoted as a time-series $"F0"$, are commonly used as a conditioning signal in $Z$ for #abbr.a[TTS] @ren_fastspeech_2019@hayashi_espnet-tts_2020@ren_fastspeech_2021 -- outside of this, they are also useful for predicting other prosodic components such as prominences and boundaries @suni_hierarchical_2017.

*Duration* refers to the length of different units present in a speech signal $S$. A common value of interest is *speaking rate*, which can be defined as the number of phones (from a transcript $T$) per unit of time in $S$. While some deep learning approaches have tried to estimate this directly from $S$ @tomashenko_sr_2014, state-of-the-art works most commonly derive it from $T$ and the duration of $S$ @lyth_parler_2024. Forced alignment models @mcauliffe_montreal_2017, which align a transcript $T$ with its corresponding signal $S$, model duration implicitly. Their output is in turn used to create an explicit duration conditioning signal for some #abbr.a[TTS] models @ren_fastspeech_2019@lancucki_fastpitch_2021@ren_fastspeech_2021.

*Energy*, also referred to as loudness or intensity, is the magnitude of the signal $S$ or its spectrogram. It is rarely used outside of simple #abbr.a[VAD] methods and has proved less useful than pitch for expressive #abbr.a[TTS] @lancucki_fastpitch_2021 - however, both pitch and energy are necessary components for conveying emotion in speech @jurafsky_slp_2008@haque_energypitch_2017.

==== Speaker <02_speaker>

Speaker characteristics relate to the qualities of a speaker's voice inherent to their biology (e.g., vocal tract anatomy), as well as other factors like accent or pathology that the speaker may not consciously alter. In TTS, these are typically captured using speaker embeddings, a type of embedding-based conditioning signal, extracted from a reference audio signal. These embeddings are then included in the conditioning set $Z$ to guide synthesis. As of the time of writing frequently used methods include x-vectors @snyder_x_2018, d-vectors @wan_generalized_2018 and ECAPA-TDNN @desplanques_ecapa_2020. For TTS conditioning, no large performance differences have been found across speaker embedding systems @stan_spkanalysis_2023 -- similarly, while some embeddings perform better for specific tasks, no clear best system has emerged @zhao_probing_2022. These systems generally take a fixed-length input signal (e.g., 10 seconds of audio) and produce a single high-dimensional vector that encapsulates speaker identity.

==== Ambient

Lastly, ambient acoustics or environmental effects are properties of a signal $S$ shaped by its recording conditions more generally. They can include the microphone being used for recording and its distance to the speaker, the acoustics of the space of recording (e.g. leading to reverberation) and background noise. These effects are sometimes overlooked in speech research, as a recent work on dysarthria detection has shown @schu_silence_2023: Higher performance was achieved when using non-speech segments than when using speech segments from $S$, indicating most previous methods had relied on ambient acoustics rather than speech characteristics for this task. This overfitting likely occurred because ambient features (e.g., room noise or microphone artifacts) provided unintended shortcuts for classification, masking the true speech-related signals of dysarthria. These can be quantified using reductive transformations to produce numerical conditioning signals such as $"SRMR"$ for reverberation @kinoshita_reverb_2013, $"PESQ"$ for quality estimates over telephone networks @rix_pesq_2001, or $"SNR"$ for signal-to-noise ratio @kim_wada_2008.

=== Masked prosody model

Building on the perceptually grounded representations of prosody outlined above, we now turn to learned approaches that capture prosodic structure in a more flexible, data-driven manner. While the self-supervised learning methods outlined in the next section create powerful reductive representations from the full speech signal $S$, we apply a similar approach to its prosodic components alone. This allows for the study of prosodic structure independent of lexical content (from $T$) or speaker identity. We call this the #abbr.l[MPM] @wallbridge_mpm_2025.

Inspired by masked language models @devlin_bert_2019, the #abbr.s[MPM] is trained to reconstruct corrupted sequences of prosodic features. It takes three parallel input streams corresponding to numerical conditioning signals: fundamental frequency, energy, and voice activity, all derived from an input signal $S$. These continuous features are first quantised into discrete codebooks. Then, random segments of the input sequences are masked, and the model, which uses a Conformer @gulati_conformer_2020 architecture, is trained to predict the original, unmasked codes. This self-supervised objective forces the model to learn the inherent "systematicity" of prosody — how pitch, loudness, and timing features co-vary and predict each other over time.

We find that the utility of the learned representations depends on the timescale of the masking strategy. For tasks requiring fine-grained local information, such as syllable segmentation, models trained with smaller masks perform better. Conversely, for tasks involving longer-term dependencies, like emotion recognition, larger masks are more effective. A general-purpose representation can be achieved by using randomly sized masks during training, which yields robust performance across a range of tasks. When compared to both the raw prosodic features and more constrained hierarchical representations like the Continuous Wavelet Transform (#abbr.s[CWT]) @grossmann_cwt_1984, the #abbr.s[MPM] representations provide significantly more predictive power, particularly for complex perceptual labels like emotion and phrasal boundaries. More surprisingly, for prosody-centric tasks like prominence and boundary detection, the #abbr.s[MPM] is competitive with, and in some cases surpasses, large-scale #abbr.s[SSL] models like HuBERT trained on the full speech signal $S$. This suggests that for certain tasks, a representation derived from a specific subset of $Z$ (e.g., $\{z^"F0", z^"Energy", z^"VAD"\}$) can make salient information more accessible than a general-purpose representation where it could be entangled with phonetic and speaker content. This makes the #abbr.s[MPM] a promising tool for both analysing and generating prosody in synthetic speech, while ensuring only prosodic information is used.

#comic.comic((80mm, 40mm), "Masked Prosody Model architecture", green) <fig_mpm_arch>

=== Self-supervised learning representations <02_ssl>

Self-supervised learning (SSL) methods learn more holistic, data-driven representations from the raw signal. These approaches provide versatile features that underpin many modern speech tasks, bridging the gap between perceptual correlates and high-level abstractions.

In #abbr.a[SSL], a model is trained to predict pseudo-labels created from the input signal $S$ itself. The first of these models were introduced for #abbr.a[NLP] in which a percentage of tokens in the original data are masked and said models are trained to predict the masked tokens using cross entropy loss @devlin_bert_2019. This methodology was adapted to speech by later works @schneider_wav2vec_2019@baevski_wav2vec_2020@hsu_hubert_2021@chen_wavlm_2022. At a high level, these models learn to reconstruct or predict masked portions of the input audio, often by first discretizing the signal into tokens (e.g., via clustering or vector quantization) and then using contrastive or cross-entropy losses to distinguish or regenerate the correct tokens. The resulting representations achieve state-of-the-art results on several benchmarks and challenges @evain_lebenchmark_2021@yang_superb_2021@shi_mlsuperb_2023@tsai_superb-sg_2022@wang_fine-tuned_2021 -- they have also been expanded to cover many languages beyond English @conneau_xlsr_2021@boito_mhubert-147_2024. However useful these models and their representations are for downstream tasks, it is not clear in which ways the latent spaces learned by these models correlate with human perception. For example, #citep(<millet_humanlike_2022>) find that #abbr.pla[MFCC] predict systematic effects of contrasts between specific pairs of phones better than self-supervised models. Similar effects can be observed with tasks such as word segmentation @pasad_words_2024, and different layers of the model have been shown to correlate with different perceptual units @pasad_layer-wise_2021. It therefore stands to reason that while these reductive transformations are useful for a wide range of tasks, they do not intuitively correlate with human perception in the same transparent way as the perceptual correlates discussed earlier.

#comic.comic((80mm, 40mm), "Comic overview of SSL training process, showing audio input being masked and predicted", blue) <fig_ssl_process>

=== Audio Codecs

Finally, we consider audio codecs, which provide reconstructible representations optimized for compression and fidelity. While less central to our work than perceptual or self-supervised representations, codecs play a key role in modern TTS pipelines, particularly for efficient generation of high-quality waveforms. For overviews of audio codec development, see @brandenburg_mp3_1999 and @defossez_encodec_2023.

Audio codecs are designed to compress and decompress audio data streams. Their primary goal is to reduce the amount of data required to store or transmit a signal $S$. Codecs can be broadly categorised as either lossless or lossy. Lossless codecs, such as #abbr.pla[FLAC], are examples of invertible transformations that allow for the perfect reconstruction of the original signal $S$, but offer limited compression. Lossy codecs, on the other hand, achieve much higher compression ratios by permanently discarding information that is deemed perceptually irrelevant to human listeners @brandenburg_mp3_1999. Given the large size of raw audio data, these reconstructible transformations are of particular interest in speech technology. These can be further divided into two families: algorithmic codecs and modern learned neural codecs.

Algorithmic codecs like MP3 and Opus rely on psychoacoustic models @brandenburg_mp3_1999. These models exploit the limitations of human auditory perception, such as frequency masking (a loud sound making a quieter, nearby frequency inaudible) and temporal masking (a sound being masked by another that occurs just before or after). By identifying and removing these perceptually masked components of the signal $S$, these codecs can significantly reduce file size while maintaining high perceived quality. Opus, in particular, has become a standard for real-time interactive applications like video conferencing due to its low latency and high efficiency across a wide range of bitrates @valin_opus_2012.

More recently, a new class of learned or neural audio codecs has emerged, which use deep learning to achieve unprecedented compression. These models, such as EnCodec @defossez_encodec_2023 and DAC @kumar_dac_2023, typically employ an autoencoder architecture. An encoder network applies a transformation $r$ to the raw waveform $S$ to create a compact, quantised latent representation $r(S)$, and a decoder network $f_theta$ reconstructs the audio from this representation to produce a synthetic signal $Syn = f_theta (r(S))$. By training end-to-end to minimise the reconstruction error between $S$ and $Syn$ (often using losses like L1 or perceptual metrics), these models learn to preserve the most perceptually salient information, achieving quality comparable or superior to algorithmic codecs at significantly lower bitrates. For example, EnCodec uses residual vector quantization—a technique that iteratively quantizes residuals from previous layers to build a hierarchical codebook—to create a hierarchical representation that can be streamed in real-time @defossez_encodec_2023, while DAC demonstrates high-fidelity reconstruction for general audio at bitrates as low as 1.5 kbps @kumar_dac_2023. Neural codecs can be semantic (focusing on content) or acoustic (focusing on fidelity), with acoustic ones often preferred for TTS.

The discrete tokens $r(S)$ produced by these neural codecs constitute a powerful, reconstructible representation of the speech signal. Unlike self-supervised representations, which are optimised for downstream discriminative tasks, the goal of a codec's representation is faithful reconstruction. This property has made neural codec tokens a popular target for modern generative models. Instead of predicting complex spectrograms or waveforms directly, many state-of-the-art systems now generate these discrete audio tokens, which are then converted to a waveform $Syn$ using a pre-trained neural codec decoder $f_theta$ @wang_valle_2023@borsos_soundstorm_2023. This approach simplifies the generation task and has led to significant advances in the quality and controllability of synthetic speech. It can be applied to audio in general or speech alone @wu_codecsuperb_2024.

#comic.comic((80mm, 40mm), "neural codec (enc-dec)", orange) <fig_codec_process>