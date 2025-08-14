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

== Factors and representations for speech synthesis <02_factors>

#q(
  [#citep(<mcclelland_trace_1986>)],
  [#emph[The TRACE Model of Speech Perception]],
  [… we could say that speech perception is the process of forming representations of the stimulus -- the speaker’s utterance -- at several levels of description.]
)

In speech technology, the acoustic signals which make up speech can be represented in a number of different ways. In this work, we consider the raw audio waveform recording as the fundamental representation of a speech signal $S$ -- these can have different fidelity, in the form of sampling rate, the frequency of samples per unit of time and bit depth, the number of possible values per sample. For many differing uses, ranging from compression to conveying the meaning of the speech signal, other representations of this fundamental signal can be useful. All these representations are derived by applying a transformation function, denoted as $cal(R)$, to this waveform.

// TODO: more gradual introduction, we're jumping too quickly into detail as usual. Talks about the use of representations first, leave math def. for later

#figure(
  image("../figures/2/invertible_to_reductive.png", width: 100%),
  caption: [Examples of an invertible transformation in the form #abbr.a[STFT], lossy in the form of Mel spectrogram and reductive transformations in the form energy and pitch extraction, as well as possible reconstruction approaches.],
  placement: top,
) <fig_transformations>

=== Transformation Classification <02_class>

We classify the aforementioned transformations by the category and purpose of the resulting representation. First, we discuss the categories, which are invertible (fully reconstructible), lossy transformations (partially reconstructible), and reductive transformations (not reconstructible), with some overlap between lossy and reductive. Throughout this section, for illustrative purposes, we will use the commonly used Mel spectrogram and related representations as an example. Further representations are discussed in @02_representations.

==== Invertible Transformations
These are lossless transformations where the original waveform can be perfectly reconstructed, such that:
$
cal(R)^(-1)(cal(R)(S)) = S
$
An example is the #abbr.a[STFT], which retains both magnitude (real) and phase (imaginary), allowing for exact reconstruction via the inverse #abbr.a[STFT] (iSTFT), as shown in @fig_transformations.

==== Lossy-Reconstructible Transformations
These transformations are intentionally lossy but are designed to retain sufficient information for a perceptually similar waveform to be synthesized. The inverse operation is an approximation, often performed by a dedicated model or algorithm $f$ with learned or configured parameters $theta$. This could be a vocoder, decoder or algorithm such as the one by #citep(<griffin_griffinlim_1984>). The resulting speech is now $tilde(S)$, since it is not reconstructed perfectly and subject to change based on the model or algorithm used, resulting in the following:

$
f_theta (cal(R)(S)) = Syn
$

These representations almost always maintain the time-series nature of the original speech, with a direct mapping of specific frames in the original corresponding to specific parts of the resulting representation $cal(R)(S)$. @fig_transformations shows the Mel spectrogram as an example, for which after performing the aforementioned #abbr.a[STFT], the spectrum is scaled according to the perceptual Mel scale @volkmann_melscale_1937 and the phase is discarded. The Mel scale is used to align more closely with human perception (as our perception of pitch is not linear), while the phase is discarded as it does not contain as much perceptually relevant information @hidaka_phaseclass_2022. Therefore, Mel spectrograms only retain a magnitude spectrum of the speech, which nevertheless has been shown to retain enough information for deep learning models to learn to create a speech signals closely resembling the original#footnote[However, algorithmically reconstructed speech (i.e. using Griffin-Lim) using Mel spectrograms alone @lee_bigvgan_2023, without the phase information, results in highly degraded speech poorly rated by listeners when compared to reconstruction with phase @webber_autovocoder_2023.].

==== Reductive Transformations
These transformations distill specific attributes from the signal into a representation that is not intended for direct audio reconstruction. We class representations as reductive if they cannot be reconstructed without leading to many-to-one scenarios where many waveforms clearly different from the original could lead to the same representation. 
In the example in @fig_transformations, we show estimated energy derived by taking the average frame-level spectral magnitude sum and the fundamental frequency estimation algorithm Yin @decheveign_yin_2002 respectively.
However, reductive transformations can also be far removed from the original signal, such as the most probable text sequence $hat(T)$ as predicted by an #abbr.l[ASR] model.

==== Edge cases
Some transformations are not clearly reductive or lossy, such as most #abbr.a[SSL]-derived latent representations. They #emph[could] be used to reconstruct a speech signal, as previous work has shown that a combination of #abbr.a[SSL] representations and speaker embeddings can be reconstructed with similar performance as Mel spectrograms @siuzdak_wavthruvec_2022. However, they could also lead to significant deviation from the original depending on the information encoded by any particular #abbr.a[SSL] model. The layer at which the representation is extracted might also play a role, with earlier layers often corresponding more closely to acoustics @pasad_layer-wise_2021.

=== Purpose of Transformations

Different representations of speech can be used for differing purposes. Here, we discuss these representations and #emph[what] they encode (e.g. the whole signal, speaker identity, lexical content), #emph[how] they encode and potentially decode the signal and the #emph[why] the resulting representations might be used in speech synthesis.

==== The #emph[what] of encoding speech
Speech can be represented at "several levels of description" @mcclelland_trace_1986 -- each level aimed to encode a particular aspect of speech. These refer to differing hierarchical abstractions of the speech signal, ranging from low-level acoustic features (e.g., raw waveforms) to mid-level perceptual correlates (e.g., pitch and energy) and high-level semantic or contextual embeddings (e.g., speaker identity or prosodic patterns). In our work, we also focus on the #emph[factors] of speech which can have levels within them. For example, the #smallcaps[Prosody] factor of speech could include the relatively low-level aforementioned perceptual correlates in addition to more high-level features such as stress markers. Any representation does not have to be pure in the sense of only belonging to one factor: It could be a combination, or represent all (or most) factors present in speech, in which case we class them as #smallcaps[Generic]. Our terminology gives us a way to classify and describe factors such as referring to WavLM @chen_wavlm_2022 embeddings as #emph[high-level] and #smallcaps[Generic], or the energy contour as #emph[low-level] and #smallcaps[Prosody]. We can also put this framework into context with the previous classifications of transformations outlined in @02_class: Invertible transformations need to encode the whole signal and are therefore #emph[low-level] #smallcaps[Generic]. However, for lossy ones, high perceptual similarity rather than an exact copy of the original recording is desirable, which allows for a higher level of abstraction, often making them #emph[mid-level]. Finally, reductive transformations can be even more selective with what they encode, as reconstruction is not a concern, and can be #emph[high-level] abstractions as well as limit themselves to just encode one factor -- although they can still be #smallcaps[Generic] or a combination of several factors.

#figure(
  image("../figures/2/representation_overview.png", width: 100%),
  caption: [Overview of representations according to factors (y-axis) and level of abstraction (x-axis).],
  placement: top,
) <fig_factors>

==== The #emph[how] of encoding speech
The process of deriving a representation by applying a transformation function $cal(R)$ to a signal $S$ can be broadly categorized into three distinct methodologies, which often correspond to different levels of abstraction as illustrated in @fig_factors. These are algorithmic, learned, and conceptual transformations.
#emph[Algorithmic Transformations], also referred to as signal-based, rely on a predefined set of mathematical rules and signal processing techniques. The transformation function $cal(R)$ is a fixed algorithm whose parameters are not learned from data. Examples include the #abbr.a[STFT], the calculation of energy, or the estimation of fundamental frequency using algorithms like Yin @decheveign_yin_2002. These methods are typically transparent and computationally efficient. They form the basis for most low-level representations, corresponding to the green dashed boxes in @fig_factors, providing direct, measurable correlates of the acoustic signal.
#emph[Learned Transformations] are data-driven. In this paradigm, the transformation function $cal(R)$ is a model, typically a deep neural network, whose parameters are optimized to perform a specific task on a large dataset. This can involve training a model to reconstruct its input (an autoencoder), to predict a masked portion of the signal (predictive coding), or to classify the input. The resulting representations, such as #abbr.a[SSL] embeddings @chen_wavlm_2022, speaker embeddings @snyder_x_2018, or neural codec tokens @defossez_encodec_2023, are often highly effective for downstream tasks because they capture complex, non-linear patterns in the data. These representations, shown as the blue solid boxes in @fig_factors, can span from mid- to high-levels of abstraction.
#emph[Conceptual Transformations] yield high-level, usually discrete labels that represent an abstract concept within the speech signal. For these representations, we extend our definition of the transformation $cal(R)$ to encompass not just algorithms, but also complex inference models or even the human annotation process itself. In the case of human annotation, the transformation $cal(R)$ can be seen as the cognitive process of the human listener, guided by a set of annotation criteria (e.g., a rubric for emotional expression), which maps the continuous perceptual experience of the speech signal $S$ onto a discrete symbolic label. The resulting representations, like "Intent" or "Acoustic Scene," are reductive by nature and correspond to the red-hatched boxes in @fig_factors. They are invaluable for supervised training and evaluation, providing a semantic grounding for more abstract models.

==== The #emph[why] of encoding speech
The purpose of any given speech representation can be understood by its position on a spectrum ranging from purely #emph[acoustic] to purely #emph[semantic]. Representations at the extremes of this spectrum serve distinct functions, while many of the most useful representations exist in the middle, acting as a bridge between the two domains.
At the purely #emph[acoustic] end of the spectrum lie representations whose primary goal is to capture the physical properties of the sound wave with the highest possible fidelity. Raw waveforms contain the most acoustic information, with lossless codecs like #abbr.pla[FLAC] containing the same level of acoustic information as well, as their representations are designed for perfect, bit-for-bit reconstruction of the original signal. The primary use for these representations is high-fidelity storage, transmission, and ultimately this level of representation has to be reached as the final step by any synthesis system aimed at producing audio.
At the opposite, purely #emph[semantic] end are representations that encode abstract meaning, content, and concepts, completely detached from the acoustic signal that conveyed them. The final text transcript from an ASR system or a conceptual "Intent" label are examples. The purpose of these representations can be analysis, understanding, and symbolic reasoning. They are the inputs and outputs of systems concerned with what is being said, by whom, and for what purpose.
The vast majority of speech processing involves bridging the gap between these two poles, and thus many representations encode aspects of both. These #emph[intermediate] representations are designed to make acoustic information tractable for semantic tasks, or to guide the generation of acoustic signals from semantic information. For example, a Mel spectrogram is fundamentally acoustic, yet it is structured in a perceptually relevant way that makes it easier for a model to recognize phonetic content, making semantic content easier to access and model. #smallcaps[Prosody] features like a pitch contour are acoustic measurements, but their shape and trajectory can convey semantic meaning such as if the speaker is expressing a certain emotion or asking a question. Conversely, a #smallcaps[Speaker] embedding is a high-level abstraction of identity (a semantic concept), but it is learned by distilling patterns directly from acoustic signals. Another #emph[intermediate] representation are high-level #abbr.a[SSL] embeddings, which are learned from raw audio with the explicit goal of being maximally useful for a wide range of downstream semantic tasks.

=== A Factor-Based Review of Speech Representations <02_representations>

We will now examine the factors of speech as laid out in @fig_factors in more detail, contextualising their roles across various speech technology applications. We follow a bottom-up order from the figure, reordered to reflect a common conceptual flow from abstract semantics to the concrete acoustic signal.

==== Semantic Representations

The #smallcaps[Semantic] factor encompasses the linguistic content of speech. Its representations lie firmly at the semantic pole of our spectrum and are central to any task involving understanding or generating language.
One of the most foundational and frequently used representations is the *Phone Sequence*. This representation converts the orthographic sequence of letters into a sequence of phonemes, the basic sound units of a language. This transformation, known as #abbr.a[g2p] conversion, resolves ambiguities in pronunciation (e.g., the different sounds of "ough" in "through," "tough," and "though") and provides a canonical representation of pronunciation. In TTS, it serves as a clean input to the acoustic model. In ASR, especially for low-resource languages, predicting a phoneme sequence can be a useful intermediate step before word-level transcription. Multilingual models like Allosaurus provide phone recognition for a vast number of languages, facilitating such cross-lingual applications @li_allosaurus_2020.

On the more abstract side, modern systems for both ASR and TTS process *Text* using powerful encoders, often using conextualised language models like BERT @devlin_bert_2019. These produce *Contextualised Text Embeddings*, which are high-dimensional vectors that capture not just the identity of words but also their meaning within the specific context of the sentence. In TTS, this helps generate more natural prosody @kakouros_what_2023. In Spoken Language Understanding (SLU), these embeddings are the basis for interpreting the user's commands. At the highest level of abstraction, *Intent* represents the speaker's pragmatic goal. In conversational AI, an ASR system first produces text, which is then classified into an intent label. This label then drives the dialogue manager and determines the content and style of the TTS response, completing the interaction loop.

==== Speaker Representations <02_speaker>

The #smallcaps[Speaker] factor encompasses the unique, identifying characteristics of an individual's voice. Its representations bridge the gap between acoustic reality and the semantic concept of identity.

At a low level, identity is encoded in physical properties. *Formant Frequencies*, the resonant peaks in the speech spectrum, are determined by the size and shape of the speaker's vocal tract and are thus a strong anatomical correlate @stevens_sources_1972. A speaker's habitual *Average Pitch* is another simple indicator. However, these features are highly variable and deeply entangled with other factors, making them unreliable for robust speaker identification.

To create a more robust representation, systems distill these variable acoustic cues into high-level, learned *Speaker Embeddings*. These fixed-dimensional vectors are extracted using a network trained to discriminate between thousands of speakers. Architectures like x-vectors @snyder_x_2018 and ECAPA-TDNN @desplanques_ecapa_2020 excel at this, learning to create a representation that is maximally sensitive to inter-speaker differences while being invariant to intra-speaker variability (like phonetic content). These embeddings are the cornerstone of *Speaker Verification* and *Diarization* @bredin_pyannote_2023, as well as generative tasks like multi-speaker TTS and voice cloning @li_styletts_2023. Beyond core timbre, other *Identifying Characteristics* contribute to our perception of a speaker, such as perceived *Age*, *Gender*, and *Accent*. While these can be treated as explicit semantic labels, they are often learned implicitly as part of a powerful speaker embedding, which captures not just anatomy but also these habitual, identity-linked speaking patterns.

==== Prosody Representations

The #smallcaps[Prosody] factor is an #emph[intermediate] factor, governing the melody and rhythm of speech that links acoustic signals to emotional and pragmatic meaning.

Low-level prosodic features provide a quantitative link between acoustics and perception. *Duration* specifies the length of phones and pauses, defining speaking rate and rhythm. Analytically, it helps distinguish speaking styles; generatively, it allows for explicit tempo control in TTS, with ground-truth data often obtained from a *Forced Aligner* @mcauliffe_montreal_2017. *Pitch* (F0) and *Energy* (loudness) contours are likewise essential. In emotion recognition, rising pitch and high energy are strong acoustic cues for excitement. In controllable synthesis, these contours can be explicitly predicted to shape the melody and stress of the output speech @lancucki_fastpitch_2021.

For a more holistic view, mid-to-high-level representations are used. *Phonologic Labels*, derived from linguistic theories like ToBI @ross_prediction_1996, provide a symbolic, semantic description of acoustic events like pitch accents and boundary tones, useful for linguistic analysis. A more data-driven approach is to learn a continuous, high-level *Prosody Embedding*. This vector can serve as an input feature for an emotion classifier or as a conditioning signal for style transfer in voice conversion and expressive TTS @wang_style_2018. At the highest level, a discrete *Emotion* label sits at the semantic end, representing the final interpretation of a complex set of underlying acoustic cues.

==== Ambient Representations

The #smallcaps[Ambient] factor encompasses all environmental aspects of a recording. The utility of its representations depends entirely on whether the task goal is analysis of the real world or synthesis of an ideal one.

For analytical tasks like ASR, modeling ambient acoustics is critical. Systems must be robust to background noise and reverberation. Low-level metrics like *Signal-to-Noise Ratio (SNR)* @kim_wada_2008 or *Reverberation Time* @kinoshita_reverb_2013 can be used as features for a speech enhancement front-end. High-level *Acoustic Scene* classification can help a system adapt its acoustic models to specific conditions like being "in a car," attaching the acoustic environment to a semantic label.

Conversely, for generative tasks like TTS, the goal is typically to produce clean speech. Here, the ambient factor is something to be eliminated. Ambient representations are therefore primarily used for quality control during data preparation, ensuring that the training data is as free from acoustic contamination as possible.

==== Generic Representations

The #smallcaps[Generic] factor includes representations that encode the full acoustic signal. They span the entire spectrum, from the purely acoustic waveform to abstract embeddings that bridge to the semantic domain.

Historically, *Spectrogram-Derived Representations* have been the classic bridge between acoustics and semantics. These include the *Mel spectrogram* and *Perceptual Linear Predictive (PLP)* analysis, along with its popular variant, Mel-Frequency Cepstral Coefficients (MFCCs). These mid-level features transform the complex waveform into a more tractable, perceptually-motivated format that highlights phonetic differences, making them the standard input for decades of ASR and speaker ID systems, and a common output target for TTS acoustic models.

Closer to the acoustic pole are representations from *Neural Audio Codecs*. Models like EnCodec @defossez_encodec_2023 learn to convert a waveform into a sequence of discrete tokens, $A_"codec"$, for the primary purpose of high-fidelity, reconstructible compression. While they are learned, their goal is faithful acoustic representation, making them ideal as a high-quality output target for generative models like TTS @wang_valle_2023.

Finally, high-level *Contextualised Speech Embeddings* from #abbr.a[SSL] models like WavLM @chen_wavlm_2022 are #emph[intermediate] representations. Learned from raw acoustics, they are explicitly optimized to be useful for a wide range of downstream semantic tasks. They are reductive yet encode rich, entangled information about all factors. This makes them a state-of-the-art auxiliary feature for nearly any speech task, capable of improving robustness and performance by providing a powerful, pre-trained understanding of the relationship between sound and meaning.

// TODO for this chapter: small intro on SSL? also tree figure with specific representations
// also talk about how w2v etc can be seen as semantic but we class them as generic?

==== Limitations and Cross-Factor Phenomena

While the factorized view provides a useful framework, it is important to acknowledge its limitations. Certain acoustic phenomena do not fit neatly into a single category, but rather exist at the intersection of multiple factors, highlighting the complex relationship between acoustics and semantics.

One prominent example is *voice quality* or *phonation mode*. A speaker can alter their phonation to be *breathy* or to *whisper*. Breathiness can be a stable characteristic of the #smallcaps[Speaker], but it can also be used dynamically to convey an intimate emotion, making it a feature of #smallcaps[Prosody]. Whispering is an extreme case that removes the fundamental frequency entirely, yet speakers can still convey prosodic contrasts through other acoustic cues. Thus, voice quality is a low-level acoustic characteristic that provides evidence for higher-level interpretations in both the #smallcaps[Speaker] and #smallcaps[Prosody] factors.

Other phenomena are inherently cross-factor. *Sarcasm*, for instance, arises from a deliberate mismatch between the literal #smallcaps[Semantic] content (e.g., "What a wonderful day") and a contradictory #smallcaps[Prosody] (e.g., a flat intonation). The meaning is not in the words or the melody alone, but in the cognitive dissonance their conflict creates. Similarly, a speaker's *register* or *speaking style* (e.g., formal presentation vs. casual conversation) is a meta-level choice that simultaneously modulates all other factors, from word choice (#smallcaps[Semantic]) to intonation patterns (#smallcaps[Prosody]). These examples underscore that while our factorized model is a powerful analytical tool, a complete understanding of speech requires appreciating the complex, dynamic interplay between the acoustic and semantic domains.

// === Representations <02_representations>

// Perceptually grounded representations are designed to capture specific aspects of speech that align closely with human auditory processing. These representations provide intuitive, interpretable features that correlate directly with perceptual phenomena, serving as building blocks for more complex models. In this work, we divide reductive transformations of a speech signal $S$ into three perceptually-grounded categories, which can be used for conditioning, i.e., as part of a set $Z$.
// - *Prosody* relates to features the speaker can control independently of lexical/verbal content.
// - *Speaker* relates to features in the speech that are present due to the speakers biology, in particular their vocal tract, but could also include any other characteristics the speaker does not consciously alter.
// - *Ambient* relates to features that have to do with recording and environmental conditions.

// We outline these in more detail in the following sections, highlighting their perceptual relevance before transitioning to how self-supervised models build upon these foundations. There are also nuances in these definitions which are outside the scope of this work -- for example, by our definition, changes to a signal $S$ by impersonators altering their voice quality to imitate another speaker would be classed as prosody, while prosody is often defined as only including changes to the speech that carry non-verbal meaning, which does not include impersonation @jurafsky_slp_2008.

// ==== Prosody

// Prosody does not have one universal definition, but we follow #citep(<jurafsky_slp_2008>) in particularly focusing on features the speaker can vary independently of the transcript $T$, such as F0, duration, and energy. We now outline how these features, which we can denote as numerical conditioning signals like $"F0"$, are estimated from a speech signal $S$. The following is a description of common prosodic correlates -- they are #emph[correlates] since there is no easily defined ground truth. This is due to perception depending on the listener and can differ from person to person.

// *Fundamental frequency (F0)* is the pitch at which the vocal folds vibrate. Both algorithmic and deep learning approaches have been proposed to estimate this from $S$. On the algorithmic side the DIO algorithm @morise_dio_2009 is commonly used. A deep-learned-based approach has recently gained popularity in pitch-estimating neural networks, however, if they universally generalise to unseen datasets is uncertain @morrison_penn_2023. The estimated pitch values, which can be denoted as a time-series $"F0"$, are commonly used as a conditioning signal in $Z$ for #abbr.a[TTS] @ren_fastspeech_2019@hayashi_espnet-tts_2020@ren_fastspeech_2021 -- outside of this, they are also useful for predicting other prosodic components such as prominences and boundaries @suni_hierarchical_2017.

// *Duration* refers to the length of different units present in a speech signal $S$. A common value of interest is *speaking rate*, which can be defined as the number of phones (from a transcript $T$) per unit of time in $S$. While some deep learning approaches have tried to estimate this directly from $S$ @tomashenko_sr_2014, state-of-the-art works most commonly derive it from $T$ and the duration of $S$ @lyth_parler_2024. Forced alignment models @mcauliffe_montreal_2017, which align a transcript $T$ with its corresponding signal $S$, model duration implicitly. Their output is in turn used to create an explicit duration conditioning signal for some #abbr.a[TTS] models @ren_fastspeech_2019@lancucki_fastpitch_2021@ren_fastspeech_2021.

// *Energy*, also referred to as loudness or intensity, is the magnitude of the signal $S$ or its spectrogram. It is rarely used outside of simple #abbr.a[VAD] methods and has proved less useful than pitch for expressive #abbr.a[TTS] @lancucki_fastpitch_2021 - however, both pitch and energy are necessary components for conveying emotion in speech @jurafsky_slp_2008@haque_energypitch_2017.

// // expand out: especially speaker

// ==== Speaker <02_speaker>

// Speaker characteristics relate to the qualities of a speaker's voice inherent to their biology (e.g., vocal tract anatomy), as well as other factors like accent or pathology that the speaker may not consciously alter. In TTS, these are typically captured using speaker embeddings, a type of embedding-based conditioning signal, extracted from a reference audio signal. These embeddings are then included in the conditioning set $Z$ to guide synthesis. As of the time of writing frequently used methods include x-vectors @snyder_x_2018, d-vectors @wan_generalized_2018 and ECAPA-TDNN @desplanques_ecapa_2020. For TTS conditioning, no large performance differences have been found across speaker embedding systems @stan_spkanalysis_2023 -- similarly, while some embeddings perform better for specific tasks, no clear best system has emerged @zhao_probing_2022. These systems generally take a fixed-length input signal (e.g., 10 seconds of audio) and produce a single high-dimensional vector that encapsulates speaker identity.

// ==== Ambient

// Lastly, ambient acoustics or environmental effects are properties of a signal $S$ shaped by its recording conditions more generally. They can include the microphone being used for recording and its distance to the speaker, the acoustics of the space of recording (e.g. leading to reverberation) and background noise. These effects are sometimes overlooked in speech research, as a recent work on dysarthria detection has shown @schu_silence_2023: Higher performance was achieved when using non-speech segments than when using speech segments from $S$, indicating most previous methods had relied on ambient acoustics rather than speech characteristics for this task. This overfitting likely occurred because ambient features (e.g., room noise or microphone artifacts) provided unintended shortcuts for classification, masking the true speech-related signals of dysarthria. These can be quantified using reductive transformations to produce numerical conditioning signals such as $"SRMR"$ for reverberation @kinoshita_reverb_2013, $"PESQ"$ for quality estimates over telephone networks @rix_pesq_2001, or $"SNR"$ for signal-to-noise ratio @kim_wada_2008.

// === Self-supervised learning representations <02_ssl>

// Self-supervised learning (SSL) methods learn more holistic, data-driven representations from the raw signal. These approaches provide versatile features that underpin many modern speech tasks, bridging the gap between perceptual correlates and high-level abstractions.

// // was image processing first?
// // expand where there is a long list of references in a row

// In #abbr.a[SSL], a model is trained to predict pseudo-labels created from the input signal $S$ itself. The first of these models were introduced for #abbr.a[NLP] in which a percentage of tokens in the original data are masked and said models are trained to predict the masked tokens using cross entropy loss @devlin_bert_2019. This methodology was adapted to speech by later works @schneider_wav2vec_2019@baevski_wav2vec_2020@hsu_hubert_2021@chen_wavlm_2022. At a high level, these models learn to reconstruct or predict masked portions of the input audio, often by first discretizing the signal into tokens (e.g., via clustering or vector quantization) and then using contrastive or cross-entropy losses to distinguish or regenerate the correct tokens. The resulting representations achieve state-of-the-art results on several benchmarks and challenges @evain_lebenchmark_2021@yang_superb_2021@shi_mlsuperb_2023@tsai_superb-sg_2022@wang_fine-tuned_2021 -- they have also been expanded to cover many languages beyond English @conneau_xlsr_2021@boito_mhubert-147_2024. However useful these models and their representations are for downstream tasks, it is not clear in which ways the latent spaces learned by these models correlate with human perception. For example, #citep(<millet_humanlike_2022>) find that #abbr.pla[MFCC] predict systematic effects of contrasts between specific pairs of phones better than self-supervised models. Similar effects can be observed with tasks such as word segmentation @pasad_words_2024, and different layers of the model have been shown to correlate with different perceptual units @pasad_layer-wise_2021. It therefore stands to reason that while these reductive transformations are useful for a wide range of tasks, they do not intuitively correlate with human perception in the same transparent way as the perceptual correlates discussed earlier.

// #comic.comic((80mm, 40mm), "Comic overview of SSL training process, showing audio input being masked and predicted", blue) <fig_ssl_process>

// // avoid referring to review paper directly (this should be self-contained)

// // add information about them and maybe group them with modeling mel spectrograms

// === Audio Codecs

// Finally, we consider audio codecs, which provide reconstructible representations optimized for compression and fidelity. While less central to our work than perceptual or self-supervised representations, codecs play a key role in modern TTS pipelines, particularly for efficient generation of high-quality waveforms. For overviews of audio codec development, see @brandenburg_mp3_1999 and @defossez_encodec_2023.

// Audio codecs are designed to compress and decompress audio data streams. Their primary goal is to reduce the amount of data required to store or transmit a signal $S$. Codecs can be broadly categorised as either lossless or lossy. Lossless codecs, such as #abbr.pla[FLAC], are examples of invertible transformations that allow for the perfect reconstruction of the original signal $S$, but offer limited compression. Lossy codecs, on the other hand, achieve much higher compression ratios by permanently discarding information that is deemed perceptually irrelevant to human listeners @brandenburg_mp3_1999. Given the large size of raw audio data, these reconstructible transformations are of particular interest in speech technology. These can be further divided into two families: algorithmic codecs and modern learned neural codecs.

// Algorithmic codecs like MP3 and Opus rely on psychoacoustic models @brandenburg_mp3_1999. These models exploit the limitations of human auditory perception, such as frequency masking (a loud sound making a quieter, nearby frequency inaudible) and temporal masking (a sound being masked by another that occurs just before or after). By identifying and removing these perceptually masked components of the signal $S$, these codecs can significantly reduce file size while maintaining high perceived quality. Opus, in particular, has become a standard for real-time interactive applications like video conferencing due to its low latency and high efficiency across a wide range of bitrates @valin_opus_2012.

// More recently, a new class of learned or neural audio codecs has emerged, which use deep learning to achieve unprecedented compression. These models, such as EnCodec @defossez_encodec_2023 and DAC @kumar_dac_2023, typically employ an autoencoder architecture. An encoder network applies a transformation $r$ to the raw waveform $S$ to create a compact, quantised latent representation $r(S)$, and a decoder network $f_theta$ reconstructs the audio from this representation to produce a synthetic signal $Syn = f_theta (r(S))$. By training end-to-end to minimise the reconstruction error between $S$ and $Syn$ (often using losses like L1 or perceptual metrics), these models learn to preserve the most perceptually salient information, achieving quality comparable or superior to algorithmic codecs at significantly lower bitrates. For example, EnCodec uses residual vector quantization—a technique that iteratively quantizes residuals from previous layers to build a hierarchical codebook—to create a hierarchical representation that can be streamed in real-time @defossez_encodec_2023, while DAC demonstrates high-fidelity reconstruction for general audio at bitrates as low as 1.5 kbps @kumar_dac_2023. Neural codecs can be semantic (focusing on content) or acoustic (focusing on fidelity), with acoustic ones often preferred for TTS.

// The discrete tokens $r(S)$ produced by these neural codecs constitute a powerful, reconstructible representation of the speech signal. Unlike self-supervised representations, which are optimised for downstream discriminative tasks, the goal of a codec's representation is faithful reconstruction. This property has made neural codec tokens a popular target for modern generative models. Instead of predicting complex spectrograms or waveforms directly, many state-of-the-art systems now generate these discrete audio tokens, which are then converted to a waveform $Syn$ using a pre-trained neural codec decoder $f_theta$ @wang_valle_2023@borsos_soundstorm_2023. This approach simplifies the generation task and has led to significant advances in the quality and controllability of synthetic speech. It can be applied to audio in general or speech alone @wu_codecsuperb_2024.

// #comic.comic((80mm, 40mm), "neural codec (enc-dec)", orange) <fig_codec_process>


// === Masked prosody model

// Building on the perceptually grounded representations of prosody outlined above, we now turn to learned approaches that capture prosodic structure in a more flexible, data-driven manner. While the self-supervised learning methods outlined in the next section create powerful reductive representations from the full speech signal $S$, we apply a similar approach to its prosodic components alone. This allows for the study of prosodic structure independent of lexical content (from $T$) or speaker identity. We call this the #abbr.l[MPM] @wallbridge_mpm_2025.

// Inspired by masked language models @devlin_bert_2019, the #abbr.s[MPM] is trained to reconstruct corrupted sequences of prosodic features. It takes three parallel input streams corresponding to numerical conditioning signals: fundamental frequency, energy, and voice activity, all derived from an input signal $S$. These continuous features are first quantised into discrete codebooks. Then, random segments of the input sequences are masked, and the model, which uses a Conformer @gulati_conformer_2020 architecture, is trained to predict the original, unmasked codes. This self-supervised objective forces the model to learn the inherent "systematicity" of prosody — how pitch, loudness, and timing features co-vary and predict each other over time.

// We find that the utility of the learned representations depends on the timescale of the masking strategy. For tasks requiring fine-grained local information, such as syllable segmentation, models trained with smaller masks perform better. Conversely, for tasks involving longer-term dependencies, like emotion recognition, larger masks are more effective. A general-purpose representation can be achieved by using randomly sized masks during training, which yields robust performance across a range of tasks. When compared to both the raw prosodic features and more constrained hierarchical representations like the Continuous Wavelet Transform (#abbr.s[CWT]) @grossmann_cwt_1984, the #abbr.s[MPM] representations provide significantly more predictive power, particularly for complex perceptual labels like emotion and phrasal boundaries. More surprisingly, for prosody-centric tasks like prominence and boundary detection, the #abbr.s[MPM] is competitive with, and in some cases surpasses, large-scale #abbr.s[SSL] models like HuBERT trained on the full speech signal $S$. This suggests that for certain tasks, a representation derived from a specific subset of $Z$ (e.g., $\{z^"F0", z^"Energy", z^"VAD"\}$) can make salient information more accessible than a general-purpose representation where it could be entangled with phonetic and speaker content. This makes the #abbr.s[MPM] a promising tool for both analysing and generating prosody in synthetic speech, while ensuring only prosodic information is used.

// #comic.comic((80mm, 40mm), "Masked Prosody Model architecture", green) <fig_mpm_arch>