#import "../abbr.typ"
#import "../comic.typ"
#import "../quote.typ": *
#import "../math.typ": *

== Factors and Representations of Speech <02_factors>

#q(
  [#citep(<mcclelland_trace_1986>)],
  [#emph[The TRACE Model of Speech Perception]],
  [… we could say that speech perception is the process of forming representations of the stimulus -- the speaker’s utterance -- at several levels of description.]
)

In speech technology, the acoustic signals which make up speech can be represented in a number of different ways. In this thesis, we consider the audio waveform recording as the fundamental representation of a speech signal $S$.#footnote[Although with differing fidelity, in the form of sampling rate, bit depth and the number of possible values per sample.] Other representations of these signals can be useful, for example for compression of the signal or for conveying the content of the speech. In this thesis, we define all these representations as being derived by applying a transformation function, denoted as $cal(R)$, to this waveform.

#figure(
  image("../figures/2/invertible_to_reductive.png", width: 100%),
  caption: [Examples of an invertible transformation in the form #abbr.a[STFT], lossy in the form of mel spectrogram and reductive transformations in the form energy and pitch extraction, as well as possible reconstruction approaches.],
  placement: top,
) <fig_transformations>

=== Transformation Classification <02_class>

We classify the aforementioned transformation functions by the category and purpose of the resulting representation. First, we discuss the categories, which are invertible transformations (fully reconstructible), lossy-reconstructible transformations (partially reconstructible), and reductive transformations (not reconstructible), with some overlap between lossy-reconstructible and reductive representations. Throughout this section, for illustrative purposes, we will use the commonly used mel spectrogram and related representations as an example. Further representations are discussed in @02_representations.

==== Invertible Transformations
These are lossless transformations where the original waveform can be perfectly reconstructed, as shown in @eqt:invertibe_t.
$
cal(R)^(-1)(cal(R)(S)) = S
$ <invertibe_t>
An example is the #abbr.a[STFT], which retains both magnitude (real) and phase (imaginary), allowing for exact reconstruction via the inverse #abbr.a[STFT] (iSTFT), while making frequencies more accessible, as shown in @fig:fig_transformations.

==== Lossy-Reconstructible Transformations
Another class of transformations are intentionally lossy but are designed to retain sufficient information for a perceptually similar waveform to be synthesised -- the lossy-reconstructible transformations. The inverse operation is an approximation, often performed by a dedicated model or algorithm $f$ with learned or configured parameters $theta$. This could be a vocoder, decoder or algorithm such as the one by #citep(<griffin_griffinlim_1984>). The resulting speech is now $tilde(S)$, since it is not reconstructed perfectly and subject to change based on the model or algorithm used, resulting in the following:

$
f_theta (cal(R)(S)) = Syn
$

These representations almost always maintain the time-series nature of the original speech, with a direct mapping of specific frames in the original corresponding to specific parts of the resulting representation $cal(R)(S)$. @fig:fig_transformations shows the mel spectrogram as an example, for which after performing the aforementioned #abbr.a[STFT], the spectrum is scaled according to the perceptual mel scale @volkmann_melscale_1937 and the phase is discarded. The mel scale is used to align more closely with human perception (as our perception of pitch is not linear), while the phase is discarded as it does not contain as much perceptually relevant information @hidaka_phaseclass_2022. Therefore, mel spectrograms only retain a magnitude spectrum of the speech, which nevertheless has been shown to retain enough information for deep learning models to learn to create a speech signals closely resembling the original.#footnote[However, algorithmically reconstructed speech (i.e. using #citea(<griffin_griffinlim_1984>) algorithm) using mel spectrograms alone @lee_bigvgan_2023, without the phase information, results in highly degraded speech poorly rated by listeners when compared to reconstruction with phase @webber_autovocoder_2023.]

==== Reductive Transformations
These transformations distil specific attributes from the signal into a representation that is not intended for direct audio reconstruction. We classify representations as reductive if they cannot be reconstructed without leading to many-to-one scenarios where many waveforms clearly different from the original could lead to the same representation. 
In the example in @fig:fig_transformations, we show estimated energy derived by taking the average frame-level spectral magnitude sum and the fundamental frequency estimation algorithm Yin @decheveign_yin_2002 respectively.
However, reductive transformations can also be far removed from the original signal, such as the most probable text sequence $hat(T)$ as predicted by an #abbr.l[ASR] model.

==== Edge cases
Some transformations are not clearly reductive or lossy, such as most #abbr.a[SSL]-derived latent representations. They #emph[could] be used to reconstruct a speech signal, as previous work has shown that a combination of #abbr.a[SSL] representations and speaker embeddings can be reconstructed with similar performance as mel spectrograms @siuzdak_wavthruvec_2022. However, they could also lead to a deviation from the original depending on the information encoded by any particular #abbr.a[SSL] model. The layer at which the representation is extracted might also play a role, with earlier layers often corresponding more closely to acoustics @pasad_layer-wise_2021.

=== Purpose of Transformations

Different representations of speech can be used for differing purposes. Here, we discuss these representations and #emph[what] they encode (e.g. the whole signal, speaker identity, lexical content), #emph[how] they encode and potentially decode the signal and the #emph[why] the resulting representations might be used in speech synthesis.

==== The #emph[what] of encoding speech
Speech can be represented at "several levels of description" @mcclelland_trace_1986 -- each level aimed to encode a particular aspect of speech. These refer to differing hierarchical abstractions of the speech signal, ranging from low-level acoustic features (e.g., audio waveforms) to mid-level perceptual correlates (e.g., pitch and energy) and high-level content or contextual embeddings (e.g., speaker identity or prosodic patterns). In our work, we also focus on the #emph[factors] of speech which can have levels within them. For example, the #smallcaps[Prosody] factor of speech could include the relatively low-level aforementioned perceptual correlates in addition to more high-level features such as stress markers. Any representation does not have to be pure in the sense of only belonging to one factor: It could be a combination, or represent all (or most) factors present in speech, in which case we class them as #smallcaps[Generic]. Our terminology gives us a way to classify and describe factors such as referring to WavLM @chen_wavlm_2022 embeddings as #emph[high-level] and #smallcaps[Generic], or the energy contour as #emph[low-level] and #smallcaps[Prosody]. We can also put this framework into context with the previous classifications of transformations outlined in @02_class: Invertible transformations need to encode the whole signal and are therefore #emph[low-level] #smallcaps[Generic]. However, for lossy ones, high perceptual similarity rather than an exact copy of the original recording is desirable, which allows for a higher level of abstraction, often making them #emph[mid-level]. Finally, reductive transformations can be even more selective with what they encode, as reconstruction is not a concern, and can be #emph[high-level] abstractions as well as limit themselves to just encode one factor -- although they can still be #smallcaps[Generic] or a combination of several factors.

#figure(
  image("../figures/2/representation_overview.png", width: 100%),
  caption: [Overview of representations according to factors (y-axis) and level of abstraction (x-axis).],
  placement: top,
) <fig_factors>

==== The #emph[how] of encoding speech
The process of deriving a representation by applying a transformation function $cal(R)$ to a signal $S$ can be broadly categorised into three distinct methodologies, which often correspond to different levels of abstraction as illustrated in @fig:fig_factors. These are algorithmic, learned, and conceptual transformations.

*Algorithmic Transformations* also referred to as signal-based, rely on a predefined set of mathematical rules and signal processing techniques. The transformation function $cal(R)$ is a fixed algorithm whose parameters are not learned from data. Examples include the #abbr.a[STFT], the calculation of energy, or the estimation of fundamental frequency using algorithms like Yin @decheveign_yin_2002. These methods are typically transparent and computationally efficient. They form the basis for most low-level representations, corresponding to the green dashed boxes in @fig:fig_factors, providing direct, measurable correlates of the acoustic signal.

*Learned Transformations* are data-driven. In this paradigm, the transformation function $cal(R)$ is a model, typically a deep neural network, whose parameters are optimised to perform a specific task on a large dataset. This can involve training a model to reconstruct its input (an autoencoder), to predict a masked portion of the signal (predictive coding), or to classify the input. The resulting representations, such as #abbr.a[SSL] embeddings @chen_wavlm_2022, speaker embeddings @snyder_x_2018, or neural codec tokens @defossez_encodec_2023, are often highly effective for downstream tasks because they capture complex, non-linear patterns in the data. These representations, shown as the blue solid boxes in @fig:fig_factors, can span from low- to high-levels of abstraction.

*Conceptual Transformations* yield high-level, usually discrete labels that represent an abstract concept within the speech signal. For these representations, we extend our definition of the transformation $cal(R)$ to encompass not just algorithms, but also complex inference models or even the human annotation process itself. In the case of human annotation, the transformation $cal(R)$ can be seen as the cognitive process of the human listener, guided by a set of annotation criteria (e.g., a rubric for emotional expression), which maps the continuous perceptual experience of the speech signal $S$ onto a discrete symbolic label. The resulting representations, like "Intent" or "Acoustic Scene," are reductive by nature and correspond to the red-hatched boxes in @fig:fig_factors. They are invaluable for supervised training and evaluation, providing a content grounding for more abstract models.

==== The #emph[why] of encoding speech
The purpose of any given speech representation can be understood by its position on a spectrum ranging from purely #emph[acoustic] to purely #emph[content]. Representations at the extremes of this spectrum serve distinct functions, while many of the most useful representations exist in the middle, acting as a bridge between the two domains.

At the purely *acoustic* end of the spectrum lie representations whose primary goal is to capture the physical properties of the sound wave with the highest possible fidelity. Audio waveforms contain the most acoustic information, with lossless codecs like #abbr.pla[FLAC] containing the same level of acoustic information as well, as their representations are designed for perfect, bit-for-bit reconstruction of the original signal. The primary use for these representations is high-fidelity storage, transmission, and ultimately this level of representation has to be reached as the final step by any synthesis system aimed at producing audio.

At the opposite, purely *content* end are representations that encode abstract meaning, linguistic structures, and concepts, completely detached from the acoustic signal that conveyed them. The final text transcript from an ASR system or a conceptual "Intent" label are examples. The purpose of these representations can be analysis, understanding, and symbolic reasoning. They are the inputs and outputs of systems concerned with what is being said, by whom, and for what purpose.

The vast majority of speech processing involves bridging the gap between these two concepts, and thus many representations encode aspects of both. These *intermediate* representations are designed to make acoustic information tractable for content tasks, or to guide the generation of acoustic signals from content information. For example, a mel spectrogram is fundamentally acoustic, yet it is structured in a perceptually relevant way that makes it easier for a model to recognise phonetic content, making linguistic content easier to access and model. #smallcaps[Prosody] features like a pitch contour are acoustic measurements, but their shape and trajectory can convey content meaning such as if the speaker is expressing a certain emotion or asking a question. Conversely, a #smallcaps[Speaker] embedding is a high-level abstraction of identity (a content concept), but it is learned by distilling patterns directly from acoustic signals. Another #emph[intermediate] representation are high-level #abbr.a[SSL] embeddings, which are learned from audio waveforms with the explicit goal of being maximally useful for a wide range of downstream content tasks.

=== Factor-based Review of Speech Representations <02_representations>

We will now examine the factors of speech as laid out in @fig:fig_factors in more detail, contextualising their roles across various speech technology applications. We follow a bottom-up order from the figure, reordered to reflect a common conceptual flow from abstract contents to the concrete acoustic signal.

==== Content representations

The #smallcaps[Content] factor encompasses the linguistic content of speech. Its representations lie firmly at the content pole of our spectrum and are central to any task involving understanding or generating language.
One of the most foundational and frequently used representations is the phone sequence. This representation converts the orthographic sequence of letters into a sequence of phonemes, the basic sound units of a language. This transformation, known as #abbr.a[G2P] conversion, resolves ambiguities in pronunciation (e.g., the different sounds of "ough" in "through," "tough," and "though") and provides a canonical representation of pronunciation. In TTS, it serves as a clean input to the acoustic model. In ASR, especially for low-resource languages, predicting a phoneme sequence can be a useful intermediate step before word-level transcription. Multilingual models like Allosaurus provide phone recognition for a vast number of languages, facilitating such cross-lingual applications @li_allosaurus_2020, while there are also approaches that require limited amounts of audio data @li_asr2k_2022.

On the more abstract side, modern systems for both ASR and TTS process text using powerful encoders, often using contextualised language models like BERT @devlin_bert_2019. These produce contextualised text embeddings, which are high-dimensional vectors that capture not just the identity of words but also their meaning within the specific context of the sentence. In TTS, this helps generate more natural prosody @kakouros_what_2023. In Spoken Language Understanding (SLU), these embeddings are the basis for interpreting the user's commands. At the highest level of abstraction, intent represents the speaker's pragmatic goal. In conversational AI, an ASR system first produces text, which is then classified into an intent label. This label then drives the dialogue manager and determines the content and style of the TTS response, completing the interaction loop.

==== Speaker representations <02_speaker>

The #smallcaps[Speaker] factor encompasses the unique, identifying characteristics of an individual's voice. Its representations bridge the gap between acoustic reality and the content concept of identity.

At a low level, identity is encoded in physical properties. Formant frequencies, the resonant peaks in the speech spectrum, are determined by the size and shape of the speaker's vocal tract and are thus a strong anatomical correlate @stevens_sources_1972. While formants determine vowel identity (part of the #smallcaps[Content] factor), their absolute frequency ranges are constrained by the vocal tract length, making them a key physical correlate of the #smallcaps[Speaker] factor. A speaker's habitual average pitch is another simple indicator. However, these features are highly variable and deeply entangled with other factors, making them unreliable for robust speaker identification.

To create a more robust representation, systems distil these variable acoustic cues into high-level, learned speaker embeddings. These fixed-dimensional vectors are extracted using a network trained to discriminate between thousands of speakers. Architectures like x-vectors @snyder_x_2018 and ECAPA-TDNN @desplanques_ecapa_2020 excel at this, learning to create a representation that is maximally sensitive to inter-speaker differences while being invariant to intra-speaker variability (like phonetic content). These embeddings are the cornerstone of speaker verification and diarisation @bredin_pyannote_2023, as well as generative tasks like multi-speaker TTS and voice cloning @li_styletts_2023. Beyond core timbre, other identifying characteristics contribute to our perception of a speaker, such as perceived age, gender, and accent. While these can be treated as explicit content labels, they are often learned implicitly as part of a powerful speaker embedding, which captures not just anatomy but also these habitual, identity-linked speaking patterns.

==== Prosody representations <02_prosody_rep>

The #smallcaps[Prosody] factor is an #emph[intermediate] factor, governing the melody and rhythm of speech that links acoustic signals to emotional and pragmatic meaning. It is important to distinguish between #emph[prosody] (the perceptual experience of rhythm and melody) and its #emph[acoustic correlates] (measurable signals like fundamental frequency, intensity, and duration). In TTS-for-ASR, we often manipulate the latter to influence the former.

Low-level prosodic features provide a quantitative link between acoustics and perception. Duration specifies the length of phones and pauses, defining speaking rate and rhythm. Analytically, it helps distinguish speaking styles; generatively, it allows for explicit tempo control in TTS, with ground-truth data often obtained from forced alignment @dugad_hmmtutorial_1990, for which the most commonly used toolkit today is the Montreal forced aligner @mcauliffe_montreal_2017. Pitch (F0) and energy (loudness) contours are likewise essential. In emotion recognition, rising pitch and high energy are strong acoustic cues for excitement. In controllable synthesis, these contours can be explicitly predicted to shape the melody and stress of the output speech @lancucki_fastpitch_2021.

For a more holistic view, mid-to-high-level representations are used. *Phonologic Labels*, derived from linguistic theories like ToBI @ross_prediction_1996, provide a symbolic, linguistic description of acoustic events like pitch accents and boundary tones, useful for prosodic analysis. A more data-driven approach is to learn a continuous, high-level prosody embedding. This vector can serve as an input feature for an emotion classifier or as a conditioning signal for style transfer in voice conversion and expressive TTS @wang_style_2018. At the highest level, a discrete emotion label sits at the content end, representing the final interpretation of a complex set of underlying acoustic cues.

==== Ambient representations

The #smallcaps[Ambient] factor encompasses all environmental aspects of a recording. The utility of its representations depends entirely on whether the task goal is analysis of the real world or synthesis of an ideal one.

For analytical tasks like ASR, modeling ambient acoustics is critical. Systems must be robust to background noise and reverberation. Low-level metrics like #abbr.a[SNR] @kim_wada_2008 or reverberation time @kinoshita_reverb_2013 can be used as features for a speech enhancement front-end. High-level acoustic scene classification can help a system adapt its acoustic models to specific conditions like being "in a car," attaching the acoustic environment to a content label.

Conversely, for generative tasks like TTS, the goal is typically to produce clean speech. Here, the ambient factor is something to be eliminated. Ambient representations are therefore primarily used for quality control during data preparation, ensuring that the training data is as free from acoustic contamination as possible.

==== Generic representations

The #smallcaps[Generic] factor includes representations that encode the full acoustic signal. They span the entire spectrum, from the purely acoustic waveform to abstract embeddings that bridge to the content domain.

Historically, *Spectrogram-Derived Representations* have been the classic bridge between acoustics and content. These include the mel spectrogram and #abbr.a[PLP] analysis, along with its popular variant, #abbr.pla[MFCC]. These mid-level features transform the complex waveform into a more tractable, perceptually-motivated format that highlights phonetic differences, making them the standard input for decades of ASR and speaker ID systems, and a common output target for TTS acoustic models.

Closer to the acoustic pole are representations from *Neural Audio Codecs*. Models like EnCodec @defossez_encodec_2023 learn to convert a waveform into a sequence of discrete tokens for the primary purpose of high-fidelity, reconstructible compression. While they are learned, their goal is faithful acoustic representation, making them ideal as a high-quality output target for generative models like TTS @wang_valle_2023.

Finally, high-level *Contextualised Speech Embeddings* from #abbr.a[SSL] models like WavLM @chen_wavlm_2022 are #emph[intermediate] representations. Learned from acoustics, they are explicitly optimised to be useful for a wide range of downstream content tasks. They are reductive yet encode rich, entangled information about all factors. This makes them a state-of-the-art auxiliary feature for nearly any speech task, capable of improving robustness and performance by providing a powerful, pre-trained understanding of the relationship between sound and meaning.

==== Limitations and cross-factor phenomena

While the factorised view provides a useful framework, it is important to acknowledge its limitations. Certain acoustic phenomena do not fit neatly into a single category, but rather exist at the intersection of multiple factors, highlighting the complex relationship between acoustics and content.

One prominent example is voice quality or phonation mode. A speaker can alter their phonation to be breathy or to whisper. Breathiness can be a stable characteristic of the #smallcaps[Speaker], but it can also be used dynamically to convey an intimate emotion, making it a feature of #smallcaps[Prosody]. Whispering is an extreme case that removes the fundamental frequency entirely, yet speakers can still convey prosodic contrasts through other acoustic cues. Thus, voice quality is a low-level acoustic characteristic that provides evidence for higher-level interpretations in both the #smallcaps[Speaker] and #smallcaps[Prosody] factors.

Other phenomena are inherently cross-factor. Sarcasm, for instance, arises from a deliberate mismatch between the literal #smallcaps[Content] (e.g., "What a wonderful day") and a contradictory #smallcaps[Prosody] (e.g., a flat intonation). The meaning is not in the words or the melody alone, but in the cognitive dissonance created by their conflict. Similarly, a speaker's register or speaking style (e.g., formal presentation vs. casual conversation) is a meta-level choice that simultaneously modulates all other factors, from word choice (#smallcaps[Content]) to intonation patterns (#smallcaps[Prosody]).