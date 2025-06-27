#import "../abbr.typ" 

== Increasing Synthetic Diversity <04_attr>

As established in the preceding chapter, a notable discrepancy exists between synthetic and real speech. While modern #abbr.s[TTS] systems can achieve high subjective naturalness, the resulting speech often exhibits a more narrow distribution, lacking the rich diversity inherent in human expression @hu_syntpp_2022. This limitation is a primary cause for the performance gap observed when using synthetic data for #abbr.s[ASR] model training, where the Word Error Rate Ratio (#abbr.s[WERR]) consistently remains well above the ideal value of 1.

To bridge this gap, it is not enough for synthetic speech to be merely natural-sounding; it must also be sufficiently diverse and robust. This chapter surveys the main paradigms developed to increase the variability of synthetic speech, moving beyond the generation of a single, deterministic output for a given input. We will explore three principal approaches: learning latent representations of style, explicitly conditioning the model on measurable attributes, and applying post-generation data augmentation.

=== Learning Latent Representations of Style

A primary approach for introducing diversity is to have the model learn a latent space that captures stylistic variations from the training data in an unsupervised manner. During inference, this space can be sampled to generate varied outputs. The two most common methods for this are Global Style Tokens and #abbr.a[VAE].

==== Global Style Tokens

Global Style Tokens (#abbr.s[GST]) learn a set of discrete, interpretable "style" embeddings from the reference audio @wang_style_2018. The model, typically through an attention mechanism, learns to represent the style of an utterance as a combination of these tokens. At inference time, the style tokens can be selected or sampled, allowing for control over the speaking style of the generated speech. While powerful, this approach models variation as a discrete set of styles.

==== #abbr.l[VAE]

A more flexible approach is to use a #abbr.a[VAE] to learn a continuous latent distribution of style @kingma_auto-encoding_2013. A #abbr.s[VAE] is trained to encode reference speech into a latent vector, which is then used by the #abbr.s[TTS] decoder to reconstruct the original speech. By enforcing that the latent space follows a simple prior distribution (typically a standard Gaussian), we can sample from this prior at inference time to generate speech with novel stylistic variations. This method has proven to be an effective solution for generating diverse speech for the #abbr.s[TTS]-for-#abbr.s[ASR] task @sun_generating_2020.

=== Explicit Conditioning on Attributes

In contrast to learning an abstract latent space, an alternative paradigm offers more direct and interpretable control by explicitly conditioning the synthesis process on specific, measurable attributes of the speech signal.

==== The Variance Adapter

In modern non-autoregressive architectures, this control is often implemented via a *variance adapter* @ren_fastspeech_2019@ren_fastspeech_2021. This module is typically inserted between the text encoder and the spectrogram decoder. During training, it is given ground-truth values for various attributes, which it learns to embed and add to the hidden text representations. This enriches the information available to the decoder, which learns to generate a spectrogram conditioned on both the phoneme sequence and the desired attributes.

At inference time, the model must be provided with target attribute values. If none are given, the adapter typically predicts the mean values seen during training, leading to a collapse in diversity. True control is therefore achieved by supplying the desired attribute values as an input to the model.

==== Controllable Attributes <04_prosodic_correlates>

The attributes used for conditioning are typically correlates of perceptually relevant phenomena, as detailed in @06_perceptual[Chapter 6]. Key examples include:

-   *Prosody*: Features such as pitch (F0), energy (loudness), and phoneme duration (speaking rate) are fundamental to speaking style. Architectures like FastSpeech 2 and FastPitch explicitly model and predict these attributes at the frame or phoneme level @ren_fastspeech_2019@ren_fastspeech_2021@lancucki_fastpitch_2021.
-   *Acoustic Environment*: For #abbr.s[ASR] robustness, it is beneficial to model environmental factors. These can be quantified using metrics like the #abbr.a[SRMR] @kinoshita_reverb_2013 and the #abbr.a[SNR], which can be estimated non-intrusively using methods like Waveform Amplitude Distribution Analysis (#abbr.s[WADA]) @kim_wada_2008.

By conditioning on such attributes, a #abbr.s[TTS] system can be guided to generate speech with specific prosodic contours or as if it were recorded in a particular acoustic environment.

=== Post-Generation Data Augmentation

The third paradigm for increasing diversity operates not within the #abbr.s[TTS] model itself, but on its output. Post-generation data augmentation involves taking clean, synthesized speech and applying transformations to simulate real-world variability. This is a common and effective technique for making training data more robust for #abbr.s[ASR].

This typically involves adding simulated background noise from various sources or convolving the clean waveforms with #abbr.pla[RIR] to simulate different acoustic spaces and reverberation characteristics @rossenbach_synthattention_2020. While this approach is highly effective for modeling the acoustic environment, it cannot introduce variation in the underlying prosody or speaker characteristics of the source synthesis. These three approaches are therefore complementary, and are often used in combination to create synthetic data that is both natural and diverse.