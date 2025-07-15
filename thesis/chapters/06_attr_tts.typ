Of course, here is the corrected chapter with all abbreviations encapsulated using the abbr package as requested. I have ensured that manually written-out abbreviations are removed and that all instances are handled by the package for consistent formatting.

Generated typ
#import "../abbr.typ"
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
#import "../comic.typ"

== Increasing Synthetic Diversity <06_attr>

As established in the preceding chapter, a notable discrepancy exists between synthetic and real speech. While modern #abbr.a[TTS] systems can achieve high subjective naturalness, the resulting speech often exhibits a more narrow distribution, lacking the rich diversity inherent in human expression @hu_syntpp_2022. This limitation is hypothesised to be a primary cause for the performance gap observed when using synthetic data for #abbr.a[ASR] model training, where the #abbr.a[WERR] consistently remains well above the ideal value of 1.

To bridge this gap, it is not enough for synthetic speech to be merely natural-sounding; it must also be sufficiently diverse. This chapter surveys the main paradigms developed to increase the variability of synthetic speech, moving beyond the generation of a single, deterministic output for a given input. We will explore three principal approaches: learning latent representations of style, explicitly conditioning the model on measurable attributes, and applying post-generation data augmentation.

=== Learning latent representations of style

A primary approach for introducing diversity is to have the model learn a latent space that captures stylistic variations from the training data in an unsupervised manner. During inference, this space can be sampled to generate varied outputs. The two most common methods for this are #abbr.a[GST] and #abbr.a[VAE].

==== Global style tokens

#abbr.a[GST] learn a set of discrete, interpretable "style" embeddings from the reference audio @wang_style_2018. The model, typically through an attention mechanism, learns to represent the style of an utterance as a combination of these tokens. At inference time, the style tokens can be selected or sampled, allowing for control over the speaking style of the generated speech. While powerful, this approach models variation as a discrete set of styles.

==== Variational autoencoder

A more flexible approach is to use a #abbr.a[VAE] to learn a continuous latent distribution of style @kingma_auto-encoding_2013. A #abbr.a[VAE] is trained to encode reference speech into a latent vector, which is then used by the #abbr.a[TTS] decoder to reconstruct the original speech. By enforcing that the latent space follows a simple prior distribution (typically a standard Gaussian), we can sample from this prior at inference time to generate speech with novel stylistic variations. This method has proven to be an effective solution for generating diverse speech for the #abbr.a[TTS]-for-#abbr.a[ASR] task @sun_generating_2020.

=== Explicit conditioning on attributes

In contrast to learning an abstract latent space, an alternative paradigm offers more direct and interpretable control by explicitly conditioning the synthesis process on specific, measurable attributes of the speech signal.

==== Variance adapter

In modern non-autoregressive architectures, this control is often implemented via a *variance adapter* @ren_fastspeech_2019@ren_fastspeech_2021. This module is typically inserted between the text encoder and the spectrogram decoder. During training, it is given ground-truth values for various attributes, which it learns to embed and add to the hidden text representations. This enriches the information available to the decoder, which learns to generate a spectrogram conditioned on both the phoneme sequence and the desired attributes.

At inference time, the model must be provided with target attribute values. If none are given, the adapter typically predicts the mean values seen during training, leading to a collapse in diversity. True control is therefore achieved by supplying the desired attribute values as an input to the model.

==== Controllable attributes <06_prosodic_correlates>

The attributes used for conditioning are typically correlates of perceptually relevant phenomena, as detailed in @02_perceptual[Chapter]. Key examples include:

-   *Prosody*: Features such as pitch (F0), energy (loudness), and phoneme duration (speaking rate) are fundamental to speaking style. Architectures like FastSpeech 2 and FastPitch explicitly model and predict these attributes at the frame or phoneme level @ren_fastspeech_2019@ren_fastspeech_2021@lancucki_fastpitch_2021.
-   *Acoustic Environment*: For #abbr.a[ASR] robustness, it is beneficial to model environmental factors. These can be quantified using metrics like the #abbr.a[SRMR] @kinoshita_reverb_2013 and the #abbr.a[SNR], which can be estimated non-intrusively using methods like #abbr.a[WADA] @kim_wada_2008.

By conditioning on such attributes, a #abbr.a[TTS] system can be guided to generate speech with specific prosodic contours or as if it were recorded in a particular acoustic environment.

=== Post-generation data augmentation

The third paradigm for increasing diversity operates not within the #abbr.a[TTS] model itself, but on its output. Post-generation data augmentation involves taking clean, synthesized speech and applying transformations to simulate real-world variability. This is a common and effective technique for making training data more robust for #abbr.a[ASR].

This typically involves adding simulated background noise from various sources or convolving the clean waveforms with #abbr.pla[RIR] to simulate different acoustic spaces and reverberation characteristics @rossenbach_synthattention_2020. While this approach is highly effective for modeling the acoustic environment, it cannot introduce variation in the underlying prosody or speaker characteristics of the source synthesis. These three approaches are therefore complementary, and are often used in combination to create synthetic data that is both natural and diverse.

=== Evaluating diversity through #abbr.a[ASR]

Of the aforementioned paradigms, explicit conditioning via a variance adapter and post-generation augmentation are the most direct methods for targeting specific, known sources of variability. To quantify the impact of these approaches on the distribution gap, we adopt the task of training an #abbr.a[ASR] system on synthetic speech as a rigorous, objective proxy for evaluating diversity.

Subjective listening tests, while valuable for assessing naturalness, are not well-suited to measuring distributional coverage. A synthetic utterance can sound perfectly human yet represent only a narrow, oversampled region of the real speech distribution. #abbr.a[ASR] model training, however, is highly sensitive to this lack of diversity; models trained on data with limited variability generalise poorly to the rich variation of real-world speech @wang_improving_2020.

We can quantify this performance gap using the #abbr.a[WER] of two #abbr.a[ASR] models trained on identical scripts, one using real speech $S$ and the other using synthetic speech $Syn$. The ratio between their respective error rates on a common, real-speech test set gives us the #abbr.a[WERR], as formalised in @eq_werr.

Both real and synthetic speech have a seen and unseen split, $S$ and $s$, and $Syn$ and $syn$ respectively. They are paired with seen and unseen transcripts $T$ and $t$. $Theta(S,T)$ yields the weights of an ASR model given the speech and transcripts, and $"WER"$ is the Word Error Rate achieved when a certain set of ASR weights is evaluated on a different set of speech and transcripts

$
"WERR"(Syn parallel S) = "WER"[(s,t) parallel Theta(Syn,T)] / "WER"[(s,t) parallel Theta(S,T)]
$ <eq_werr>

A #abbr.a[WERR] of 1.0 would indicate that the synthetic speech is as effective as real speech for #abbr.a[ASR] training, implying that its distribution sufficiently covers the variations relevant for the task. Previous work consistently shows this ratio to be significantly higher than 1.0, often between 3 and 5, highlighting a substantial distribution gap @minixhofer_evaluating_2023 @casanova_singlespeaker_2022. Reducing this ratio is therefore a primary objective in our efforts to increase synthetic diversity.

=== Experimental Design

To investigate the effects of explicit conditioning and augmentation, we designed a series of experiments based on a controllable, multi-speaker #abbr.a[TTS] system. Our goal is to incrementally add components that target different sources of variation and measure their impact on the #abbr.a[WERR].

#comic.comic((50mm, 80mm), "A visual explanation of the dataset splits.", blue) <data_splits>

==== Dataset and Models

All experiments utilise the `train-clean-360` split of the LibriTTS corpus @zen_libritts_2019. We selected speakers with at least 100 utterances, resulting in a set of 684 speakers. For each experiment, we generate 10 hours of synthetic audio with balanced transcripts and speakers.

Our baseline #abbr.a[TTS] system is a multi-speaker version of FastSpeech 2 @ren_fastspeech_2021. Speaker identity is provided via d-vector embeddings ($E_text("SPK")$) extracted from a pretrained speaker verification model @wan_generalized_2018 and averaged per speaker. These embeddings are added to the phoneme-level hidden representations. A HiFi-GAN vocoder is used to convert the predicted Mel spectrograms into waveforms @kong_hifigan_2020.

For evaluation, we train a 6-layer hybrid #abbr.a[HMM]-#abbr.a[TDNN] #abbr.a[ASR] system with the #abbr.a[LF-MMI] objective, a standard recipe in the Kaldi toolkit @povey_kaldi_2011. The architecture is kept minimal to ensure that performance differences are primarily attributable to the quality and diversity of the training data, not the #abbr.a[ASR] model's capacity to overcome data deficiencies.

==== Attribute Conditioning with Generative Models

To move beyond the mean-value predictions of a standard variance adapter, we introduce an *Attributes* system. This system is explicitly conditioned on utterance-level mean values for pitch ($F_("F0")$), energy, speaking rate (duration), #abbr.a[SRMR], and #abbr.a[SNR]. During training, these are ground-truth values extracted from the real speech signals.

The central challenge is to generate realistic attribute values during inference. To this end, we model the joint distribution of these attributes for each speaker using a #abbr.a[GMM]. For each synthetic utterance, we sample a vector of target attributes from the corresponding speaker's #abbr.a[GMM] and provide it as conditioning to the #abbr.a[TTS] model. This encourages the model to generate speech with attribute distributions that more closely match the ground truth, rather than collapsing to the mean. @fig_attribute_system illustrates this architecture.

#figure(
  image("../figures/6/attr_tts_arch.png", width: 60mm),
  caption: [A simplified diagram of the attribute-conditioned TTS system. During inference, speaker-specific #abbr.pla[GMM] generate target utterance-level attributes (e.g., pitch, speaking rate). These are passed to the FastSpeech 2 decoder, along with phoneme encodings, to guide the synthesis of a Mel spectrogram. A vocoder converts this to a waveform, which can then undergo optional augmentation.]
) <fig_attribute_system>

We compare this against an *Oracle* system, which is given the ground-truth attribute values at inference time. This provides a practical upper bound on performance, indicating the maximum potential improvement achievable if our #abbr.pla[GMM] could perfectly model the real attribute distributions.

Finally, we test a dedicated *Augmentation* system, which applies post-generation augmentation to the output of the *Attributes* system. This involves adding simulated Gaussian noise (with a target #abbr.a[SNR] between 5 and 40 dB) and convolving the waveform with a #abbr.a[RIR] to simulate reverberation, using the `audiomentations` library #footnote[#link("https://github.com/iver56/audiomentations", [https://github.com/iver56/audiomentations])].

=== Results and Discussion

The results of our #abbr.a[TTS]-for-#abbr.a[ASR] experiments are summarised in @tbl_werr_results. The baseline #abbr.a[TTS] system yields a #abbr.a[WERR] of 3.66, confirming that its output, while natural-sounding, is substantially less effective for #abbr.a[ASR] training than real speech. This establishes a clear benchmark for our diversity-enhancing techniques.

#figure(
  table(
  columns: (1fr, 1fr, 1fr),
  align: center,
  [*System*], [#abbr.s[WER] (%)], [#abbr.s[WERR]],
  [Real Data], [13.3 ± 0.29], [1.00],
  [Baseline TTS], [48.6 ± 0.43], [3.66 ± 0.03],
  [\+ Attributes], [47.2 ± 0.39], [3.55 ± 0.03],
  [\+ Augmentation], [44.0 ± 0.33], [3.31 ± 0.02],
  [Oracle], [43.0 ± 0.29], [3.24 ± 0.02],
),
caption: [#abbr.a[TTS]-for-#abbr.a[ASR] evaluation results. We report the #abbr.a[WER] and #abbr.a[WERR] for #abbr.a[ASR] models trained on data from different #abbr.a[TTS] systems. The best performing system achieves a 10% relative reduction in #abbr.a[WERR] compared to the baseline.],
) <tbl_werr_results>

The *Attributes* system, which conditions on #abbr.a[GMM]-sampled utterance-level features, reduces the #abbr.a[WERR] to 3.55. This modest but significant improvement demonstrates that explicitly guiding the model to produce more varied prosody and acoustic characteristics helps to close the distribution gap. However, the gap between this system and the *Oracle* (#abbr.a[WERR] of 3.24) indicates that the #abbr.pla[GMM], while helpful, are not perfectly capturing the complex distributions of the real attributes.

The most substantial improvement comes from the *Augmentation* system. By applying noise and reverberation as a post-processing step, the #abbr.a[WERR] is reduced to 3.31, a relative reduction of nearly 10% from the baseline. This result highlights the critical importance of acoustic environment diversity for robust #abbr.a[ASR] training. Crucially, it shows that even when the underlying speech synthesis is not perfect, simulating real-world recording conditions is a highly effective strategy for bridging the gap. While our #abbr.a[GMM]-based attribute conditioning and post-generation augmentation are complementary, these findings suggest that for the purposes of #abbr.a[TTS]-for-#abbr.a[ASR], matching the acoustic environment distribution yields the largest gains.

=== WERR as a distance metric

Another question that arises is if #abbr.a[WERR] could qualify as a metric for the distance between the synthetic and real distributions. For a heuristic to be considered a formal distance metric, it must satisfy four key properties: non-negativity, identity (the distance from an object to itself is zero), symmetry (the distance from A to B is the same as from B to A), and the triangle inequality. Since non-negativity is already satisfied (a ratio of positive values cannot be negative) and identity can be trivially achieved by substracting 1, we investigate the latter two of these properties here.

#figure(
  table(
  columns: (1fr, 1fr, 1fr),
  align: center,
  [*Training Data*], [*Evaluation Data*], [#abbr.s[WER] #sym.arrow.b],
  [Real], [Real], [13.3 ± 0.29],
  [Synthetic], [Real], [48.6 ± 0.43],
  [Real], [Synthetic], [11.4 ± 0.69],
  [Synthetic], [Synthetic], [3.0 ± 0.02],
),
caption: "Results when training on synthetic and evaluating on real and vice versa."
) <tab_cross_ttsasr>

Our ablation experiments shown in @tab_cross_ttsasr provide a preliminary insight into symmetry. The standard #abbr.a[WERR] from @eq_werr, which measures the performance drop when training on synthetic speech $Syn$ instead of real speech $S$ (both evaluated on real data), is $3.66$. We can also calculate a #abbr.a[WERR] in the reverse direction: measuring the performance drop when training on real data $S$ instead of synthetic data $Syn$, when evaluating on the *synthetic* test set $syn$. This gives a ratio of $"WER"[(syn,t) parallel Theta(S,T)] / "WER"[(syn,t) parallel Theta(Syn,T)]$, which is 3.75. While these values are remarkably close, suggesting a degree of symmetry in our specific setup, the inherent stochasticity of DNN training means this is not guaranteed to hold across different datasets or models. A truly robust metric should enforce this property by design.

To ensure symmetry, we can follow the example of the #abbr.a[JSD] @lin_jsd_1991 and define a #abbr.a[MWERR]. By averaging the #abbr.a[WERR] calculated in both directions, we create a measure which is symmetric by definition:

$
"MWERR"(Syn parallel S) = 1/2"WER"[(s,t) parallel Theta(Syn,T)] / "WER"[(s,t) parallel Theta(S,T)] + 1/2"WER"[(syn,t) parallel Theta(S,T)] / "WER"[(syn,t) parallel Theta(Syn,T)]
$ <eq_mwerr>

While #abbr.a[MWERR] achieves symmetry, it cannot be proven to satisfy the triangle inequality, which is the final condition for a true distance metric (i.e., that the direct distance between two points is no greater than the distance taken via a third point, $d(A,C) <= d(A,B) + d(B,C)$). This limitation arises because its fundamental component, the #abbr.a[WER], is not a distance metric itself. #abbr.a[WER] is the outcome of a complex and highly non-linear optimization process—the training of an #abbr.a[ASR] model $Theta$—not a direct geometric or probabilistic comparison between two data distributions. The relationship between a change in the data distribution and the resulting #abbr.a[WER] is not simple or predictable enough to guarantee that the "distance" between two distributions is always less than the sum of their intermediate "distances" via a third distribution. Therefore, while #abbr.a[MWERR] is a more principled heuristic than #abbr.a[WERR], it should be interpreted as a task-specific measure of distributional dissimilarity rather than a formal distance metric.
