#import "../abbr.typ"
#import "../quote.typ": *
#import "../math.typ": *
#import "../moremath.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

== Enhancing Synthetic Speech Diversity <06_diversity>

#q(
[#citep(<oord_wavenet_2016>)],
[#emph[WaveNet: A Generative Model for Raw Audio]],
[We condition the model $dots$ in two different ways: global conditioning and local conditioning.]
)

Building on the significant performance gap between ASR models trained on real versus synthetic speech, as quantified by the Word Error Rate Ratio (WERR) in @05_werr, this chapter introduces and evaluates methodologies to increase the diversity of synthetic speech. As established in the previous chapter, synthetic speech, despite its high human naturalness ratings, often lacks the intricate variability inherent in real human speech. This limitation directly impedes its utility for training robust ASR systems. Here, we systematically explore three complementary paradigms aimed at bridging this distributional gap: learning latent representations, explicit conditioning on attributes, and post-generation data augmentation. By introducing and controlling these aspects of speech, we aim to reduce the distributional distance between synthetic and real speech, which should, in turn, manifest as a reduction in WERR. Throughout this chapter, we adhere to the formal notation established in the introduction, where $Q_theta$ represents the Text-to-Speech (TTS) model's approximation of the true speech distribution $Q(S|T)$.

The primary contribution of this chapter is a systematic methodology for enhancing synthetic speech diversity through explicit conditioning on acoustic attributes, combined with a novel probabilistic sampling approach for generating this conditioning information. We conduct a controlled experimental evaluation of this methodology alongside post-generation augmentation, quantifying their impact on the synthetic-real gap using both the task-oriented WERR and direct distributional distance measures. These contributions were covered in the following work:

- #cite(<minixhofer_evaluating_2023>, form: "full")

=== Paradigms for Enhancing Speech Diversity

To address the lack of variability in synthetic speech, various techniques have been developed. These can be broadly categorised into three approaches: learning abstract representations of style from data in an unsupervised manner, explicitly conditioning the synthesis process on measurable acoustic attributes, and applying augmentations to the generated audio as a post-processing step.

==== Learning Latent Representations

Latent representations offer a powerful, unsupervised approach to capturing and injecting stylistic and acoustic variability into synthetic speech. These methods learn an encoding of information, such as style or prosody, from the training data without requiring explicit labels or pre-definition of these attributes. As discussed in @02_representations, such representations are often classified as "learned transformations". They can be reduced in the time domain, such as a single high-dimensional vector representing the characteristics of an entire utterance, or vary over time. This paradigm is particularly valuable for addressing the inherent "one-to-many" problem in TTS, where a single text input can correspond to countless valid acoustic realisations.

These latent representations can encode different factors of speech. A #smallcaps[Generic] factor might be captured by a self-supervised learning (SSL) model, which learns a representation containing entangled information about all aspects of the speech signal. A #smallcaps[Speaker] factor is commonly encoded using speaker embeddings like d-vectors @wan_generalized_2018, which are high-dimensional but fixed-length vectors that capture the unique, time-invariant characteristics of an individual. Finally, a #smallcaps[Prosody] factor can be captured by models designed to learn stylistic variations that change over time.

Mathematically, these methods model a latent variable $Z$ which is first inferred from a reference utterance $S$. The TTS model is then trained to reconstruct the utterance from its text $T$ and the inferred latent variable $Z$, learning the conditional distribution $Q_theta (tilde(S) | T, Z)$. For synthesis, a latent variable $Z$ is provided alongside the input text $T$ to generate a novel utterance $tilde(S)$.

===== Global Style Tokens

Building on the general concept of latent representations, Global Style Tokens (GSTs) provide a discrete approach to modelling stylistic variations in speech @wang_style_2018. The core idea involves learning a fixed set of $K$ distinct, global style embeddings or "tokens," denoted as $cal(G) = {g_1, ..., g_K}$. During the training phase, the TTS model is provided with a reference utterance $S$. An attention mechanism within the model then learns to weigh these tokens based on the acoustic characteristics of the reference audio. This process encodes the utterance's observed style as a weighted combination of the learned tokens. Formally, the style vector $Z$ for a given reference utterance $S$ is computed as:
$ Z = sum_(k=1)^K alpha_k g_k $
where $alpha_k$ are the attention weights learned from the reference $S$. These weights sum to one, making $Z$ a combination of the style tokens. At inference time, instead of deriving $Z$ from a reference utterance, the style tokens can be directly selected, combined, or interpolated to control the output style, enabling the generation of speech that exhibits desired characteristics, such as being more expressive or neutral. While effective for inducing categorical style shifts and improving perceived naturalness, the discrete nature of GSTs can limit the ability to capture and generate very fine-grained, continuous variations compared to other latent methods.

===== Variational Autoencoders

#abbr.pla[VAE] extend the concept of latent style modelling by learning a continuous latent distribution, allowing for smoother and more nuanced sampling of stylistic variations @kingma_auto-encoding_2013. A VAE consists of an encoder and a decoder. During training, the encoder maps a given reference speech utterance $S$ into the parameters of a probabilistic latent distribution, typically assumed to be a Gaussian posterior distribution. Specifically, the encoder learns to output the mean $mu_phi (S)$ and variance $sigma_phi (S)$ (or log-variance) of this Gaussian:
$ q_phi (Z | S) = cal(N)(mu_phi (S), sigma_phi (S)) $
This approximate posterior $q_phi (Z|S)$ is trained to approximate the true posterior distribution $p(Z|S)$ by maximising the Evidence Lower Bound (ELBO). The ELBO objective balances two terms: the reconstruction likelihood and a Kullback-Leibler (KL) divergence term that regularises the latent space $Z$ by pushing $q_phi (Z|S)$ to be close to a simple prior distribution $p (Z)$ (i.e. a standard Gaussian $cal(N)(0, I)$):
$ cal(L)_"ELBO" = EE_(q_phi (Z|S)) [log p_theta (S | Z)] - "KL"(q_phi (Z|S) || p (Z)) $
The TTS decoder then takes a sampled latent vector $Z$ and the input text $T$ to reconstruct the synthetic utterance $tilde(S)$, aiming to approximate $p_theta (tilde(S) | T, Z)$. The regularisation of the latent space ensures that samples drawn from the simple prior $p(Z)$ at inference time can generate novel and plausible speech styles, even if those specific styles were not explicitly present as reference utterances. This flexibility has made VAEs a powerful tool for TTS-for-ASR, leading to significant Word Error Rate (WER) improvements in scenarios with limited real data @sun_generating_2020. However, VAEs can be susceptible to #emph[posterior collapse], a phenomenon where the KL divergence term dominates the loss, causing the model to ignore the latent variable $Z$ and instead rely solely on the text input $T$ as shown by @wang_vaecollapse_2021. This results in a degenerate latent space that fails to capture diverse stylistic variations, thereby reducing the effective diversity of the generated speech. Careful tuning and architectural considerations are necessary to mitigate this issue.

==== Explicit Attribute Conditioning and Augmentation

To address the limitations of latent representations in providing fine-grained control and interpretability, an alternative paradigm focuses on explicit conditioning on measurable attributes and post-generation data augmentation. While latent representations offer an unsupervised means to inject variability, they often provide limited insight into which specific factors of speech are lacking in diversity. By directly controlling measurable attributes, we can more precisely diagnose and address deficiencies of synthetic speech in terms of factors like #smallcaps[Speaker] identity, #smallcaps[Prosody], and #smallcaps[Ambient] conditions. This approach allows for the precise injection of specific variabilities that are known to improve the robustness of ASR training, such as variations in prosody or acoustic conditions @rossenbach_duration_2023. Post-generation augmentation serves as a complementary strategy, focusing on transforming the clean synthetic audio output to simulate external, real-world acoustic variability without requiring alterations to the core synthesis model.

=== Controlled Diversity Enhancement

This work introduces a controlled experimental framework to systematically enhance synthetic speech diversity and to explain the persistent WERR gap discussed in @05_werr_results. While Variantional Autoencoders and Global Style tokens can capture speech diversity, they are hard to interpret. To investigate which factors of the speech most contribute to the TTS-for-ASR gap, we focus on explicit conditioning, where measurable speech attributes are directly incorporated into the TTS generation process, as well as on post-generation augmentation.

==== Core Architecture and Variance Adapter

A class of modern non-autoregressive TTS models, including FastSpeech @ren_fastspeech_2019, FastSpeech 2 @ren_fastspeech_2021, and FastPitch @lancucki_fastpitch_2021, implement explicit attribute conditioning through a component often referred to as a #emph[variance adapter] or #emph[variance predictor]. Our work is based on the FastSpeech 2 architecture. The variance adapter module is inserted between the text encoder (Enc.) and the acoustic decoder (Dec.) of the TTS system, as illustrated in @fig:fig_variance_adapter.

#figure(
image("../figures/6/variance_adapter.png", width: 70%),
caption: [The TTS architecture including the variance adapter which predicts phone-wise duration and frame-wise pitch and energy values. The length regulator (LR) repeats the phone sequence based on the predicted durations.],
placement: top,
) <fig_variance_adapter>

These systems operate on a phone sequence, which is aligned with the audio using forced alignment during data preprocessing, as detailed in @02_prosody_rep. In FastSpeech 2, after the duration for each phone is predicted by the variance adapter, the hidden representations from the encoder are expanded accordingly using a length regulator (LR). This mechanism addresses the alignment problem in non-autoregressive models, mapping a short text sequence to a much longer acoustic sequence. Simultaneously, other predictors within the variance adapter predict frame-level pitch (F0) and energy contours from the reference speech $S$. This modified representation is then passed to the acoustic decoder to generate the Mel spectrogram.

==== Attribute Selection and Probabilistic Sampling

The attributes selected for explicit conditioning are perceptually salient correlates of speech, enabling fine-tuned control over the synthesised utterance. As elaborated in @02_factors, these attributes can be broadly categorised into #smallcaps[Speaker], #smallcaps[Prosody], and #smallcaps[Ambient] factors. For our experiments, we use high-level learned speaker embeddings (d-vectors) for the #smallcaps[Speaker] factor. For #smallcaps[Prosody], we use the fundamental frequency (pitch), energy, and phoneme durations. For the #smallcaps[Ambient] factor, we use metrics that correlate with reverberation (Speech-to-Reverberation Modulation Energy Ratio, SRMR) and signal-to-noise ratio (Waveform Amplitude Distribution Analysis SNR, WADA SNR).

To generate realistic and diverse values for these attributes at inference time, we introduce a method based on probabilistic sampling. This is a key aspect of our methodology for injecting controlled variability. We fit speaker-dependent Gaussian Mixture Models (GMMs) to the distributions of the utterance-level statistics (e.g., mean pitch, mean energy, speaking rate, SRMR, and WADA SNR) observed in the real training data. For each speaker, a GMM captures the multi-modal nature of their attribute distributions. At inference time, values for the conditioning variable set $Z$ are sampled from these trained GMMs. This process allows for the generation of varied, yet statistically plausible, attributes for the synthetic speech, aiming to approximate the true attribute distribution $Q(Z)$ and thereby inject realistic variability into the synthesised output.

==== Augmenting the Acoustic Environment

Post-generation data augmentation is employed as a complementary strategy. This external approach is particularly effective for simulating environmental effects that are difficult for a generative model to synthesise internally but are straightforward to simulate. Formally, given a synthetic utterance $tilde(S) = f_theta (T, Z)$, augmentation applies a transformation function $A(dot)$ to produce an augmented utterance $tilde(S)' = A(tilde(S))$. The function $A$ introduces factors like background noise and reverberation, enriching the acoustic realism of the dataset. Our methodology involves adding background noise from the AudioSet corpus @gemmeke_audioset_2017 and convolving the audio with Room Impulse Responses (RIRs) to simulate reverberation, using the toolkit by #citep(<jordal_audiomentations_2022>).#footnote[#link("https://github.com/iver56/audiomentations", underline[github.com/iver56/audiomentations])]

==== Quantifying Distributional Distance

In addition to using the task-based WERR metric, we also directly quantify the distance between the distributions of the real and synthetic attributes. As introduced in @03_objective_metrics and discussed in detail in @09_dist, distributional metrics offer a way to compare entire sets of speech samples. We use the 2-Wasserstein distance, which is intuitively understood as the "Earth Mover's Distance" and measures the minimum cost to transform one distribution into another @rubner_earth_2000. For the one-dimensional attributes (#smallcaps[Prosody] and #smallcaps[Ambient] factors), we compute the 2-Wasserstein distance directly from their empirical distributions. For the high-dimensional d-vectors (#smallcaps[Speaker] factor), we compute the Fréchet distance, which is the 2-Wasserstein distance between two multivariate Gaussians fitted to the data. This allows for a quantitative assessment of how closely the synthetic attribute distributions match the real ones.

=== Experimental Evaluation

To evaluate the effectiveness of these diversity enhancement paradigms, we conduct a series of controlled experiments, incrementally enhancing a baseline TTS system and quantifying the improvements via both the WERR and direct measurement of distributional distances.

==== Experimental Setup

The experiments follow the controlled TTS-for-ASR setup detailed in @05_setup. All experiments utilise data from the `train-clean-360` split of the LibriTTS dataset @zen_libritts_2019, with data splits for TTS training, synthetic speech generation, and ASR training kept strictly separate to ensure fair comparison. 10 hours were held out for TTS-for-ASR training, with the rest split into $D^"TTS"_"train"$ and $D^"ASR"_"train"$. The TTS system is a multi-speaker FastSpeech~2 model, and the ASR system is the hybrid HMM-TDNN model described previously. We define five incrementally enhanced systems for evaluation:
The #emph[Baseline System] generates speech conditioned only on speaker d-vectors.
The #emph[Environment System] extends the baseline by incorporating environmental attributes (#smallcaps[Ambient] factor) into the variance adapter, training the model to predict frame-level SRMR and WADA SNR.
The #emph[Attributes System] builds on the baseline by explicitly conditioning on utterance-level attributes sampled from speaker-dependent GMMs. These attributes are mean pitch, energy, duration, SRMR, and WADA SNR.
The #emph[Augmentation System] applies post-generation augmentation (additive noise and RIRs) to the output of the #emph[Attributes System].
The #emph[Oracle System] serves as an empirical upper bound for the effectiveness of explicit conditioning. It uses the ground-truth values for all attributes for conditioning, instead of sampling from GMMs, and also includes post-generation augmentation. This allows us to quantify the maximum potential improvement if attributes could be perfectly controlled.

==== Results and Analysis

#figure(
  table(
    columns: (auto, auto, auto),
    align: center,
    toprule(),
    table.header([*System*], [*WER*], [*WERR*]),
    toprule(),
    [Baseline], [48.6 ± 0.43], [3.66 ± 0.03],
    [\+ Environment], [49.2 ± 0.51], [3.70 ± 0.04],
    [\+ Attributes], [47.2 ± 0.39], [3.55 ± 0.03],
    [\+ Augmentation], [44.0 ± 0.33], [3.31 ± 0.02],
    midrule(),
    [Oracle], [*43.0* ± 0.29], [*3.24* ± 0.02],
    [Real Data], [13.3], [1.0],
    toprule(),
  ),
  caption: [Values of the WER and WERR for our systems.],
  placement: top,
) <tab_div_systems>

#figure(
  table(
    columns: (2.5fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1.1fr, 1.1fr),
    align: center,
    toprule(),
    [*System*], table.cell(colspan: 2)[*Speaker*], table.cell(colspan: 3)[*Prosody*], table.cell(colspan: 2)[*Environment*],
    [], [#set text(10pt); Intra], [#set text(10pt); Inter], [#set text(10pt); F0], [#set text(10pt); Energy], [#set text(10pt); SR], [#set text(10pt); SRMR], [#set text(10pt); WADA],
    midrule(),
    [Baseline], [0.36], [0.57], [0.57], [1.97], [0.21], [1.58], [0.57],
    [\+ Environment], [0.33], [0.93], [0.13], [1.35], [0.17], [1.33], [0.56],
    [\+ Attributes], [0.30], [*0.50*], [0.06], [0.15], [0.15], [1.23], [0.58],
    [\+ Augmentation], [0.44], [0.66], [*0.02*], [*0.08*], [0.15], [*0.05*], [*0.23*],
    midrule(),
    [Oracle], [*0.17*], [0.65], [*0.09*], [0.81], [0.09], [0.16], [0.27],
    bottomrule(),
  ),
  caption: [Wasserstein-2 distance of the different attribute distributions for our systems. Lower is better.],
  placement: top,
) <tab_div_dist>

#figure(grid(columns: 1, row-gutter: 2mm, column-gutter: 1mm,
  image("../figures/6/tts_dist_a.png", width: 110%),
  [
    a) Distributions produced by the baseline TTS system.
  ],
  image("../figures/6/tts_dist_b.png", width: 110%),
  [
    b) Distributions produced by the TTS system utilising measures and with acoustic environment augmentation.
  ]),
  caption: [Distributions of the real (blue) and synthetic (orange) measures for the baseline (top) and improved (bottom) systems.],
  placement: top,
) <fig_div_dist_plot>

Our experimental results confirm that targeted diversity enhancements reduce the synthetic-real gap, a conclusion supported by both the WERR metric and direct distributional distance measurements. @tab_div_systems summarises the WER and WERR values, while @tab_div_dist shows the corresponding Wasserstein-2 distances for the different attributes. The #emph[Baseline System] yielded a WERR of 3.66, underscoring the significant initial disparity between synthetic and real speech for ASR training.

===== Impact on Prosodic Realism
The prosody of the #emph[Baseline System] shows significant deviation from real speech. As illustrated in the top row of @fig_div_dist_plot and quantified in @tab_div_dist, the synthetic distributions for prosodic attributes diverge noticeably. The pitch (F0) distribution is bimodal rather than unimodal, the energy distribution is shifted, and the duration (SR) distribution exhibits a smaller variance. The #emph[Attributes System], which conditions on GMM-sampled prosody statistics, dramatically reduces these distances. The Wasserstein-2 distance for F0 plummets from 0.57 to 0.06, and for energy from 1.97 to 0.15. This confirms that explicitly modelling and sampling from attribute distributions enables the generation of significantly more realistic prosody. This improvement in distributional similarity translates to a reduction in WERR to 3.55.

===== Impact on Environmental Diversity
Modelling the acoustic environment proves to be a more complex challenge. The #emph[Environment System], which attempted to predict environmental attributes internally via the variance adapter, was ineffective. It resulted in a slightly higher WERR of 3.70 and had a limited effect on the SRMR and WADA SNR distances. This suggests that the TTS model struggles to synthesise the complex characteristics of noise and reverberation directly. In stark contrast, the #emph[Augmentation System], which applied these effects in a post-processing step, yielded the most substantial improvement. It achieved the lowest WERR of 3.31 among the practical systems and drastically reduced the environmental attribute distances, with the SRMR distance dropping from 1.58 (in the baseline) to 0.05. The visual alignment in the bottom row of @fig_div_dist_plot confirms this success. This finding strongly indicates that for complex acoustic phenomena like environmental conditions, direct simulation via augmentation is a more effective strategy than internal generation within the synthesis model.

===== Impact on Speaker Diversity
For speaker characteristics, the #emph[Attributes System] reduced both the intra-speaker distance (from 0.36 to 0.30) and the inter-speaker distance (from 0.57 to 0.50). This suggests that conditioning on utterance-level statistics helps to better model the specific vocal characteristics of a speaker for a given utterance, improving both consistency within a speaker and distinctiveness between speakers. However, the #emph[Oracle System] achieves a significantly lower intra-speaker distance of 0.17. This implies that our GMM-based sampling of a single d-vector per speaker is a key limitation and does not fully capture the range of intra-speaker variability present in the real data.

===== Overall Performance and the Remaining Gap
The incremental enhancements lead to a progressive reduction in the WERR, from 3.66 for the baseline to 3.31 for the #emph[Augmentation System], a relative improvement of nearly 10%. The #emph[Oracle System] provides an empirical upper bound, reaching a WERR of 3.24. While these improvements are significant, a large gap remains when compared to training on real data (WERR of 1.0). The WER of the Oracle system is 43.0%, more than three times higher than the 13.3% WER achieved with real data. This substantial residual gap demonstrates that even with perfect knowledge of the specific speaker, prosodic, and environmental attributes modelled in this work, the synthetic speech is still not a complete substitute for real speech in the context of ASR training.

=== Summary of Findings and Limitations

The experimental results provide valuable insights into both the effectiveness and the limitations of enhancing synthetic speech diversity through explicit conditioning and augmentation -- our systematic evaluation demonstrates that explicitly introducing diversity across #smallcaps[Prosody], #smallcaps[Speaker], and #smallcaps[Ambient] factors can significantly reduce the distributional gap between synthetic and real speech, leading to tangible improvements in downstream ASR performance. The most effective strategy was a combination of conditioning on GMM-sampled attributes and applying post-generation augmentation, which reduced the WERR by approximately 10%. The analysis of distributional distances confirmed that these improvements were a direct result of the synthetic distributions becoming more similar to the real ones.
However, the large residual WERR, even in the #emph[Oracle] condition, points to several inherent limitations. Firstly, the probabilistic models used for attribute sampling may not be sufficient. The speaker-dependent GMMs are a simplification and may not fully capture the complexity and correlations of the true attribute distributions in real speech. Secondly, the TTS architecture itself has inherent limitations. The FastSpeech 2 model, despite conditioning, is trained with an MSE objective that can encourage regression to the mean, potentially limiting the fine-grained acoustic detail it can produce. Thirdly, there are likely unmodelled sources of diversity. The differences between synthetic and real speech may extend beyond the specific attributes explicitly modelled in this work, involving more subtle, entangled acoustic phenomena that current synthesis models fail to capture.

These findings have several implications for the fields of speech synthesis and recognition. The pronounced success of post-generation augmentation for environmental effects suggests that a modular approach may be most effective: TTS systems could focus on generating clean speech with high-fidelity core characteristics (speaker and prosody), while separate, specialised modules handle the simulation of complex acoustic environments. This decouples two very different generation tasks.
Furthermore, the existence of a substantial performance gap even under oracle conditions strongly motivates moving beyond the specific techniques explored here. It suggests that simply improving the control over a fixed set of attributes within an MSE-based framework has its limits. This points towards the need for more fundamental changes in TTS modelling. The persistent gap motivates our investigation into scaling paradigms and alternative training methodologies, such as the diffusion models explored in the next chapter, which may offer a more promising avenue for generating truly diverse and distributionally complete synthetic data.