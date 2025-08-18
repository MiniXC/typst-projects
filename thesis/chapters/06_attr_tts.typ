#import "../abbr.typ"
#import "../quote.typ": *
#import "../math.typ": *
#import "../moremath.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

== Enhancing Synthetic Speech Diversity <06_attr>

#q(
[#citep(<oord_wavenet_2016>)],
[#emph[WaveNet: A Generative Model for Raw Audio]],
[We condition the model $dots$ in two different ways: global conditioning and local conditioning.]
)

#ac("TODO: reframe the chapter a bit through local vs global conditioning angle and include WaveNet in background OR find a better quote.")

Building on the significant performance gap between ASR models trained on real versus synthetic speech, as quantified by the Word Error Rate Ratio (WERR) in @05_ttsasr[Chapter], this chapter details various methodologies developed to increase the diversity and realism of synthetic speech. As established, synthetic speech, despite its high human naturalness ratings, often lacks the intricate variability inherent in real human speech. This limitation directly impedes its utility for training robust ASR systems. Here, we systematically explore three complementary paradigms aimed at bridging this distributional divide: learning latent representations, explicit conditioning on attributes, and post-generation data augmentation. By introducing and controlling these aspects of speech, we aim to reduce the distributional distance between synthetic and real speech, which should, in turn, manifest as a reduction in WERR. The discussion progresses from methods that implicitly capture broad stylistic variations to those offering fine-grained, targeted control, culminating in empirical validation that demonstrates their impact. Throughout this chapter, we adhere to the formal notation established in the introduction, where $Q_theta$ represents the Text-to-Speech (TTS) model's approximation of the true speech distribution $Q(S|T)$.

=== Learning Latent Representations

Latent representations offer a powerful, unsupervised approach to capturing and injecting stylistic and acoustic variability into synthetic speech. These methods implicitly learn a lower-dimensional encoding of information (e.g., style, prosody) from the training data, without requiring explicit labeling or pre-definition of these attributes. As discussed in @02_factors[Chapter 2], such representations are often classified as "learned transformations" and can range from "mid-level" to "high-level" abstractions, encoding "generic," "speaker," or "prosody" factors. This paradigm is particularly valuable for addressing the inherent "one-to-many" problem in TTS, where a single text input can correspond to countless valid acoustic realizations. By allowing sampling from a learned distribution of styles, these methods enable the generation of diverse outputs, which has proven beneficial in TTS-for-ASR for improving model robustness @sun_generating_2020. Mathematically, these methods aim to model a latent variable $Z$ such that the TTS distribution becomes $Q_theta (tilde(S) | T, Z)$, where $Z$ is a vector sampled from a learned or constrained distribution. Unlike explicit conditioning, $Z$ is not inferred from specific attributes of a reference utterance at inference time but is rather sampled from a prior distribution (e.g., a standard normal distribution) to generate novel, unobserved styles.

==== Global Style Tokens

Global Style Tokens (GSTs) provide a discrete approach to modeling stylistic variations in speech @wang_style_2018. The core idea involves learning a fixed set of $K$ distinct, global style embeddings or "tokens," denoted as $cal(G) = {g_1, ..., g_K}$. During the training phase, the TTS model is provided with a reference utterance $S$. An attention mechanism within the model then learns to weigh these tokens based on the acoustic characteristics of the reference audio. This process encodes the utterance's observed style as a weighted combination of the learned tokens. Formally, the style vector $Z$ for a given reference utterance $S$ is computed as:
$ Z = sum_(k=1)^K alpha_k g_k $
where $alpha_k$ are the attention weights learned from the reference $S$. These weights sum to one, making $Z$ a convex combination of the style tokens. At inference time, instead of deriving $Z$ from a reference utterance, the style tokens can be directly selected, combined, or interpolated to control the output style, enabling the generation of speech that exhibits desired characteristics (e.g., more expressive or neutral). While effective for inducing categorical style shifts and improving perceived naturalness, the discrete nature of GSTs can limit the ability to capture and generate very fine-grained, continuous variations compared to other latent methods.

==== Variational Autoencoders

Variational Autoencoders (VAEs) extend the concept of latent style modeling by learning a continuous latent distribution, allowing for smoother and more nuanced sampling of stylistic variations @kingma_auto-encoding_2013. A VAE consists of an encoder and a decoder. The encoder maps a given reference speech utterance $S$ into the parameters of a probabilistic latent distribution, typically assumed to be a Gaussian posterior distribution. Specifically, the encoder learns to output the mean $mu_phi (S)$ and variance $sigma_phi (S)$ (or log-variance) of this Gaussian:
$ q_phi (Z | S) = cal(N)(mu_phi (S), sigma_phi (S)) $
This approximate posterior $q_phi (Z|S)$ is trained to approximate the true posterior distribution $p(Z|S)$ by maximizing the Evidence Lower Bound (ELBO). The ELBO objective balances two terms: the reconstruction likelihood and a Kullback-Leibler (KL) divergence term that regularizes the latent space $Z$ by pushing $q_phi (Z|S)$ to be close to a simple prior distribution $p (Z)$ (i.e. a standard Gaussian $cal(N)(0, I)$):
$ cal(L)_"ELBO" = EE_(q_phi (Z|S)) [log p_theta (S | Z)] - "KL"(q_phi (Z|S) || p (Z)) $
The TTS decoder then takes a sampled latent vector $Z$ and the input text $T$ to reconstruct the synthetic utterance $tilde(S)$, aiming to approximate $p_theta(S | Z)$. The regularization of the latent space ensures that samples drawn from the simple prior $p(Z)$ at inference time can generate novel and plausible speech styles, even if those specific styles were not explicitly present as reference utterances. This flexibility has made VAEs a powerful tool for TTS-for-ASR, leading to significant Word Error Rate (WER) improvements in scenarios with limited real data @sun_generating_2020. However, VAEs can be susceptible to #emph[posterior collapse], a phenomenon where the KL divergence term dominates the loss, causing the model to ignore the latent variable $Z$ and instead rely solely on the text input $T$ @wang_vaecollapse_2021. This results in a degenerate latent space that fails to capture diverse stylistic variations, thereby reducing the effective diversity of the generated speech. Careful tuning and architectural considerations are necessary to mitigate this issue.

=== Explicit Conditioning and Augmentation

To address the limitations of latent representations in providing fine-grained control and interpretability, this part of the thesis shifts focus to explicit conditioning on attributes and post-generation data augmentation. The primary motivation for this approach is twofold. Firstly, while latent representations offer an unsupervised means to inject variability, they often provide limited insight into which specific factors of speech are lacking in diversity within the synthetic output. By directly controlling measurable attributes, we can more precisely diagnose the deficiencies of synthetic speech in terms of factors like #smallcaps[Speaker] identity, #smallcaps[Prosody], and #smallcaps[Ambient] conditions. Secondly, by systematically varying these attributes, we aim to demonstrate that increased diversity in synthetic speech directly translates to improved ASR performance, thereby providing a more controlled experimental framework to explain the persistent WERR gap discussed in @05_ttsasr[Chapter].

==== Methodology and Novelty

For scenarios requiring more interpretable and targeted control over specific aspects of speech diversity, explicit conditioning directly incorporates measurable speech attributes into the TTS generation process. Unlike latent methods, which infer styles abstractly from data, this approach enables the system to generate speech that adheres to certain attributes (e.g., a specific speaking rate, a particular pitch contour, or speaker identity). As discussed in @02_factors[Chapter], these attributes are typically derived using mid- or high-level representations of #smallcaps[Speaker], #smallcaps[Prosody] or even #smallcaps[Ambient] factors. For example, in ParlerTTS, a prompt specifies speaker characteristics, speaking rate and noisyness of the speech, among other attributes @lyth_parler_2024.
This approach can be particularly advantageous for TTS-for-ASR, as it allows for the precise injection of specific variabilities that are known to improve the robustness of ASR training, such as variations in prosody or acoustic conditions, such as demonstrated by #citep(<rossenbach_duration_2023>) for phone duration.

Formally, the TTS model $f^"TTS"_theta$ is trained to minimize a loss over its conditioned outputs. Let $T$ be the input text, $S$ be the ground-truth speech utterance, and $Z$ be a set of extracted conditioning attributes from $S$. The training objective is to minimize the distance between the model's prediction and the ground-truth target, expressed as:
$ cal(L)(theta) = EE_(T,S,Z) [l(f_theta (T,Z), S)] $
where $l$ is typically a reconstruction loss, such as Mean Squared Error (MSE) or Mean Absolute Error (L1 loss), applied to the acoustic representation (e.g., Mel spectrograms). The choice of reconstruction loss encourages the synthetic output to closely match the ground truth across the specified attributes.

The novelty of our methodology lies in several key aspects. Firstly, we are among the first to systematically condition the TTS model on utterance-level statistics of multiple diverse attributes, encompassing #smallcaps[Speaker], #smallcaps[Prosody], and #smallcaps[Ambient] factors. Secondly, we introduce a method for generating diverse synthetic speech by sampling these utterance-level attributes from speaker-dependent Gaussian Mixture Models (GMMs) during inference, thereby injecting realistic, yet controlled, variability that approximates the true distribution of these attributes. Thirdly, we are novel in also directly conditioning the TTS model on environmental correlates (SRMR and WADA SNR), an approach distinct from purely post-generation noise addition.

==== Variance Adapter

#figure(
image("../figures/6/variance_adapter.png", width: 70%),
caption: [The variance adapter which predicts phone-wise duration and frame-wise pitch and energy values.],
placement: top,
) <fig_variance_adapter>

A class of modern non-autoregressive TTS models; FastSpeech @ren_fastspeech_2019, FastSpeech 2 @ren_fastspeech_2021 and FastPitch @lancucki_fastpitch_2021 implement explicit attribute conditioning through a component often referred to as a #emph[variance adapter] or #emph[variance predictor]. This module is inserted between the text encoder and the acoustic decoder of the TTS system, as illustrated in @fig_variance_adapter.

These systems operate on a phone sequence, which is aligned with the audio using forced alignment (see @02_prosody_rep) during data preprocessing. Pitch and energy ground-truth value are extracted from the reference speech $S$ during data preparation as well. In FastSpeech 2, after the duration is predicted, the hidden representations are expanded accordingly, which until this point each corresponds to one input phone. This mechanism addresses the alignment problem in non-autoregressive models, mapping a short text sequence to a much longer acoustic sequence. Simultaneously, other predictors within the variance adapter predict frame-level pitch (F0) and energy contours @lancucki_fastpitch_2021. The architecture of FastPitch, for example, predicts a single pitch value for every input symbol and projects it to match the dimensionality of the hidden representation, which is then added to it. This modified representation is then passed to the acoustic decoder to generate the Mel spectrogram.

==== Controllable Attributes <06_prosodic_correlates>

The attributes selected for explicit conditioning are typically perceptually salient correlates of speech, enabling fine-tuned control over different factors of the synthesized utterance. As elaborated in @02_factors[Chapter 2], these attributes can be broadly categorized and can manifest as either fixed-dimensional vectors or sequences that vary with the length of the speech signal:

#smallcaps[Speaker]: For instance, high-level learned speaker embeddings, such as d-vectors @wan_generalized_2018 or x-vectors @snyder_x_2018, are fixed-dimensional representations capturing the unique, identifying characteristics of an individual's voice. These are crucial for multi-speaker and voice-cloning TTS systems.

#smallcaps[Prosody]: This factor encompasses attributes that govern the melody and rhythm of speech. These are typically sequences that vary with the length of the speech signal.
    #emph[Pitch (F0)]: The fundamental frequency contour, extracted using algorithms like PyWORLD @morise_world_2016, represents the perceived pitch of the voice. Its variations convey intonation and emotion.
    #emph[Energy]: The Root Mean Square (RMS) energy of Mel frames reflects the perceived loudness or intensity of the speech.
    #emph[Duration]: The length of phonemes and pauses, often obtained via forced alignment tools like the Montreal Forced Aligner @mcauliffe_montreal_2017, defines the speaking rate and rhythm of the utterance.

#smallcaps[Ambient]: Attributes related to the recording conditions, such as:
    #emph[Speech-to-Reverberation Modulation Energy Ratio (SRMR)]: A metric correlating with the amount of reverberation present in an audio signal @kinoshita_reverb_2013.
    #emph[Waveform Amplitude Distribution Analysis Signal-to-Noise Ratio (WADA SNR)]: An estimation of the signal-to-noise ratio (SNR) @kim_wada_2008.

By conditioning on these diverse attributes, TTS systems can generate speech that matches specific prosodic contours or environmental characteristics, thereby enhancing the realism and variability of the ASR training data.

To generate realistic and diverse values for $Z$ at inference time, especially for attributes that vary at the utterance level (e.g., mean pitch, energy, duration, SRMR, SNR), we often employ probabilistic models. A common approach, as used in our work, involves fitting speaker-dependent Gaussian Mixture Models (GMMs) to the distributions of these attributes observed in the real training data. For each speaker, a GMM captures the multi-modal nature of their attribute distributions. At inference, values for $Z$ are then sampled from these trained GMMs, allowing for the generation of varied, yet statistically plausible, attributes for the synthetic speech. This process aims to approximate the true distribution of these attributes $Q(Z)$ and inject realistic variability into the synthesized output.

==== Post-Generation Data Augmentation

Post-generation data augmentation serves as a complementary strategy to internal TTS model enhancements, focusing on transforming the clean, raw synthetic audio output to simulate external, real-world acoustic variability. This external approach is particularly effective and straightforward for TTS-for-ASR, as it directly addresses potential environmental mismatches and noise present in real speech, without requiring alterations to the core TTS synthesis model itself @rossenbach_synthattention_2020. The idea behind this is that some processes are easier to simulate than to generate, such as adding gaussian background noise.

Formally, given a synthetic utterance $tilde(S) = f_theta (T, Z)$ generated by the TTS model (where $Z$ includes any internal conditioning, e.g., speaker and prosody), post-generation augmentation applies a transformation function $A(dot)$ to produce $tilde(S)'$:
$tilde(S)' = A(tilde(S))$
where $A$ introduces factors like background noise or reverberation. This results in an augmented synthetic utterance $tilde(S)'$ that aims to approximate $Q(S | T, Z, Z_text("env"))$, thereby enriching the acoustic realism of the dataset.

Common techniques for post-generation augmentation include:
#emph[Adding Background Noise]: This involves superimposing diverse types of background noise (e.g., from an openly available dataset like AudioSet @gemmeke_audioset_2017) onto the clean synthetic speech. For instance, Gaussian noise can be added with a target Signal-to-Noise Ratio (SNR) ranging from 5 dB to 40 dB, simulating various levels of environmental interference.
#emph[Convolving with Room Impulse Responses (RIRs)]: To mimic reverberation, synthetic speech can be convolved with RIRs sampled from various acoustic environments. This simulates the effect of sound reflecting off surfaces in a room. For example, RIRs can be applied with a probability (e.g., 0.8) and target RT60 (reverberation time) ranging from 0.15s to 0.8s, replicating different room acoustics.

Tools such as the `audiomentations` library @jordal_audiomentations_2022 facilitate the probabilistic application of these and other augmentation techniques, allowing for a diverse range of acoustic conditions to be simulated. While highly effective for introducing acoustic environmental variability and improving robustness to noise, post-generation augmentation cannot retroactively adjust intrinsic speech properties like prosodic patterns or speaker characteristics that are determined during the core synthesis process. Therefore, it is best utilized synergistically with internal TTS enhancements (latent methods or explicit conditioning) to achieve comprehensive diversity across all relevant speech factors.

=== Experiments on TTS-for-ASR Diversity

#figure(
  image("../figures/6/attr_tts_arch.png", width: 70%),
  placement: top,
  caption: [Pipeline of a TTS system with explicit attribute conditioning and post-generation augmentation. #ac("TODO: new style and labels")],
) <fig_tts_aug>

To evaluate the effectiveness of these diversity enhancement paradigms, we conduct a series of controlled experiments, incrementally enhancing a baseline TTS system and quantifying the improvements via the Word Error Rate Ratio (WERR), as defined in @05_ttsasr[Chapter]. This experimental setup ensures that observed differences in ASR performance can be directly attributed to the increased diversity of the synthetic training data, rather than confounding factors.

The overall framework for our TTS training involves minimizing an MSE loss between the generated Mel spectrogram $f^"TTS"_theta (T, Z)$ and the ground-truth Mel spectrogram $S$:
$ cal(L)(theta) = EE_(T,S,Z) [||S - f^"TTS"_theta (T, Z)||^2_2] $
This objective encourages the TTS model $f^"TTS"_theta$ to learn an approximation $Q_theta (S | T, Z)$ of the true distribution of speech conditioned on text and attributes. For ASR training, we utilize a standard discriminative objective, specifically the Lattice-Free Maximum Mutual Information (LF-MMI) criterion, to learn the parameters $phi$ of the ASR model $f^"ASR"_phi$. The effectiveness of the synthetic training data is then measured by the WER of $f^"ASR"_phi$ when evaluated on a real test set.

All experiments utilize data from the `train-clean-360` split of the LibriTTS dataset @zen_libritts_2019. From this split, we select speakers with at least 100 utterances, resulting in a total of 684 speakers. For each speaker, half of their utterances are reserved as the TTS training set, and the other half are used for inference to generate the synthetic speech (ensuring that the ASR model is evaluated on speech generated from unseen utterances). To ensure a fair comparison, the transcripts for the synthetic dataset are carefully selected to balance across speakers, and a total of 10 hours of synthetic audio is generated for each system, matching the size and linguistic content of the real ASR training data. The seen/unseen splits for TTS inference ensure that the synthetic speech used for ASR training is generated from transcripts that the TTS model has not previously seen during its own training phase, thereby probing its generalization capabilities. For details on this procedure, see @05_setup.

The baseline TTS system is a multi-speaker FastSpeech 2 @ren_fastspeech_2021 model. This model is conditioned on speaker identity using d-vector embeddings ($E_text("SPK")$) extracted from a speaker verification network @wan_generalized_2018. The Mel spectrogram outputs from FastSpeech 2 are converted into raw audio waveforms using a HiFi-GAN vocoder @kong_hifigan_2020. The TTS models are trained for 40 epochs, which required approximately 24 hours on 4 NVIDIA GTX 1080 Ti GPUs.

The ASR training setup uses a 6-layer hybrid HMM-Time-Delay Neural Network (TDNN) system, trained with the LF-MMI objective @povey_kaldi_2011 for 4 epochs, which required approximately 3-4 hours of training on 4 NVIDIA GTX 1080 Ti GPUs. We keep the ASR system configuration minimal and consistent across experiments to ensure that differences in WER primarily reflect differences in the acoustic properties and diversity of the synthetic training data, rather than confounding factors related to ASR model architecture or training.

==== Increasing Diversity

We incrementally build upon the baseline to evaluate the impact of different diversity enhancement techniques as follows:
As described above, the #emph[Baseline System] generates synthetic speech conditioned only on speaker d-vectors.
Next, the #emph[Environment System] extends the baseline by incorporating environmental attributes (#smallcaps[Ambient] factor). Specifically, the FastSpeech 2 variance adapter is extended to predict frame-level Speech-to-Reverberation Modulation Energy Ratio (SRMR) and Waveform Amplitude Distribution Analysis Signal-to-Noise Ratio (WADA SNR) features in addition to the original prosodic attributes of energy and pitch.
The #emph[Attributes System] system builds on the baseline by explicitly conditioning on and predicting various utterance-level attributes, which are sampled from speaker-dependent GMMs. These attributes include mean pitch ($F_("F0")$), energy, duration (speaking rate), SRMR, and WADA SNR. Two components per GMM and a variance floor of $10^{-3}$ are used for sampling.
The post-generation #emph[Augmentation System]: builds additionally applies post-generation augmentation to its synthesized output. This involves adding Gaussian noise with a target SNR ranging from 5 dB to 40 dB and convolving the audio with Room Impulse Responses (RIRs) with a target RT60 (reverberation time) ranging from 0.15s to 0.8s. RIRs are applied with a probability of 0.8. These augmentations are performed using the `audiomentations` library, and the parameters of both were adjusted to represent the range of values in the original data.
Finally, the #emph[Oracle System] serves as an empirical upper bound for the effectiveness of explicit conditioning. For this system, the ground-truth values for all attributes are used directly for conditioning the TTS model during synthesis, instead of sampling from GMMs. Post-generation augmentation (noise and RIRs) is also applied to its output. This allows us to quantify the maximum potential improvement achievable if attributes could be perfectly predicted or controlled. @fig_tts_aug illustrates the full system with all systems applied.

=== Implications on Distribution Gap

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
    [Real Data], [13.3], [-],
    toprule(),
  ),
  caption: [Values of the WER and WERR for our systems.],
  placement: top,
) <tab_div_systems>

Our experimental results, which quantify the impact of these diversity enhancements on ASR performance, confirm that targeted improvements can reduce the synthetic-real gap, although they do not fully eliminate it. @tab_div_systems provides a summary of the WERs and WERR values for each system: The #emph[Baseline System], conditioned only on speaker d-vectors, yielded a WERR of 3.66. This substantial value underscores the significant distributional disparity between basic synthetic speech and real speech, confirming the limited variability discussed in the initial motivation for this thesis. This means that an ASR model trained on baseline synthetic speech performs 3.66 times worse than one trained on real speech.
The #emph[Environment System], which extended the FastSpeech 2 variance adapter to predict frame-level SRMR and WADA SNR features, resulted in a WERR of 3.70. This minor increase in WERR, despite efforts to model environmental factors, suggests that merely training the TTS model to predict these attributes internally, without explicit post-generation application, might not inherently make the generated speech more suitable for ASR robustness, or that the model struggles to accurately capture the full variability of these subtle cues.
The #emph[Attributes System], which conditions on and predicts utterance-level means for pitch ($F_("F0")$), energy, duration (speaking rate), SRMR, and WADA SNR, demonstrated a positive impact, reducing the WERR to 3.55 (3% relative). This indicates that providing the TTS model with explicit control over these specific, perceptually motivated attributes, and injecting variability via GMM sampling, helps bridge the distributional gap.
In the #smallcaps[Ambient] domain, the #emph[Augmentation System], which added post-generation data augmentation, yielded the most substantial reduction in WERR to 3.31, representing a relative gain of approximately 10% over the baseline. This highlights the importance of realistic acoustic environments (background noise and reverberation) for synthetic speech for ASR training, which is not usually deemed important for TTS. The initial #emph[Environment System], which attempted to predict environmental attributes via the variance adapter, showed less effectiveness in reducing WERR. This suggests that the TTS model may struggle to produce the background noise and reverberation characteristics which can be simulated by post-generation data augmentation.

The #emph[Oracle System], which utilized ground-truth attributes, achieves the lowest WERR of 3.24. This performance serves as an empirical upper bound, for our architectural setup indicating the maximum potential improvement achievable if all relevant attributes could be perfectly controlled or predicted. This large remaining gap between the Oracle system and training on `Real Data` (WER of 43% compared to 13%) suggests that while these methods significantly reduce the distributional gap, inherent limitations persist.

While ASR language models can influence WER, our experimental design minimized this interference by keeping the ASR setup consistent and focusing on acoustic modeling, limiting interference of #smallcaps[Semantic] factors. The fact that post-generation augmentation, provided the biggest reduction in WERR, and that the gap remains large after applying the #emph[Oracle System] ground-truth values for conditioning, suggest the TTS system itself has some inherent limitations, either due to how it is trained or limitations.

==== Training and Scaling Limitations

Despite the improvements demonstrated by these methods in enhancing synthetic speech diversity and reducing the WERR, several limitations and challenges remain. Firstly, Post-generation augmentation, while highly effective for #emph[Ambient] factors, cannot retroactively compensate for deficiencies in the core synthesis model's ability to generate natural prosody or speaker characteristics.
Secondly, the gains observed tend to plateau, indicating a potential ceiling for these specific techniques in completely bridging the synthetic-real gap. This suggests that the differences between synthetic and real speech extend beyond the attributes explicitly modeled in this work.  The persistent gap implies that ASR models are sensitive to distributional differences that human listeners may not perceive as "unnatural". Human judges might rate synthetic speech highly because it lacks major distortions and adheres to general phonetic and prosodic rules, yet the synthetic data may occupy a very narrow, high-density region within the vast, complex space of real speech, failing to represent the full diversity required for robust ASR training. TTS-for-ASR thus might be a better proxy for how much of this distribution remains elusive to TTS models. The persistant gap, most likely in part caused by the model training and data itself, motivates investigation into scaling paradigms and alternative training methodologies, which may offer another avenue to generate more diverse and robust synthetic data, as explored in the next chapter.