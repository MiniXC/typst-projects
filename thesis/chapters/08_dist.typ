#import "../abbr.typ"
#import "../quote.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

== Measuring Distributional Distance using TTSDS <08_dist>

#q(
  [#citep(<rubner_earth_2000>)],
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval")],
  [… we want to define a consistent measure of distance, or dissimilarity, between two distributions of mass in a space that is itself endowed with a ground distance. … Practically, it is important that [such] distances between distributions correlate with human perception.]
)

As established throughout this work, it is useful to conceptualise speech as a distribution rather than as isolated instances. This chapter introduces a methodology to quantify the dissimilarity between real and synthetic speech distributions. Our approach moves beyond single-utterance evaluation and aims to provide a consistent measure of distance that correlates with human perception. We begin by discussing the theoretical underpinnings of measuring distances between complex distributions, before detailing the first iteration of our proposed metric, the Text-to-Speech Distribution Score (TTSDS).

The primary contribution of this chapter is the introduction of the Text-to-Speech Distribution Score (TTSDS), a robust and factorised framework for objectively evaluating synthetic speech by measuring the distributional distance to real speech. This framework utilises a range of perceptually-motivated factors, incorporating novel feature extractors such as the Masked Prosody Model (MPM), a self-supervised model designed to learn representations of prosodic structure. We conduct an extensive validation of this methodology, demonstrating that TTSDS scores strongly correlate with subjective human judgments across diverse #abbr.a[TTS] systems and datasets spanning over a decade of research. These contributions were covered in the following works:

- #cite(<minixhofer_ttsds_2024>, form: "full")

- #cite(<wallbridge_mpm_2025>, form: "full")

=== The Challenge of Speech Distributions

When considering the entire space of possible speech recordings, even under specific constraints, the complexity of accurately replicating the real speech distribution becomes evident. For example, if utterances are restricted to a maximum duration of 60 seconds, and each data point within an utterance is quantised to one of $2^16$ values (corresponding to a 16-bit depth), with a sampling rate set at 16 kHz, this yields a total of $16,000 times 60 = 960,000$ values per recording. Consequently, the number of potential unique recordings would be $2^(16 times 960,000)$, representing a vast space. However, it is crucial to recognise that to human listeners, the overwhelming majority of these theoretical recordings would manifest as incoherent or meaningless noise.

In the development of a system designed to produce synthetic speech, the objective is to accurately model the real speech distribution, which is a comparatively small subset within this expansive recording space. The "performance" of such models is often verified by collecting subjective judgments from human listeners, as detailed in @03_subjective. Alternatively, the degree to which a synthetic distribution resembles its real counterpart can be objectively quantified, as illustrated by the differing distributions of speaker embeddings visualised in @fig:fig_vector.

#figure(
  image("../figures/8/xvector.svg", width: 100%),
  placement: top,
  caption: [KDE of X-Vector speaker embeddings in 2D PCA space for ground truth, synthetic, and noise data, with normalized density scaled by $10^(-5)$.],
) <fig_vector>

==== Quantifying Distributional Distance

An intuitive approach to quantifying the dissimilarity between two distributions is the #abbr.a[EMD], a concept initially conceptualised and introduced by #citep(<rubner_earth_2000>) for the purpose of assessing perceptual similarity in image retrieval tasks. This metric is conceptually derived from the Wasserstein distance @vaserstein_markov_1969, which, in turn, draws upon the Kantorovich–Rubinstein metric @kantorovich_planning_1939. The fundamental motivation behind the #abbr.a[EMD] is articulated as follows:

#q(
  [#citep(<rubner_earth_2000>)],
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval")],
  [Intuitively, given two distributions, one can be seen as a mass of earth properly spread in space, the other as a collection of holes in that same space. Then, the #abbr.a[EMD] measures the least amount of work needed to fill the holes with earth. Here, a unit of work corresponds to transporting a unit of earth by a unit of ground distance.]
)

This conceptualization situates the #abbr.a[EMD] within the broader class of transport problems @hitchcock_transport_1941. However, when applied to speech representations, which are frequently characterized by high dimensionality @wan_generalized_2018, the direct computation of the #abbr.a[EMD] can become computationally expensive. While the generalized #abbr.a[EMD] is associated with significant computational complexity, a particular variant, specifically the #emph[2-Wasserstein distance], offers computationally tractable solutions in certain contexts that are highly pertinent to the comparison of speech representation distributions.

Formally, the Wasserstein distance quantifies the dissimilarity between two empirical probability distributions, which we denote as the real distribution $Q$ and the synthetic distribution $tilde(Q)$. Its definition is predicated on determining the minimum cost required to transform one distribution into the other. This cost is determined by multiplying the quantity of mass transported by the Euclidean distance over which it is moved. The collective set of all conceivable methods for mass redistribution is termed the transport plans, denoted as $Pi(Q, tilde(Q))$. Within this framework, each transport plan $gamma(x,y)$ represents a joint probability distribution, whose marginal distributions correspond to $Q$ and $tilde(Q)$. The $p$-Wasserstein distance is then defined as the minimum cost across all valid transport plans:

$ W_p (Q, tilde(Q)) = (inf_(gamma in Pi(Q, tilde(Q))) E_((x,y)~gamma)[d(x,y)^p])^(1/p) $

where $d(x,y)$ signifies the distance between points $x$ and $y$. For simplicity, we focus on the case where $p=2$ and $d(x,y)$ corresponds to the Euclidean distance $||x-y||_2$, rather than the #abbr.a[EMD] for which $p=1$. Direct computation of this distance for arbitrary high-dimensional distributions remains a challenging endeavor. However, two specific scenarios permit efficient, closed-form solutions. In the one-dimensional case, the 2-Wasserstein distance possesses a straightforward closed-form solution. Given the #abbr.a[CDF]s for the real and synthetic distributions, denoted as $C_R$ and $C_S$, respectively, the squared 2-Wasserstein distance is the squared L2-distance between their inverse functions @kolouri_optimal_2017:

$ W_2^2(P_R, P_S) = integral_0^1(C_R^(-1)(z)-C_S^(-1)(z))^2d z $

For sets of high-dimensional vectors, as proposed by #citep(<heusel_fid_2017>) in the context of image generation, a simplifying assumption can be made: that the embedding distributions can be approximated by multivariate Gaussians. This approximation enables the calculation of the 2-Wasserstein distance in a closed form. Let the real and synthetic embedding distributions be modeled by multivariate Gaussians $N(mu, Sigma)$ and $N(tilde(mu), tilde(Sigma))$, respectively. The squared 2-Wasserstein distance, also recognized as the Fréchet distance @frechet_1925, between these two Gaussian distributions is given by #citep(<dowson_frechet_1982>):

$ W_2^2(Q, tilde(Q)) = ||mu - tilde(mu)||_2^2 + text("Tr")(Sigma + tilde(Sigma)) - 2(Sigma tilde(Sigma))^(1/2)) $

where $mu$ and $tilde(mu)$ are the mean vectors, $Sigma$ and $tilde(Sigma)$ are the covariance matrices, and $text("Tr")(dot)$ is the trace of a matrix. This metric forms the basis of the #abbr.a[FID] @heusel_fid_2017 and the Fréchet Audio Distance @kilgour_fad_2019.

==== Factorised Evaluation

The task of synthetic speech generation inherently lacks a singular ground truth, given its one-to-many nature. Instead, #abbr.a[TTS] evaluation is framed as a problem of distributional similarity. Here, we introduce the initial iteration of our metric, TTSDS, which evaluates the quality of synthetic speech across multiple perceptual factors. The five primary factors were chosen based on their established relevance in speech perception and synthesis evaluation. Intelligibility and prosody are known to be crucial measures of synthetic speech quality @Campbell2007. The speaker factor is included to evaluate how closely #abbr.a[TTS] systems can model realistic speakers @Fan2015, while the environment factor is motivated by the prevalence of audible artifacts in synthetic speech @Wagner2019. Finally, the general factor serves as a holistic measure of distributional similarity, analogous to previous approaches such as the Fréchet DeepSpeech Distance @Gritsenko2020. Each factor is assessed through specific features, as detailed in @tbl:tab_ttsds1_features.

- #smallcaps[Generic]: This factor assesses overall distributional similarity, typically achieved by utilizing #abbr.a[SSL] representations of speech, such as those derived from HuBERT (base) @hsu_hubert_2021  and wav2vec 2.0 (base) models @baevski_wav2vec_2020. These representations are extracted from the intermediate layers of the neural networks, which are thought to be the most generalist @pasad_layer-wise_2021.
- #smallcaps[Ambient]: This factor quantifies the presence of noise or distortion within the speech signal. It leverages two one-dimensional correlates of noise: VoiceFixer @liu_voicefixer_2021 is employed to mitigate noise, followed by the application of #abbr.l[PESQ] @rix_pesq_2001 to measure the perceived quality between the noise-enhanced and original samples. Additionally, WADA SNR @kim_wada_2008 is used to estimate the signal-to-noise ratio of each individual sample.
- #smallcaps[Intelligibility]: This factor measures the ease with which the lexical content of the speech can be recognized. This is achieved by computing the #abbr.l[WER] from reference transcripts and automated transcripts generated by a wav2vec 2.0 model @baevski_wav2vec_2020 fine-tuned on 960 hours of LibriSpeech @panayotov_librispeech_2015, and additionally by a Whisper (small) model @radford_robust_2023.
- #smallcaps[Prosody]: This factor evaluates the realism of speech #smallcaps[Prosody]. It employs frame-level representations derived from our self-supervised prosody model and frame-level pitch features extracted using the WORLD vocoder @morise_world_2016. Furthermore, a proxy for segmental durations is obtained by utilizing HuBERT tokens and measuring their lengths, which corresponds to the number of consecutive occurrences of the same token.
- #smallcaps[Speaker]: This factor quantifies the degree of similarity between the synthetic speaker's voice and that of a real speaker. This is achieved by employing representations obtained from speaker verification systems, specifically d-vectors @wan_generalized_2018 and the more contemporary WeSpeaker @wang_wespeaker_2023 representations.

===== Masked Prosody Model

#figure(
  image("../figures/8/mpm.png", width: 60%),
  placement: top,
  caption: [Architecture of the Masked Prosody Model.],
) <fig_mpm_architecture>

Inspired by masked language models in text processing, the Masked Prosody Model (#abbr.a[MPM]) was developed to learn to reconstruct corrupted sequences of prosodic features—pitch, loudness, and voice activity—independent of lexical content. As illustrated in @fig:fig_mpm_architecture, the model architecture consists of separate input sequences for each prosodic feature, which are then processed by a series of Conformer blocks @gulati_conformer_2020, well-suited to their high-resolution and continuous nature.

The input features are extracted at a resolution of approximately 10ms. Pitch (F0) and voice activity are extracted using the WORLD vocoder @morise_world_2016, while energy is computed as the Root-Mean-Square of each Mel spectrogram frame. All feature contours are normalised across the utterance, which removes information about absolute feature values but allows the model to encode prosody from unseen speakers. Before masking, the continuous input sequences are quantised into discrete codebooks. Random segments of the aligned sequences are then masked such that approximately 50% of the input signal remains. The model is trained to reconstruct the original quantised values for each feature with an independent Categorical Cross Entropy loss.

The downstream evaluation of the model's representations on tasks such as syllable segmentation, emotion recognition, and boundary/prominence detection was conducted as part of a joint first-author publication with Sarenne Wallbridge to investigate the systematicity of prosody. However, for the purpose of this thesis, the MPM serves as a robust, #abbr.a[SSL]-based feature extractor for the prosody factor in TTSDS.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left), 
    [#strong[Factor]], [#strong[Feature]], 
    toprule(),
    [#strong[Environment]], [#strong[Noise/Artifacts]],
    [], [VoiceFixer @liu_voicefixer_2021 + PESQ @rix_pesq_2001],
    [], [WADA SNR @kim_wada_2008],
    toprule(),
    [#strong[Speaker]], [#strong[Speaker Embedding]],
    [], [d-vector @wan_generalized_2018],
    [], [WeSpeaker @wang_wespeaker_2023],
    toprule(),
    [#strong[Prosody]], [#strong[Segmental Length]],
    [], [Hubert @hsu_hubert_2021 token length],
    [], [#strong[Pitch]],
    [], [WORLD @morise_world_2016],
    [], [#strong[SSL Representations]],
    [], [MPM (Ours)],
    toprule(),
    [#strong[Intelligibility]], [#strong[ASR WER]],
    [], [wav2vec 2.0 @baevski_wav2vec_2020],
    [], [Whisper @radford_robust_2023],
    toprule(),
    [#strong[General]], [#strong[SSL Representations]],
    [], [Hubert @hsu_hubert_2021],
    [], [wav2vec 2.0 @baevski_wav2vec_2020],
  ),
  caption: [Features used in the TTSDS benchmark and their respective factors. The overall TTSDS score is computed as an average of individual factor scores.],
  placement: top,
) <tab_ttsds1_features>

===== Formulation of TTSDS
The scoring mechanism for each feature within TTSDS is derived by comparing its empirical distribution in synthetic speech $hat(P)(cal(R)(tilde(S))|tilde(D))$ to the distributions obtained from real speech datasets $hat(P)(cal(R)(S)|D)$ and noise datasets $hat(P)(cal(R)(S)|D^"NOISE"_i)$. The normalized similarity score for a given feature $cal(R)(S)$ is defined as:

$ "TTSDS"(D,tilde(D),D^"NOISE") = 100 times (min[W_2(tilde(D),D^"NOISE"_i)]_(i=0)^N) / (W_2(tilde(D),D) + min[W_2(tilde(D),D^"NOISE"_i)]_(i=0)^N $

In this formulation, the term $min[W_2(tilde(D),D^"NOISE"_i)]_(i=0)^N$ represents the minimum 2-Wasserstein distance between the synthetic speech and a designated set of distractor noise datasets, while $W_2(tilde(D),D)$ denotes the 2-Wasserstein distance to the distribution of real speech. This yields scores ranging from 0 to 100, where values exceeding 50 signify a stronger similarity to genuine speech than to noise, as exemplified for the pitch feature in @fig:fig_f0. The final TTSDS score is the unweighted arithmetic mean of the individual factor scores.

#figure(
  image("../figures/8/pitch_distributions.png", width: 80%),
  placement: top,
  caption: [Distribution of $F_0$ in TTSDS for ground-truth, synthetic, and noise datasets. The distance between the synthetic and real distributions ($d_"gt"$) and the distance to noise ($d_"n"$) are shown, as well as how the overall score is computed.],
) <fig_f0>

=== Experimental Validation

#figure(
  image("../figures/8/heatmaps_ttsds1.png", width: 100%),
  placement: top,
  caption: [Development of factor score correlation coefficients over time from early speech synthesis (Blizzard'08) to the latest systems (TTS Arena).],
) <fig_ttsds1_factor_correlation>

To validate our benchmark, we compare its factor and overall scores to subjective measures using three different datasets, ranging from legacy to state-of-the-art systems. The subjective measures utilized for comparison were derived from three primary sources:
- The Blizzard Challenge 2008 @king_blizzard_2008: This challenge provided #abbr.a[MOS] ratings for 22 distinct #abbr.a[TTS] systems across a variety of synthesis tasks. For the purposes of TTSDS evaluation, the "Voice A" audiobook task, which encompasses 15 hours of audio data, was specifically utilized.
- The "Back to the Future" (BTTF) dataset: This dataset facilitates comparisons between unit selection, hybrid, and statistical parametric #abbr.a[HMM]-based systems from the Blizzard Challenge 2013 @le_maguer_back_2022 and more advanced deep learning systems inspired by the Blizzard Challenge 2021 @ling_blizzard_2021, which included architectures such as FastPitch @lancucki_fastpitch_2021 and Tacotron @wang_tacotron_2017.
- The #abbr.a[TTS] Arena leaderboard @ttsarena: This publicly accessible resource offers crowdsourced A/B preference tests for contemporary #abbr.a[TTS] systems, predominantly those leveraging discrete speech representations generated by large language model-like systems. Only systems released between 2023 and 2024 were incorporated into this evaluation.

The reference speech datasets employed to compute the distributional distances in TTSDS include LibriTTS @zen_libritts_2019, LibriTTS-R @koizumi_libritts-r_2023, LJSpeech @ito_lj_2017, VCTK @vctk, and the training sets provided by the various Blizzard challenges @king_blizzard_2008@ling_blizzard_2021@le_maguer_back_2022. From each of these datasets, 100 utterances were randomly sampled, preferentially from their respective test splits if available. For the generation of distractor noise datasets, TTSDS utilised the ESC dataset @piczak_esc_2015 of background noises, in conjunction with synthetic noise types, including random uniform noise, random normal noise, and silent (all zeros and all ones) samples.

We compare our benchmark with two MOS prediction methods. The first is WVMOS @andreev_hifi_2022, which uses a wav2vec 2.0 model fine-tuned to predict #abbr.a[MOS] scores. The second is UTMOS @saeki_utmos_2022, an ensemble MOS prediction system that performed well in the 2022 VoiceMOS challenge.

#figure(
  table(
    columns: (100pt, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: center,
    toprule(),
    rotate(-90deg, reflow: true)[#strong[System]], rotate(-90deg, reflow: true)[#strong[UTMOS]], rotate(-90deg, reflow: true)[#strong[WVMOS]], rotate(-90deg, reflow: true)[#smallcaps[Generic]], rotate(-90deg, reflow: true)[#smallcaps[Environ.]], rotate(-90deg, reflow: true)[#smallcaps[Intell.]], rotate(-90deg, reflow: true)[#smallcaps[Prosody]], rotate(-90deg, reflow: true)[#smallcaps[Speaker]], rotate(-90deg, reflow: true)[#strong[TTSDS]], rotate(-90deg, reflow: true)[#strong[Arena Elo#linebreak()Rating]],
    midrule(),
    [StyleTTS 2], [*4.36*], [4.48], [93.7], [84.7], [91.6], [89.8], [71.5], [86.3], [*1237*],
    [XTTSv2], [3.89], [4.36], [94.3], [79.3], [91.4], [90.5], [*72.6*], [85.6], [1232],
    [OpenVoice], [4.10], [4.57], [91.7], [88.0], [91.6], [*91.8*], [68.8], [*86.4*], [1158],
    [WhisperSpeech], [3.78], [3.89], [90.0], [83.9], [*92.2*], [80.7], [72.4], [83.9], [1149],
    [Parler TTS], [3.97], [4.16], [*94.7*], [80.8], [87.5], [83.0], [74.1], [84.0], [1140],
    [Vokan TTS], [3.80], [4.22], [88.6], [85.1], [91.6], [85.3], [69.1], [83.9], [1126],
    [OpenVoice v2], [4.29], [*4.75*], [90.7], [*91.2*], [91.6], [88.6], [68.7], [86.2], [1120],
    [VoiceCraft 2], [4.21], [3.71], [87.0], [78.0], [91.6], [84.4], [66.0], [81.4], [1114],
    [Pheme], [3.92], [4.26], [94.0], [81.9], [91.5], [85.1], [66.1], [83.7], [1029],
    bottomrule(),
  ),
  caption: [Ranking, factor scores, TTSDS score and MOS predictions for the TTS Arena systems.],
  placement: top,
) <tab_tts_arena_ranking>

==== Results and Analysis

The correlations observed between TTSDS factor scores and subjective evaluation metrics across these temporal stages are depicted in @fig:fig_ttsds1_factor_correlation. Our results show that the TTSDS score, computed as an unweighted average of factors, consistently correlates well with human evaluations, with Spearman coefficients from 0.60 to 0.83. This is in contrast to the baseline MOS predictors, which achieve mixed results, performing well on some datasets but poorly on others, indicating a lack of generalisation.

#figure(grid(columns: 2, row-gutter: 2mm, column-gutter: 1mm,
  image("../figures/8/overtime.png", width: 100%),
  image("../figures/8/wilcoxon.png", width: 100%),
  [a) Development of factor score correlation coeffi- cients over time from early speech synthesis (Blizzard’08) to the latest systems (TTS Arena).],
  [b) Results of Wilcoxon signed-rank tests between systems' extracted features for the TTS Arena dataset.],),
  placement: top,
  caption: [Factor scores over time and Wilcoxon signed-rank test.],
) <fig_overtime>

A more detailed factor-by-factor analysis reveals shifting listener priorities, as shown in @fig:fig_overtime a). For older systems (Blizzard'08), the #smallcaps[Intelligibility] and #smallcaps[Speaker] factors show high correlations, while the #smallcaps[Environment] factor is most important for the mixed-era BTTF dataset, likely due to its ability to detect artifacts in the older systems. For modern systems in the TTS Arena, the #smallcaps[Prosody] factor shows the strongest correlation, suggesting its increased importance as other aspects like intelligibility have improved.

The ranking of modern systems from the TTS Arena is shown in @tbl:tab_tts_arena_ranking. While the UTMOS baseline correctly predicts the best system, its other scores show little correlation with the subjective Elo ratings. In contrast, our prosody factor, speaker factor, and overall TTSDS score correlate well. To assess the metric's discriminative power, a Wilcoxon signed-rank test was performed, with results shown in @fig:fig_overtime b). This shows that while worst-performing systems are distinguishable from top-performers, there are no statistically significant differences between the very best systems, a common challenge in subjective tests as well. This validation establishes TTSDS as a reliable objective measure capable of predicting human perception and providing interpretable insights into specific areas of improvement for #abbr.a[TTS] systems.

While the validation of TTSDS successfully demonstrated that a factorised, distributional approach can provide a robust and interpretable objective metric for synthetic speech, correlating strongly with human judgments across different eras of #abbr.a[TTS] technology, this initial validation was primarily conducted on English audiobook-style speech. The increasing capability of modern #abbr.a[TTS] systems necessitates evaluation methods that are robust not only to clean, read speech but also to more challenging real-world conditions, including noisy environments, conversational styles, and diverse speaker populations like children. Furthermore, the global nature of speech technology requires metrics that can be reliably applied across multiple languages. Motivated by these requirements for greater robustness and broader applicability, the next chapter introduces TTSDS2, an enhanced and expanded version of the framework. There, we detail the specific refinements to the feature set and methodology designed to address these challenges and present a comprehensive validation across these diverse domains and languages.