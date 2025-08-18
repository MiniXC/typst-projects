#import "../abbr.typ"
#import "../quote.typ": *
#import "../comic.typ"
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

== Measuring distributional distance <09_dist>

#q(
  [#citep(<rubner_earth_2000>)],
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval")],
  [… we want to define a consistent measure of distance, or dissimilarity, between two distributions of mass in a space that is itself endowed with a ground distance. … Practically, it is important that [such] distances between distributions correlate with human perception.]
)

As we have established throughout this work it is useful to conceptualize speech as a distribution rather than as isolated instances. In this chapter, we expand this perspective beyond TTS-for-ASR and introduce a methodology to quantify the dissimilarity between real and synthetic speech distributions across diverse systems, domains, and languages. This approach aims to provide a consistent measure of distance that correlates with human perception.

=== Audio and speech distributions

When considering the entire space of possible speech recordings, even under specific constraints, the complexity of accurately replicating the real speech distribution becomes evident. For example, if utterances are restricted to a maximum duration of 60 seconds, and each data point within an utterance is quantized to one of $2^16$ values (corresponding to a 16-bit depth), with a sampling rate set at 16 kHz, this yields a total of $16,000 times 60 = 960,000$ values per recording. Consequently, the number of potential unique recordings would be $2^(16 times 960,000)$, represent a vast space. However, it is crucial to recognize that to human listeners, the overwhelming majority of these theoretical recordings would manifest as incoherent or meaningless noise.

In the development of a system designed to produce synthetic speech, the objective is to accurately model the real speech distribution, which is a comparatively small subset within this expansive recording space. If the precise real speech distribution was already known, it would be unnecessary to model it. Therefore, practitioners typically resort to estimating this distribution from available data. The "performance" of such models is often verified by collecting subjective judgments from human listeners, as detailed in @08_eval[Chapter]. Alternatively, or complementarily, the degree to which a synthetic distribution resembles its real counterpart can be objectively quantified, a methodology that is elaborated upon in the subsequent sections of this chapter.

=== Earth Mover's Distance

#figure(
  image("../figures/9/xvector.svg", width: 100%),
  placement: top,
  caption: [KDE of X-Vector speaker embeddings in 2D PCA space for ground truth, synthetic, and noise data, with normalized density scaled by $10^(-5)$.],
) <fig_vector>

An intuitive approach to quantifying the dissimilarity between two distributions is the #abbr.a[EMD], a concept initially conceptualized and introduced by #citep(<rubner_earth_2000>) for the purpose of assessing perceptual similarity in image retrieval tasks. This metric is conceptually derived from the Wasserstein distance @vaserstein_markov_1969, which, in turn, draws upon the Kantorovich–Rubinstein metric @kantorovich_planning_1939. The fundamental motivation behind the #abbr.a[EMD] is articulated as follows:

#q(
  [#citep(<rubner_earth_2000>)],
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval")],
  [Intuitively, given two distributions, one can be seen as a mass of earth properly spread in space, the other
as a collection of holes in that same space. Then, the #abbr.a[EMD] measures the least amount of work needed to fill the holes with earth. Here, a unit of work corresponds to transporting a unit of earth by a unit of ground
distance.]
)

This conceptualization situates the #abbr.a[EMD] within the broader class of transport problems @hitchcock_transport_1941, which can be analytically solved for specific cases, such as the two-dimensional representation of image histograms as presented in the original formulation @rubner_earth_2000. However, when applied to speech representations, which are frequently characterized by high dimensionality @wan_generalized_2018, the direct computation of the #abbr.a[EMD] can become computationally expensive.

=== Wasserstein metric

While the generalized #abbr.a[EMD] is associated with significant computational complexity, a particular variant, specifically the #emph[2-Wasserstein distance], offers computationally tractable solutions in certain contexts that are highly pertinent to the comparison of speech representation distributions.

Formally, the Wasserstein distance quantifies the dissimilarity between two empirical probability distributions, which we denote as the real distribution $Q$ and the synthetic distribution $tilde(Q)$. Its definition is predicated on determining the minimum cost required to transform one distribution into the other. This cost is determined by multiplying the quantity of mass transported by the Euclidean distance over which it is moved. The collective set of all conceivable methods for mass redistribution is termed the transport plans, denoted as $Pi(Q, tilde(Q))$. Within this framework, each transport plan $gamma(x,y)$ represents a joint probability distribution, whose marginal distributions correspond to $Q$ and $tilde(Q)$.
The $p$-Wasserstein distance is then defined as the minimum cost across all valid transport plans:

$ W_p (Q, tilde(Q)) = (inf_(gamma in Pi(Q, tilde(Q))) E_((x,y)~gamma)[d(x,y)^p])^(1/p) $

where $d(x,y)$ signifies the distance between points $x$ and $y$. For simplicity, we focus on the case where $p=2$ and $d(x,y)$ corresponds to the Euclidean distance $||x-y||_2$, rather than the #abbr.a[EMD] for which $p=1$. Analogous to the generalized #abbr.a[EMD], direct computation of this distance for arbitrary high-dimensional distributions remains a challenging endeavor. However, two specific scenarios permit efficient, closed-form solutions.

#emph[One-dimensional case]

In the one-dimensional case, the 2-Wasserstein distance possesses a straightforward closed-form solution, thereby precluding the necessity of an exhaustive search across all possible transport plans. This distance can be computed directly from the inverse cumulative distribution functions (#abbr.a[CDF]s) of the two distributions. Given the #abbr.a[CDF]s for the real and synthetic distributions, denoted as $C_R$ and $C_S$, respectively, the squared 2-Wasserstein distance is simply the squared L2-distance between their inverse functions @kolouri_optimal_2017:

$ W_2^2(P_R, P_S) = integral_0^1(C_R^(-1)(z)-C_S^(-1)(z))^2d z $

This intrinsic property forms the theoretical underpinning of the Sliced-Wasserstein distance, an approach that computes the average Wasserstein distance between distributions by projecting them onto numerous random one-dimensional subspaces. However, an alternative closed-form solution exists for the high-dimensional scenario that does not rely on such slicing.

#emph[High-dimensional case with Gaussian assumption]

For sets of high-dimensional vectors, which are frequently encountered with #abbr.a[DNN] features, the computation of quantile functions becomes computationally infeasible. Nevertheless, as proposed by @heusel_fid_2017 in the context of image generation, a simplifying assumption can be made: that the embedding distributions can be approximated by multivariate Gaussians. This is considered a reasonable assumption for embeddings that have been mapped into a well-behaved latent space @heusel_fid_2017. This approximation enables the calculation of the 2-Wasserstein distance in a closed form, requiring only the mean vectors and covariance matrices of the distributions.

In this case, let the real and synthetic embedding distributions be modeled by multivariate Gaussians $N(mu, Sigma)$ and $N(tilde(mu), tilde(Sigma))$, respectively. The squared 2-Wasserstein distance, also recognized as the Fréchet distance @frechet_1925, between these two Gaussian distributions is given by @dowson_frechet_1982:

$ W_2^2(Q, tilde(Q)) = ||mu - tilde(mu)||_2^2 + text("Tr")(Sigma + tilde(Sigma) - 2(Sigma tilde(Sigma))^(1/2)) $

where:
- $mu$ and $tilde(mu)$ represent the mean vectors of the real and synthetic embeddings.
- $Sigma$ and $tilde(Sigma)$ denote the covariance matrices of the real and synthetic embeddings.
- $text("Tr")(dot)$ refers to the trace of a matrix.
- $(Sigma tilde(Sigma))^(1/2)$ signifies the matrix square root of the product of the covariance matrices.

The initial term, $|| mu - tilde(mu) ||_2^2$, quantifies the dissimilarity between the central tendencies of the two distributions. The second term, conversely, measures the differences in their spread and orientation, encapsulated by their covariance structures. In practical applications, the sample mean and covariance are estimated from a substantial number of real and synthetic embeddings, respectively, and the distance is then computed using the aforementioned formula. This metric forms the #abbr.a[FID] @heusel_fid_2017 in the field of image generation, and the identical principle can be extended to audio embeddings to derive a Fréchet Audio Distance @kilgour_fad_2019.

=== Perceptually-motivated factorized evaluation

As elaborated in @08_distances, various objective methodologies exist for evaluating the alignment of synthetic speech with its real counterpart.
The task of synthetic speech generation inherently lacks a singular ground truth, given the one-to-many nature of the task. Instead, #abbr.a[TTS] evaluation is framed as a problem of distributional similarity. Here, $D$ denotes an audio dataset and $cal(R)(S)$ represents a feature extracted from it. The objective is to quantify the fidelity of synthetic speech in mirroring real speech by deriving correlates for each perceptual factor and assessing their distance from both genuine speech datasets and various noise dataset distributions.

The initial iteration of this metric, TTSDS2 version 1.0 (hereafter referred to as TTSDS1), evaluates the quality of synthetic speech across multiple perceptual factors and ascertains its correlation with human judgments over a substantial period, spanning from legacy systems of 2008 to more contemporary ones from 2024. This foundational version benchmarks 35 #abbr.a[TTS] systems, demonstrating that a score computed as an unweighted average of various factors exhibits a strong correlation with human evaluations from each distinct time period. TTSDS1 delineates five primary factors, each assessed through specific features, to offer a comprehensive evaluation:

- #smallcaps[Generic]: This factor assesses overall distributional similarity, typically achieved by utilizing #abbr.a[SSL] representations of speech, such as those derived from HuBERT (base) @hsu_hubert_2021  and wav2vec 2.0 (base) models @baevski_wav2vec_2020. These representations are extracted from the intermediate layers of the neural networks, which are thought to be the most generalist @pasad_layer-wise_2021.
- #smallcaps[Ambient]: This factor quantifies the presence of noise or distortion within the speech signal. It leverages two one-dimensional correlates of noise: VoiceFixer @liu_voicefixer_2021 is employed to mitigate noise, followed by the application of #abbr.l[PESQ] @rix_pesq_2001 to measure the perceived quality between the noise-enhanced and original samples. Additionally, WADA SNR @kim_wada_2008 is used to estimate the signal-to-noise ratio of each individual sample.
- #smallcaps[Intelligibility]: This factor measures the ease with which the lexical content of the speech can be recognized. This is achieved by computing the #abbr.l[WER] from reference transcripts and automated transcripts generated by a wav2vec 2.0 model @baevski_wav2vec_2020 fine-tuned on 960 hours of LibriSpeech @panayotov_librispeech_2015, and additionally by a Whisper (small) model @radford_robust_2023.
- #smallcaps[Prosody]: This factor evaluates the realism of speech #smallcaps[Prosody]. It employs frame-level representations derived from our self-supervised prosody model and frame-level pitch features extracted using the WORLD vocoder @morise_world_2016. Furthermore, a proxy for segmental durations is obtained by utilizing HuBERT tokens and measuring their lengths, which corresponds to the number of consecutive occurrences of the same token.
- #smallcaps[Speaker]: This factor quantifies the degree of similarity between the synthetic speaker's voice and that of a real speaker. This is achieved by employing representations obtained from speaker verification systems, specifically d-vectors @wan_generalized_2018 and the more contemporary WeSpeaker @wang_wespeaker_2023 representations.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left), // Left-align content in both columns
    [#strong[Factor]], [#strong[Feature]], // Table headers, bolded as in the original LaTeX
    toprule(),
    // Environment Factor
    [#strong[Environment]], [#strong[Noise/Artifacts]],
    [], [VoiceFixer #cite(<liu_voicefixer_2021>) + PESQ #cite(<rix_pesq_2001>)],
    [], [WADA SNR #cite(<kim_wada_2008>)],
    // Speaker Factor
    toprule(),
    [#strong[Speaker]], [#strong[Speaker Embedding]],
    [], [d-vector #cite(<wan_generalized_2018>)],
    [], [WeSpeaker #cite(<wang_wespeaker_2023>)],
    // Prosody Factor
    toprule(),
    [#strong[Prosody]], [#strong[Segmental Length]],
    [], [Hubert #cite(<hsu_hubert_2021>) token length],
    [], [#strong[Pitch]],
    [], [WORLD #cite(<morise_world_2016>)],
    [], [#strong[SSL Representations]],
    [], [MPM (Ours)],
    // Intelligibility Factor
    toprule(),
    [#strong[Intelligibility]], [#strong[ASR WER]],
    [], [wav2vec 2.0 #cite(<baevski_wav2vec_2020>)],
    [], [Whisper #cite(<radford_robust_2023>)],
    // General Factor
    toprule(),
    [#strong[General]], [#strong[SSL Representations]],
    [], [Hubert #cite(<hsu_hubert_2021>)],
    [], [wav2vec 2.0 #cite(<baevski_wav2vec_2020>)],
  ),
  caption: [Features used in the benchmark and their respective factors. The overall TTSDS1 score is computed as an average of individual factor scores.],
  placement: top,
) <fig_ttsds1_features>

The specific features integrated into TTSDS1 for each factor are visually represented and detailed in @fig_ttsds1_features.

The scoring mechanism for each feature within TTSDS1 is derived by comparing its empirical distribution in synthetic speech $hat(P)(cal(R)(tilde(S))|tilde(D))$ to the distributions obtained from real speech datasets $hat(P)(cal(R)(S)|D)$ and noise datasets $hat(P)(cal(R)(S)|D^"NOISE"_i)$, drawn from a set of differing noise datasets $cal(D)^"NOISE"$, leveraging the 2-Wasserstein distance. The normalized similarity score for a given feature $cal(R)(S)$ is mathematically defined as follows:
$ "TTSDS"(D,tilde(D),D^"NOISE") = 100 times (min[W_2(tilde(D),D^"NOISE"_i)]_(i=0)^N) / (W_2(tilde(D),D) + min[W_2(tilde(D),D^"NOISE"_i)]_(i=0)^N $
In this formulation, $(min[W_2(tilde(D),D^"NOISE"_i)]_(i=0)^N)$ represents the minimum 2-Wasserstein distance between the synthetic speech and a designated set of distractor noise datasets, while $W_2(tilde(D),D)$ denotes the 2-Wasserstein distance to the distribution of real speech. An example of this scoring method, specifically for the one-dimensional pitch feature, is provided in @fig_f0, where the equation yields scores ranging from 0 to 100. Values exceeding 50 signify a stronger similarity to genuine speech than to noise. The final TTSDS1 score is computed as the unweighted arithmetic mean of the individual factor scores, with each factor score itself being the mean of its belonging feature scores.

#figure(
  image("../figures/9/pitch_distributions.png", width: 100%),
  placement: top,
  caption: [Distribution of $F_0$ in TTSDS for ground-truth, synthetic, and noise datasets. The distance between the synthetic and real distributions ($d_"gt"$) and the distance to noise ($d_"n"$) are shown, as well as how the overall score is computed.],
) <fig_f0>

#figure(
  image("../figures/9/heatmaps_ttsds1.png", width: 100%),
  placement: top,
  caption: [Development of factor score correlation coefficients over time from early speech synthesis (Blizzard'08) to the latest systems (TTS Arena).],
) <fig_ttsds1_factor_correlation>

The correlations observed between TTSDS1 factor scores and subjective evaluation metrics across different temporal stages are depicted in @fig_ttsds1_factor_correlation, illustrating the increasing importance of factors such as #smallcaps[Prosody] over time as TTS systems gain in perceived naturalness.

Building upon the framework established by TTSDS1, we introduce an augmented and more robust iteration, designated as TTSDS version 2.0 (subsequently referred to as TTSDS2). This advancement is aimed at ensuring its applicability across a wider range of domains and languages. While retaining the fundamental principle of factorized distributional similarity, specific feature implementations were refined: Within TTSDS2, the #smallcaps[Intelligibility] factor transitioned from relying on direct WER values to utilizing the final-layer activations of #abbr.a[ASR] models. This modification was implemented to enhance robustness, as prior #abbr.a[WER] features occasionally yielded anomalously low scores for authentic data across various domains. Regarding #smallcaps[Prosody], TTSDS2 now computes the utterance-level speaking rate by dividing the count of deduplicated HuBERT tokens within an utterance by the corresponding number of frames, thereby replacing earlier token length features that sometimes resulted in diminished scores for real speech. The #smallcaps[Generic] factor in TTSDS2 is also augmented by incorporating WavLM features @chen_wavlm_2022, alongside the existing HuBERT and wav2vec 2.0 features, to foster increased diversity in representation. For enhanced multilingual compatibility, HuBERT was substituted with mHuBERT-147 @boito_mhubert-147_2024 and wav2vec 2.0 with its cross-lingual counterpart, XLSR-53 @conneau_xlsr_2021. The aggregate set of features utilized in TTSDS2 is summarized in @ttsds2_features.

#import "../abbr.typ" // Assuming this import is available
#import "../quote.typ": * // Assuming this import is available

#figure(
  table(
    columns: (auto, 0.6fr), // First column adjusts to content, second column takes 60% of remaining space
    stroke: (top: 1.5pt + black, bottom: 1.5pt + black), // Thicker top and bottom rules
    align: (left, left), // Left-align content in both columns

    // Table header row
    [#strong[Factor]], [#strong[Features]],

    // Generic Factor content
    [#smallcaps[Generic]], [
      WavLM #cite(<chen_wavlm_2022>) layer 11 activations (actv.) \
      HuBERT #cite(<hsu_hubert_2021>) (base) layer 7 actv. \
      wav2vec 2.0 #cite(<baevski_wav2vec_2020>) (base) layer 8 actv.
    ],

    // Speaker Factor content
    [#smallcaps[Speaker]], [
      d‑Vector embeddings #cite(<wan_generalized_2018>) \
      WeSpeaker embeddings #cite(<wang_wespeaker_2023>)
    ],

    // Prosody Factor content
    [#smallcaps[Prosody]], [
      Mean F0 (PyWORLD) #cite(<morise_world_2016>) \
      HuBERT #cite(<hsu_hubert_2021>) speaking‑rate \
      Allosaurus #cite(<li_allosaurus_2020>) speaking-rate \
      MPM (Ours)
    ],

    // Intelligibility Factor content
    [#smallcaps[Intelligibility]], [
      wav2vec 2.0 #cite(<baevski_wav2vec_2020>) ASR head actv. \
      Whisper #cite(<radford_robust_2023>) (small) ASR head actv.
    ],

    // General Factor content
    [#smallcaps[General]], [
      SSL Representations \
      HuBERT #cite(<hsu_hubert_2021>) \
      wav2vec 2.0 #cite(<baevski_wav2vec_2020>)
    ],
  ),
  caption: [Revised feature set used for each TTSDS2 factor.],
) <ttsds2_features>

=== Datasets and Systems for TTSDS1 and TTSDS2

Here, we detail the specific datasets and #abbr.a[TTS] systems employed for the evaluation of both TTSDS1 and TTSDS2, along with their observed correlations with human subjective ratings.

==== TTSDS1 Datasets
For the evaluation of TTSDS1, datasets were meticulously chosen to encompass a broad spectrum of #abbr.a[TTS] systems, ranging from established legacy technologies to cutting-edge state-of-the-art architectures. The subjective measures utilized for comparison were derived from three primary sources:
- The Blizzard Challenge 2008 @king_blizzard_2008: This challenge provided #abbr.a[MOS] ratings for 22 distinct #abbr.a[TTS] systems across a variety of synthesis tasks. For the purposes of TTSDS1 evaluation, the "Voice A" audiobook task, which encompasses 15 hours of audio data, was specifically utilized.
- The "Back to the Future" (BTTF) dataset: This dataset facilitates comparisons between unit selection, hybrid, and statistical parametric #abbr.a[HMM]-based systems from the Blizzard Challenge 2013 @le_maguer_back_2022 and more advanced deep learning systems inspired by the Blizzard Challenge 2021 @ling_blizzard_2021, which included architectures such as FastPitch @lancucki_fastpitch_2021 and Tacotron @wang_tacotron_2017.
- The #abbr.a[TTS] Arena leaderboard @ttsarena: This publicly accessible resource offers crowdsourced A/B preference tests for contemporary #abbr.a[TTS] systems, predominantly those leveraging discrete speech representations generated by large language model-like systems. Only systems released between 2023 and 2024 were incorporated into this evaluation.

The reference speech datasets employed to compute the distributional distances in TTSDS1 include LibriTTS @zen_libritts_2019, LibriTTS-R @koizumi_libritts-r_2023, LJSpeech @ito_lj_2017, VCTK @vctk, and the training sets provided by the various Blizzard challenges @king_blizzard_2008@ling_blizzard_2021@le_maguer_back_2022. From each of these datasets, 100 utterances were randomly sampled, preferentially from their respective test splits if available. For the generation of distractor noise datasets, TTSDS1 utilized the ESC dataset @piczak_esc_2015 of background noises, in conjunction with synthetic noise types, including random uniform noise, random normal noise, and silent (all zeros and all ones) samples.

==== TTSDS2 Datasets
TTSDS2 expands the scope of evaluation to ensure robustness across four distinct domains within the English language, and further extends its applicability across 14 diverse languages. The English-language datasets comprised:
- #smallcaps[Clean]: This dataset served as the foundational baseline and consisted of samples drawn from the LibriTTS test split @zen_libritts_2019. It specifically features clean, read speech, having undergone filtration based on #abbr.a[SNR]. Utterances selected for this dataset ranged in duration from 3 to 30 seconds and originated from a single speaker.
- #smallcaps[Noisy]: This dataset was constructed by scraping LibriVox recordings from 2025, deliberately excluding #abbr.a[SNR] filtering during acquisition. This methodology was designed to rigorously assess how evaluation metrics are influenced by the presence of ambient noise within recordings.
- #smallcaps[Wild]: This dataset was compiled from recently uploaded YouTube videos in 2025. Utterances from these videos were extracted and subsequently diarized using the Whisper system @clapa_WhisperSpeech_2024. The data collection strategy for this dataset was specifically tailored to emphasize diverse speaking styles and varying recording conditions, drawing inspiration from the Emilia dataset @he_emilia_2024.
- #smallcaps[Kids]: This dataset constitutes a subset of the My Science Tutor Corpus @pradhan_my_2024, characterized by children's conversational speech interacting with a virtual tutor in an educational setting. This dataset served to evaluate the generalization capabilities of the developed evaluation metrics to less conventional #abbr.a[TTS] domains.
For all evaluated #abbr.a[TTS] systems, a cohort of 100 distinct speakers was randomly selected, with each speaker contributing two utterances. The acquired data underwent a manual filtering process to exclude content that was difficult to transcribe, potentially controversial, or offensive, ultimately yielding a refined set of 60 speakers per dataset. The first utterance from each selected speaker functioned as the reference for the #abbr.a[TTS] system, while the transcript corresponding to the second, entirely distinct utterance was employed as the input for text-to-speech synthesis.

==== TTS Systems Evaluated
For TTSDS2, a selection of 20 open-source, open-weight #abbr.a[TTS] systems released between 2022 and 2024 were chosen for evaluation. These systems collectively spanned 14 languages and were selected based on their capability to utilize a speaker reference and a transcript to govern their output. Noteworthy exceptions included MetaVoice @sharma_metavoice_2024 and GPT-SOVITS @RVCBoss_gptsovits_2024, which were excluded due to specific constraints related to reference audio length requirements. The most recent available checkpoints for each system, preceding January 1, 2025, were utilized, with the exception of XTTSv2 @casanova_xtts_2024, for which challenges in grapheme-to-phoneme conversion necessitated a downgrade to XTTSv1.

// Helper for superscript citations, mimicking the LaTeX \supercite command
#let supercite(ref_key) = {
  linebreak()
  text(0.7em, h(0.1em) + [*#cite(ref_key)*])
}

#let supercite2(ref_key1,ref_key2) = {
  linebreak()
  text(0.7em, h(0.1em) + [*#cite(ref_key1)#cite(ref_key2)*])
}

// Convenience macros for symbols in text mode as defined in the original LaTeX
#let OK = sym.checkmark
#let parityeq = "~" // near‑parity
#let paritygt = ">"     // better than GT
#let paritylt = "<"     // worse than GT

#context[
  #set text(size: 9pt)
#figure(
  placement: top,
  table(
    columns: (auto, .5fr, .7fr, .7fr, .7fr, 2fr), // Define 6 columns. Last column (ID) takes remaining space.
    
    // Top rule from booktabs
    toprule(),
    
    // Header row
    [#strong[System]], [#strong[Year]], [#strong[Obj.]], [#strong[Subj.]], [#strong[Parity]], [#strong[ID on `replicate.com`]],
    
    // Mid rule after header
    midrule(),

    // Content rows
    [Bark#supercite2(<bark_2023>,<BarkVC_2023>)], [2023], [], [], [], [#link("https://replicate.com/ttsds/e2")[`ttsds/e2`]],
    [†\*E2‑TTS#supercite2(<eskimez_e2_2024>,<chen_f5_2024>)], [2024], [#OK], [#OK], [#parityeq], [#link("https://replicate.com/ttsds/bark")[`ttsds/bark`]],
    [†F5‑TTS#supercite(<chen_f5_2024>)], [2024], [#OK], [#OK], [#paritygt], [#link("https://replicate.com/ttsds/f5")[`ttsds/f5`]],
    [FishSpeech 1.5#supercite(<liao_fishspeech_2024>)], [2024], [], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/fishspeech_1_5")[`ttsds/fishspeech_1_5`]],
    [GPT‑SoVITS~v2#supercite(<RVCBoss_gptsovits_2024>)], [2024], [], [], [], [#link("https://replicate.com/ttsds/gptsovits_2")[`ttsds/gptsovits_2`]],
    [†HierSpeech++~1.1#supercite(<lee_hierspeechpp_2023>)], [2023], [#OK], [#OK], [#parityeq], [#link("https://replicate.com/ttsds/hierspeechpp_1_1")[`ttsds/hierspeechpp_1_1`]],
    [†MaskGCT#supercite2(<wang_maskgct_2024>,<zhang_amphion_2024>)], [2024], [#OK], [#OK], [#paritygt], [#link("https://replicate.com/ttsds/maskgct")[`ttsds/maskgct`]],
    [MetaVoice‑1B#supercite(<sharma_metavoice_2024>)], [2024], [], [], [], [#link("https://replicate.com/ttsds/metavoice")[`ttsds/metavoice`]],
    [†\*NaturalSpeech~2#supercite2(<shen_natural_2018>,<zhang_amphion_2024>)], [2023], [#OK], [#OK], [#parityeq], [#link("https://replicate.com/ttsds/amphion_naturalspeech2")[`ttsds/amphion_naturalspeech2`]],
    [OpenVoice~v2#supercite(<qin_openvoice_2023>)], [2024], [], [], [], [#link("https://replicate.com/ttsds/openvoice2")[`ttsds/openvoice2`]],
    [†\*ParlerTTS~Large~1.0#supercite2(<lyth_parler_2024>,<lacombe_parlertts_2024>)], [2024], [#OK], [#OK], [#paritygt], [#link("https://replicate.com/ttsds/parlertts_large_1_0")[`ttsds/parlertts_large_1_0`]],
    [†Pheme#supercite(<budzianowski_pheme_2024>)], [2024], [#OK], [], [], [#link("https://replicate.com/ttsds/pheme")[`ttsds/pheme`]],
    [†SpeechT5#supercite(<ao_speecht5_2022>)], [2022], [], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/speecht5")[`ttsds/speecht5`]],
    [†StyleTTS~2#supercite(<li_styletts_2023>)], [2023], [#OK], [#OK], [#paritygt], [#link("https://replicate.com/ttsds/styletts2")[`ttsds/styletts2`]],
    [TorToiSe#supercite(<betker_tortoise_2023>)], [2022], [], [], [], [#link("https://replicate.com/ttsds/tortoise")[`ttsds/tortoise`]],
    [†\*VALL‑E#supercite2(<wang_valle_2023>,<zhang_amphion_2024>)], [2024], [#OK], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/amphion_valle")[`ttsds/amphion_valle`]],
    [†Vevo#supercite(<zhang_vevo_2025>)], [2024], [#OK], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/amphion_vevo")[`ttsds/amphion_vevo`]],
    [†VoiceCraft‑830M#supercite(<peng_voicecraft_2024>)], [2024], [#OK], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/voicecraft")[`ttsds/voicecraft`]],
    [WhisperSpeech~Medium#supercite(<clapa_WhisperSpeech_2024>)], [2024], [], [], [], [#link("https://replicate.com/ttsds/whisperspeech")[`ttsds/whisperspeech`]],
    [†XTTS‑v1#supercite(<casanova_xtts_2024>)], [2023], [#OK], [], [], [#link("https://replicate.com/ttsds/xtts_1")[`ttsds/xtts_1`]],
    
    // Bottom rule from booktabs
    bottomrule(),
  ),
  caption: [
    Open-source #abbr.a[TTS] systems, prior evaluation, and results for each system relative to ground‑truth (GT) speech:
    $dagger$ = accompanied by publication;
    $ast$ = third‑party implementation; \
    Parity column: Reported #abbr.a[MOS]/#abbr.a[CMOS] are close to GT (#parityeq), surpassing GT (#paritygt) or below GT (#paritylt).
  ],
) <tab_ttsds2_systems>
]

A comprehensive enumeration of the #abbr.a[TTS] systems evaluated for TTSDS2, along with their respective publication years, objective and subjective evaluation statuses, reported parity with real speech, and Replicate.com identifiers, is presented in @tab_ttsds2_systems. Of the 20 systems included, 13 were accompanied by peer-reviewed research papers, with 10 of these reporting both subjective and objective evaluation results. Notably, seven of these systems claimed #abbr.a[MOS] or #abbr.a[CMOS] scores within 0.05 of the ground-truth values or even surpassing them. Regarding objective evaluation methods, #abbr.l[WER] and #smallcaps[Speaker] Similarity were reported in five instances, UTMOS, #abbr.l[CER], Fréchet-type distances, and #abbr.l[MCD] in two instances each, and #abbr.l[PESQ] and #abbr.l[STOI] were reported only once.

// Define colors as per LaTeX original
#let negstrong = rgb("#F3C2C1") // –1 … –0.5
#let negweak = rgb("#F3E3E4")  // –0.5 … 0
#let posweak = rgb("#E2EAD5")  //  0 … 0.5
#let posstrong = rgb("#B8D78F") //  0.5 … 1

// Define a function for colored cells, with optional bolding
#let scorecell(value_str, bold: false) = {
  let value = float(value_str)
  let content = if bold { strong(value_str) } else { value_str }
  let color
  if value < -0.5 { color = negstrong }
  else if value < 0.0 { color = negweak }
  else if value < 0.5 { color = posweak }
  else { color = posstrong }
  table.cell(fill: color, content) // Added padding for visual spacing
}

#context[
  #set text(size: 9pt)
  #figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),

    // Top rule from booktabs
    toprule(),
    
    // Header row with multirow and multicolumn
      table.cell(
      colspan: 1, rowspan: 2, [\ #strong[Metric]]),
      table.vline(),
      // Multirow for Metric
      table.cell(colspan: 3, [#strong[Clean]]), // Multicolumn for Clean
      table.cell(colspan: 3, [#strong[Noisy]]), // Multicolumn for Noisy
      table.cell(colspan: 3, [#strong[Wild]]), // Multicolumn for Wild
      table.cell(colspan: 3, [#strong[Kids]]), // Multicolumn for Kids
    // ),
    // table.crule(start: 2, end: 4), // Trimmed c_midrule for Clean
    // crule(start: 5, end: 7), // Trimmed c_midrule for Noisy
    // crule(start: 8, end: 10), // Trimmed c_midrule for Wild
    // crule(start: 11, end: 13), // Trimmed c_midrule for Kids
    table.hline(start: 1, end: 13),
    
    // Second header row (sub-headers)
    [#smallcaps[mos]], [#smallcaps[cmos]], [#smallcaps[smos]], [#smallcaps[mos]], [#smallcaps[cmos]], [#smallcaps[smos]], [#smallcaps[mos]], [#smallcaps[cmos]], [#smallcaps[smos]], [#smallcaps[mos]], [#smallcaps[cmos]], [#smallcaps[smos]],
    
    // Mid rule after headers
    midrule(),

    // Content rows
    [TTSDS2 (Ours)], scorecell("0.75", bold: true), scorecell("0.69", bold: true), scorecell("0.73", bold: true), scorecell("0.59"), scorecell("0.54"), scorecell("0.71"), scorecell("0.75"), scorecell("0.71"), scorecell("0.75"), scorecell("0.61"), scorecell("0.50"), scorecell("0.70"),
    [TTSDS1 (Ours)], scorecell("0.60"), scorecell("0.62"), scorecell("0.52"), scorecell("0.49"), scorecell("0.61"), scorecell("0.66"), scorecell("0.67"), scorecell("0.57"), scorecell("0.67"), scorecell("0.70"), scorecell("0.52"), scorecell("0.60"),
    midrule(),
    [X‑Vector], scorecell("0.46"), scorecell("0.42"), scorecell("0.56"), scorecell("0.40"), scorecell("0.29"), scorecell("0.77"), scorecell("0.82"), scorecell("0.82", bold: true), scorecell("0.62"), scorecell("0.70"), scorecell("0.57"), scorecell("0.75", bold: true),
    [RawNet3], scorecell("0.36"), scorecell("0.26"), scorecell("0.52"), scorecell("0.44"), scorecell("0.37"), scorecell("0.82", bold: true), scorecell("0.85", bold: true), scorecell("0.80"), scorecell("0.64"), scorecell("0.73", bold: true), scorecell("0.61", bold: true), scorecell("0.77", bold: true),
    [SQUIM], scorecell("0.68"), scorecell("0.46"), scorecell("0.37"), scorecell("0.48"), scorecell("0.48"), scorecell("0.60"), scorecell("0.62"), scorecell("0.75"), scorecell("0.79", bold: true), scorecell("0.57"), scorecell("0.55"), scorecell("0.45"),
    [ECAPA‑TDNN], scorecell("0.36"), scorecell("0.29"), scorecell("0.47"), scorecell("0.29"), scorecell("0.22"), scorecell("0.72"), scorecell("0.81"), scorecell("0.78"), scorecell("0.58"), scorecell("0.69"), scorecell("0.60"), scorecell("0.72"),
    [DNSMOS], scorecell("0.41"), scorecell("0.37"), scorecell("0.22"), scorecell("0.57"), scorecell("0.36"), scorecell("0.22"), scorecell("0.35"), scorecell("0.28"), scorecell("0.03"), scorecell("0.31"), scorecell("0.10"), scorecell("0.28"),
    [AE-CE], scorecell("0.60"), scorecell("0.46"), scorecell("0.32"), scorecell("0.58"), scorecell("0.53"), scorecell("0.21"), scorecell("0.19"), scorecell("0.10"), scorecell("0.11"), scorecell("-0.02"), scorecell("-0.12"), scorecell("-0.10"),
    [AE-CU], scorecell("0.49"), scorecell("0.37"), scorecell("0.30"), scorecell("0.60", bold: true), scorecell("0.58", bold: true), scorecell("0.13"), scorecell("0.35"), scorecell("0.24"), scorecell("0.22"), scorecell("-0.09"), scorecell("-0.21"), scorecell("-0.13"),
    [AE-PQ], scorecell("0.49"), scorecell("0.33"), scorecell("0.21"), scorecell("0.55"), scorecell("0.48"), scorecell("0.04"), scorecell("0.21"), scorecell("0.16"), scorecell("0.12"), scorecell("0.03"), scorecell("-0.08"), scorecell("-0.05"),
    [UTMOSv2], scorecell("0.39"), scorecell("0.25"), scorecell("0.09"), scorecell("0.34"), scorecell("0.36"), scorecell("0.19"), scorecell("0.16"), scorecell("0.14"), scorecell("-0.04"), scorecell("0.05"), scorecell("0.03"), scorecell("-0.02"),
    [FAD (CLAP)], scorecell("-0.22"), scorecell("0.06"), scorecell("-0.01"), scorecell("0.45"), scorecell("0.30"), scorecell("0.16"), scorecell("-0.03"), scorecell("0.08"), scorecell("0.25"), scorecell("0.12"), scorecell("0.26"), scorecell("0.04"),
    [UTMOS], scorecell("0.51", bold: true), scorecell("0.30"), scorecell("0.31"), scorecell("0.47"), scorecell("0.29"), scorecell("0.00"), scorecell("-0.12"), scorecell("-0.12"), scorecell("-0.26"), scorecell("-0.02"), scorecell("-0.18"), scorecell("-0.04"),
    [STOI], scorecell("-0.11"), scorecell("0.01"), scorecell("0.02"), scorecell("-0.06"), scorecell("0.00"), scorecell("0.19"), scorecell("0.07"), scorecell("0.41"), scorecell("0.24"), scorecell("-0.32"), scorecell("-0.08"), scorecell("0.05"),
    [PESQ], scorecell("0.01"), scorecell("-0.16"), scorecell("0.27"), scorecell("-0.34"), scorecell("0.00"), scorecell("0.07"), scorecell("-0.14"), scorecell("0.01"), scorecell("-0.06"), scorecell("-0.08"), scorecell("-0.04"), scorecell("-0.38"),
    [NISQA], scorecell("0.05"), scorecell("0.00"), scorecell("0.06"), scorecell("0.05"), scorecell("-0.21"), scorecell("-0.53"), scorecell("-0.32"), scorecell("-0.33"), scorecell("-0.64"), scorecell("-0.29"), scorecell("-0.27"), scorecell("-0.46"),
    [MCD], scorecell("-0.46"), scorecell("-0.37"), scorecell("-0.27"), scorecell("-0.45"), scorecell("-0.58"), scorecell("-0.74"), scorecell("-0.33"), scorecell("-0.45"), scorecell("-0.51"), scorecell("-0.31"), scorecell("-0.13"), scorecell("-0.38"),
    [WER], scorecell("-0.19"), scorecell("-0.18"), scorecell("-0.17"), scorecell("-0.11"), scorecell("-0.30"), scorecell("-0.13"), scorecell("-0.28"), scorecell("-0.17"), scorecell("-0.22"), scorecell("-0.45"), scorecell("-0.26"), scorecell("-0.39"),

    // Bottom rule from booktabs
    bottomrule(),
  ),
  caption: [
    Spearman rank correlations. Colours: #box(fill: negstrong, rect(width: 7pt, height: 7pt)) –1 … –0.5, #box(fill: negweak, rect(width: 7pt, height: 7pt)) –0.5 … 0, #box(fill: posweak, rect(width: 7pt, height: 7pt)) 0 … 0.5, #box(fill: posstrong, rect(width: 7pt, height: 7pt)) 0.5 … 1.
  ],
) <fig_ttsds2_spearman_correlation>]


#emph[*Correlations:*]
For each of the 20 evaluated #abbr.a[TTS] systems, human ratings for #abbr.a[MOS], #abbr.a[CMOS], and #abbr.a[SMOS] were aggregated and averaged across the #smallcaps[Clean], #smallcaps[Noisy], #smallcaps[Wild], and #smallcaps[Kids] datasets, establishing a set of gold standard ratings. Spearman rank correlation coefficients were then computed between these human judgments and the aforementioned objective metrics, with emphasis on their ability to accurately rank the performance of the systems.
As presented in @fig_ttsds2_spearman_correlation, TTSDS2 demonstrated the most robust correlation across all evaluated datasets, achieving an average correlation coefficient of 0.67. This performance represented a 10% relative improvement over the original TTSDS1. All computed correlations for both TTSDS1 and TTSDS2 were found to be statistically significant (p < 0.05). #smallcaps[Speaker] Similarity metrics, specifically RawNet3 and X-Vector, ranked second in performance with average correlations of 0.6. Among the #abbr.a[MOS] Prediction networks, only SQUIM-MOS exhibited robust performance, with an average correlation of 0.57. Following the ECAPA-TDNN metric, other objective metrics showed substantially lower average correlations, generally falling below 0.3. It was observed that many metrics, including Audiobox Aesthetics and UTMOSv2, performed well on #smallcaps[Noisy] and #smallcaps[Clean] audiobook speech. However, their performance notably deteriorated on #smallcaps[Kids] data, which is an expected outcome given that children's speech is perceptually and acoustically further removed from the typical domains on which most #abbr.a[TTS] systems are trained.

#figure(
  image("../figures/9/scatterplot_ttsds_combined.png", width: 100%),
  placement: top,
  caption: [Correlation of three representative objective metrics with human MOS across the four datasets.  
  Each colour/marker denotes a domain. Solid line = overall least-squares fit; dashed/dotted lines = domain-specific fits; each with corresponding Pearson $r$.],
) <ttsds_correlations_plot>

Additionally, as illustrated in @ttsds_correlations_plot, TTSDS2 generally exhibits a continuous scale of evaluation, whereas both SQUIM-MOS and X-Vector #smallcaps[Speaker] Similarity display a tendency towards some clustering behavior. The precise underlying causes for these observed behaviors are not certain, but they could potentially suggest instances of overfitting to particular #abbr.a[TTS] systems.

#figure(
  placement: top,
  table(
    columns: (auto, auto, auto, auto, auto), // First column adjusts to content, remaining 4 columns adjust automatically
    
    // Top rule from booktabs
    toprule(),
    
    // Header row
    [#strong[Dataset]], [#strong[Generic]], [#strong[Speaker]], [#strong[Prosody]], [#strong[Intelligibility]],
    
    // Mid rule after header
    midrule(),

    // Content rows
    [Clean], [0.42], [#strong[0.84]], [0.38], [0.47],
    [Noisy], [0.59], [#strong[0.86]], [0.46], [0.59],
    [Wild], [0.53], [#strong[0.59]], [0.34], [0.58],
    [Kids], [#strong[0.80]], [0.70], [0.60], [0.63],
    
    // Bottom rule from booktabs
    bottomrule(),
  ),
  caption: [Pearson correlation ($r$) between each factor and #abbr.a[MOS].],
) <fig_ttsds2_factor_correlation>

The Pearson correlation coefficients quantifying the relationship between the individual TTSDS2 factors and #abbr.a[MOS] ratings are presented in @fig_ttsds2_factor_correlation. For the #smallcaps[Clean] and #smallcaps[Noisy] datasets, the #smallcaps[Speaker] factor consistently demonstrates a dominant influence on the overall #abbr.a[MOS] score. Conversely, for the #smallcaps[Wild] and #smallcaps[Kids] datasets, the #smallcaps[Speaker] factor exhibits reduced impact, with the #smallcaps[Intelligibility] and #smallcaps[Generic] factors showing comparable correlations. The #smallcaps[Prosody] factor achieves its highest correlation for the #smallcaps[Kids] dataset, showing its utility in evaluating the prosodic patterns in children’s speech. Overall, while the #smallcaps[Speaker] factor proves to be the most generally beneficial across diverse contexts, the other factors are complementary, particularly for non-read speech categories such as #smallcaps[Wild] and #smallcaps[Kids]. All observed factor correlations were found to be statistically significant (p < 0.05).

=== Multilingual and Recurring Evaluation

Building upon the English-language evaluation, the TTSDS2 framework is extended to allow for multilingual applications, providing a publicly accessible benchmark encompassing 14 distinct languages. This benchmark is designed for frequent updates to prevent data leakage and to ensure representativeness across a wide range of recording conditions, speakers, and acoustic environments. To achieve these objectives, an automated data collection and processing pipeline is introduced.

#figure(
  image("../figures/9/neurips_ttsds.png", width: 100%),
  placement: top,
  caption: [*Overview of TTSDS2:* Both public (YouTube, LibriVox) and academic (Linguistic Data Consortium) sources were used for validating TTSDS2 as a metric, by showing robust correlations with listening test results across domains. A multilingual YouTube dataset is automatically scraped and synthesised quarterly, and with TTSDS2, provides ranking of TTS.],
) <fig_ttsds_pipeline>

#emph[Pipeline:]
The TTSDS2 pipeline, the source code of which is publicly available at github.com/ttsds/pipeline, is systematically employed to quarterly reconstruct the multilingual dataset, as schematically depicted in @fig_ttsds_pipeline. This comprehensive process encompasses several sequential stages:

#enum(
[
#emph[Data Scraping]: The initial stage involves the collection of 250 videos per language. To achieve this, ten distinct keywords are translated into each target language, and a narrowly focused YouTube search is conducted using the platform's API, specifically targeting content uploaded within the previous quarter. The search results are then ordered by view count to prioritize higher quality or authentic content, and only videos exceeding 20 minutes in duration are retained to filter out low-quality or synthetic material. Videos are then diarized using the Whisper system @radford_robust_2023, and FastText @joulin_fasttext_2016@bojanowski_fasttext_2016 is applied for language identification on the automatically generated transcripts to ensure that the content aligns with the specified target language.
],
[
#emph[Preprocessing]: In this stage, up to 16 utterances are extracted from the central portion of each video. Only utterances originating from a single speaker are retained, as identified by the preceding diarization process.
],
[
#emph[Filtering]: Extracted utterances undergo a filtering process to identify and exclude potentially offensive or controversial content, utilizing XNLI @conneau_xnli_2018 for entailment-based classification. Pyannote #smallcaps[Speaker] diarisation @bredin_pyannote_2023 is employed to detect instances of crosstalk within the audio. Concurrently, Demucs @rouard_demucs_2022 source separation is utilized to identify and remove any background music present in the recordings. From the remaining clean and filtered utterances, 50 speaker-matched pairs are meticulously selected for each language, which are then partitioned into distinct #smallcaps[Reference] and #smallcaps[Synthesis] sets.
],
[
#emph[Synthesis]: For all #abbr.a[TTS] systems integrated into the benchmark, the speaker identities extracted from the #smallcaps[Reference] set are used to synthesize speech with the textual content provided by the #smallcaps[Synthesis] set. This entire synthesis process is accessible and facilitated through replicate.com/ttsds.
],
[
#emph[TTSDS2]: The multilingual TTSDS2 metric is then applied to compute evaluation scores for each synthesized system. These computed scores are subsequently published on ttsdsbenchmark.com. This quarterly repetition of the entire pipeline, incorporating systems released in the previous quarter, serves as a crucial mechanism to prevent data contamination from newly released models. Future plans include expanding the number of evaluated systems and languages in subsequent evaluation rounds.
]
)

#emph[Multilingual validity of TTSDS:]
While the endeavor of collecting gold standard #abbr.a[MOS] labels for 14 languages falls outside the practical scope of this specific work, the applicability of TTSDS2 to the multilingual context was validated using Uriel+ @khan_uriel_2024. This resource provides typological distances for a set of 7970 languages. We posit that if TTSDS2 distances exhibited a correlation with linguistic distances established by linguists, this would be a strong indicator for the viability of application to a multi-lingual setting. Analysis conducted on each individual TTSDS2 factor reveales that when comparing ground-truth language datasets, the derived scores correlate with typological distances ($rho = -0.39$ for regular TTSDS2 and $rho = -0.51$ for multilingual TTSDS2, with both correlations being statistically significant at p < 0.05). The observed negative correlations are expected, as a higher TTSDS2 score signifies a smaller perceptual distance, and the stronger correlation by the multilingual TTSDS2 scores is an encouraging indication of its cross-linguistic efficacy.

#figure(
  image("../figures/9/ttsds_boxplot.png", width: 100%),
  placement: top,
  caption: [TTSDS2 scores across 14 languages.],
) <fig_language_scores>


The TTSDS2 scores obtained across 14 distinct languages are visually represented in @fig_language_scores, which graphically illustrates the metric's performance across a diverse spectrum of linguistic families. This comprehensive evaluation demonstrates the robust capability of TTSDS2 to quantify the quality of synthetic speech in a multilingual context.

This chapter has detailed the methodology employed for measuring distributional distances in synthetic speech, explaining the motivation, development and validation of #abbr.a[TTSDS] version 1.0 and its enhanced iteration, version 2.0. The robust correlations observed with human subjective evaluations across various domains and languages show its utility as a reliable objective metric for assessing synthetic speech quality. This evaluation framework, alongside the previously established TTS-for-ASR based evaluation, help quantify the gap between real and synthetic speech. Next, we discuss the overall conclusions drawn from this work and an outline its inherent limitations.