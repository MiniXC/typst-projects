#import "../abbr.typ"
#import "../quote.typ": *
#import "../math.typ": *
#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#let blob(pos, label, tint: white, width: 26mm, ..args) = node(
	pos, align(center, label),
	width: width,
	fill: tint.lighten(60%),
	stroke: 1pt + tint.darken(20%),
	corner-radius: 5pt,
	..args,
)
#import "../comic.typ"

== The Synthetic-Real Gap in ASR Training <05_ttsasr>

#q(
[#citep(<nikolenko_synthetic_2021>)],
[#emph[Synthetic Data for Deep Learning]],
[As soon as researchers needed to solve a real-world
computer vision problem with a neural network, synthetic data appeared.]
)

Using synthetic data for training has been a cornerstone of machine learning since its early days, offering a practical solution to the challenges of real-world data collection. Its first recorded use dates back to 1988, when it was employed to train a self-driving neural network. The motivation was efficiency: "changes in parameters such as camera orientation would require collecting an entirely new set of road images" @pomerleau_alvinn_1988. In today's deep learning era, the rationale for leveraging Text-to-Speech (TTS)-generated speech in Automatic Speech Recognition (ASR) remains similar—synthetic data can be generated with precise control over properties like speaker identity, lexical content, or even phonetic durations, often more efficiently than gathering equivalent real data @du_speaker_2020@casanova_singlespeaker_2022@rosenberg_speechaug_2019@fazel_synthasr_2021@rossenbach_duration_2023.

While synthetic data is frequently used to augment real datasets, the primary perspective in this work treats it as a proxy for real speech. Training an ASR model on TTS-generated data—TTS-for-ASR—serves as an objective lens to examine the distributional distance between real and synthetic speech. This approach probes how well synthetic speech captures the variability of real speech in a way that directly impacts a downstream task. If the distributions of real speech $S$ and synthetic speech $tilde(S)$ were truly identical, we would expect equivalent ASR performance when training on either. As we will show, this is not the case, and the resulting performance gap forms the core inquiry of this part of the thesis.

=== Augmenting Real Data

Synthetic data often serves to enhance existing real datasets, introducing targeted diversity that can improve ASR robustness. Before delving into specifics, it's worth noting why this augmentation is effective: real speech datasets, while authentic, may lack coverage in areas like rare vocabularies or speaking styles. TTS can fill these voids by generating tailored samples, but the key lies in balancing the mix to avoid diluting the realism of the original data.

A common strategy is to supplement a real dataset $S$ with synthetic speech $tilde(S)$, particularly for lexical adaptation. For instance, synthesizing speech from out-of-domain transcripts $T$ allows ASR models to adapt to specialized vocabularies, as explored in works adapting to medical terminology @fazel_synthasr_2021.

The ratio of real to synthetic data is critical. Studies consistently show that a 50:50 split provides a solid baseline, yielding reliable improvements @li_synthaug_2018@rosenberg_speechaug_2019@wang_improving_2020. However, this is flexible—when synthetic speech introduces high style diversity (e.g., via Variational Autoencoders), even a 9:91 real-to-synthetic ratio can deliver a 16% relative WER reduction @sun_generating_2020. Commonly used datasets for these experiments include read audiobooks like LibriSpeech @panayotov_librispeech_2015 and its TTS-optimized variant, LibriTTS @zen_libritts_2019. While effective, augmentation assumes some real data is available; the true test of synthetic speech's potential lies in standalone use.

=== Training on Synthetic Data Alone

Fully replacing real data with synthetic speech tests the core assumption of TTS parity: if synthetic distributions match real ones, ASR performance should be comparable. Before examining the evidence, consider the implications—a persistent gap would reveal systematic limitations in TTS, beyond surface-level naturalness, that hinder downstream tasks like ASR.

Our findings, echoed in prior work, show this parity is elusive. The Word Error Rate Ratio (WERR) between synthetic- and real-trained ASR models typically hovers around 2, meaning synthetic speech yields nearly double the errors of real speech on real test sets. This contrasts sharply with subjective evaluations; for example, Tacotron 2 achieved a Mean Opinion Score (MOS) ratio of ~1.02 to real speech @shen_natural_2018, yet its synthetic data performs far worse in ASR training. Such discrepancies suggest human listeners overlook distributional flaws (e.g., reduced prosodic variability) that ASR systems exploit. This gap motivates deeper investigation: What elements of real speech diversity are missing in synthesis?

#comic.comic((80mm, 40mm), "Comic showing WERR gap between real and synthetic training", red) <fig_werr_tts_asr>

=== What is Synthetic Speech Missing?

The observed performance gap stems from synthetic speech's narrower distribution, often lacking the prosodic, speaker, and environmental variability of real speech. Before exploring solutions, an overview of key methods reveals a progression from unsupervised latent capture to targeted control and post-hoc simulation.

A primary approach introduces variation through latent variables, learning styles independent of text. Global Style Tokens (GST) discretize styles via attention-weighted embeddings @wang_style_2018, while Variational Autoencoders (VAE) model continuous distributions for sampling novel variations @kingma_auto-encoding_2013, proving effective for TTS-for-ASR @sun_generating_2020.

More direct control comes from explicit conditioning on measurable attributes (e.g., pitch, energy), often via variance adapters in non-autoregressive models @ren_fastspeech_2019. Post-generation augmentation simulates real-world conditions by adding noise or reverberation @rossenbach_synthattention_2020. These complementary techniques—latent, explicit, and augmentative—aim to close the gap, but their efficacy requires objective evaluation, as detailed next.

=== Evaluating Diversity: WERR

To rigorously assess diversity enhancements, we use ASR training as a proxy, quantifying how well synthetic speech covers real distributions. WERR measures this gap: Train two ASR models on identical transcripts—one with real speech $S$, one with synthetic $tilde(S)$—and compute the ratio of their Word Error Rates (WER) on a real test set $(s, t)$:

$
"WERR"(tilde(S) parallel S) = "WER"[(s,t) parallel Theta(tilde(S),T)] / "WER"[(s,t) parallel Theta(S,T)]
$ <eq_werr>

Here, $Theta(S,T)$ denotes ASR weights trained on speech $S$ and transcripts $T$; seen/unseen splits ensure fair comparison. A WERR of 1 indicates synthetic equivalence; values >1 reveal gaps. While ASR language models (LMs) can influence WER, our hybrid HMM-TDNN setup minimizes this by emphasizing acoustic modeling with minimal LM interference.

#comic.comic((80mm, 40mm), "Comic explaining WERR calculation with real/synthetic splits", green) <fig_werr_calc>

=== WERR as Distance Metric

Beyond evaluation, WERR approximates distributional distance, but does it qualify as a formal metric? It satisfies non-negativity (ratios are positive) and identity (subtract 1 for self-distance=0). For symmetry, cross-training experiments (training on synthetic/evaluating on real, and vice versa) yield similar ratios (~3.66 vs. ~3.75), but stochastic training prevents guarantees.

To enforce symmetry, we define Mean Word Error Rate Ratio (MWERR) as the average of forward and reverse WERR:

$
"MWERR"(tilde(S) parallel S) = 1/2 times ("WERR"(tilde(S) parallel S) + "WERR"(S parallel tilde(S)))
$ <eq_mwerr>

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

However, MWERR lacks the triangle inequality, as WER emerges from non-linear optimization, not direct distribution comparison. Augmentation experiments further highlight limitations: Techniques like SpecAugment improve ASR but distort distributions away from real speech @park_specaugment_2019. Thus, MWERR is a task-specific heuristic for dissimilarity, not a true metric.

#comic.comic((80mm, 40mm), "Comic illustrating MWERR symmetry with real/synthetic swaps", blue) <fig_mwerr_symmetry>

Having quantified the gap via WERR, we now empirically test methods to enhance synthetic diversity in the next chapter.