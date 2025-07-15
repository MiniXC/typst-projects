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

== Synthetic Speech as a Proxy for Real Speech <05_ttsasr>

#q(
[#citep(<nikolenko_synthetic_2021>)],
[#emph[Synthetic Data for Deep Learning]],
[As soon as researchers needed to solve a real-world
computer vision problem with a neural network, synthetic data appeared.]
)

Using synthetic data for training has been used in machine learning since its early days. Its first recorded use for training a self-driving neural network in 1988 was motivated by the need for an efficient alternative to data collection, as "changes in parameters such as camera orientation would require collecting an entirely new set of road images" @pomerleau_alvinn_1988. The motivation behind using #abbr.a[TTS]-generated speech for #abbr.a[ASR] in today's very different deep-learning landscape is similar -- it can be more efficient to generate data with specific properties rather than to collect it. For #abbr.a[ASR], these properties could be speaker identity @du_speaker_2020@casanova_singlespeaker_2022, lexical content @rosenberg_speechaug_2019@fazel_synthasr_2021, or even the duration of phones within the speech signal @rossenbach_duration_2023.

While this approach is commonly used to augment existing real datasets, the main lens through which we view synthetic speech in this work is as a proxy for real speech. The task of training an #abbr.a[ASR] model, or #abbr.a[TTS]-for-#abbr.a[ASR], serves as an objective test for the distance between real and synthetic speech distributions. It allows us to probe the distribution gap between synthetic and real speech in a way that is directly relevant to a downstream task. If the distributions of $S$ and $Syn$
 were truly identical, we would expect them to yield equivalent performance when used as training data for an #abbr.a[ASR] system. As we will demonstrate, this is far from the case, and this discrepancy forms the central investigation of this part of our work.

=== Augmenting Real Data

The most common application of #abbr.a[TTS]-for-#abbr.a[ASR] is to supplement an existing real dataset $S$ with synthetic speech $Syn$. This is particularly effective for introducing lexical diversity. By synthesizing speech for out-of-domain transcripts $T$, an #abbr.a[ASR] model can be adapted to new vocabularies, a task explored extensively by works such as #citea(<fazel_synthasr_2021>) for adapting to the medical domain.

When augmenting an existing real dataset, a crucial consideration is the ratio of real to synthetic data. Several studies have found that a 50:50 split between $S$ and $Syn$ provides a robust and effective baseline, yielding consistent improvements @li_synthaug_2018@rosenberg_speechaug_2019@wang_improving_2020. This is not a fixed rule, however. When the synthetic speech offers significantly increased style diversity, for example through a #abbr.a[VAE], a much smaller proportion of real data can be effective. For instance, #citea(<sun_generating_2020>) demonstrated a 16% relative WER improvement with a split of only 9% real data to 91% synthetic data, showcasing that high-diversity synthetic data can heavily outweigh real data. The data most commonly used for these experiments is read audiobook speech, such as LibriSpeech @panayotov_librispeech_2015 and its #abbr.a[TTS]-specific derivative, LibriTTS @zen_libritts_2019.

=== Training on Synthetic Data Alone

The more fundamental test of the distribution gap, however, is to train an #abbr.a[ASR] model using only synthetic data. If modern #abbr.a[TTS] systems truly achieve human parity, as listener ratings might suggest @chen_vall-e_2024; @tan_naturalspeech_2024, then we would expect the performance of an #abbr.a[ASR] model trained on $Syn$ to be equivalent to one trained on $S$.

Our findings, along with those of others, consistently show this is not the case. As illustrated in @fig_werr_tts_asr, the #abbr.l[WERR], defined as the ratio between the #abbr.a[WER] from synthetic-only training and real-only training, consistently tends towards a value of approximately 2. This means that even state-of-the-art synthetic speech produces nearly twice the word error rate of its real counterpart. This observation is striking when contrasted with subjective scores. For example, Tacotron 2 @shen_natural_2018 achieved a #abbr.a[MOS] ratio of approximately 1.02 compared to real speech, yet its #abbr.a[WERR] in our experiments is substantially higher. This indicates that human listeners, in standard #abbr.a[MOS] tests, are not sensitive to the acoustic properties or distributional shortcomings that cause an #abbr.a[ASR] system to fail. This observation sets up much of the exploration in the remainder of this thesis: What is missing from synthetic speech that explains this persistent and significant performance gap?

#comic.comic((80mm, 40mm), "WERR explanation figure.", red) <fig_werr_tts_asr>

=== What is Synthetic Speech Missing?

To understand the source of the distribution gap, we investigate methods designed to improve the diversity of synthetic speech, focusing on factors we hypothesize are lacking.

A variety of techniques have been developed to generate speech that is not just natural-sounding, but also diverse enough for robust #abbr.a[ASR] training. A primary approach is to introduce and control sources of variation through latent variables. Models learn a latent space that is ideally independent of the text content and can be sampled from to control the style of the generated speech. The two most common methods for this are #abbr.pla[GST] @wang_style_2018, which learn a set of discrete style tokens from reference speech, and #abbr.pla[VAE] @kingma_auto-encoding_2013, which learn a continuous latent distribution. #abbr.pla[VAE] have become a common solution for improving controllability in #abbr.a[TTS]-for-#abbr.a[ASR] @casanova_yourtts_2022@sun_generating_2020.

A more direct form of control is achieved through explicit conditioning. Here, specific attributes are extracted from reference audio and used as additional inputs to the #abbr.a[TTS] model. This commonly includes conditioning on speaker representations like d-vectors to control speaker identity $E_"SPK"$ @du_speaker_2020@wang_improving_2020, as well as prosodic correlates like pitch $F_"F0"$, energy, and phoneme durations to control the speaking style @rossenbach_duration_2023.

Finally, the synthetic speech can be made more suitable for real-world #abbr.a[ASR] by applying post-generation data augmentation. This involves adding simulated background noise or acoustic reverberation to the clean synthetic output, which helps the #abbr.a[ASR] system become more robust to varied environmental conditions @rossenbach_synthattention_2020. While we find in #link(<part_01>, [Part II]) that these changes do improve #abbr.a[TTS]-for-#abbr.a[ASR] performance, conditioning and scaling only explain a small part of the gap, and our research indicates diminishing returns. This suggests something more fundamental is missing from the synthetic speech distribution, a question that motivates the development of our evaluation framework in #link(<part_02>, [Part III]).