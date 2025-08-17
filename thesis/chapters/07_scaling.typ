#import "../abbr.typ"
#import "../quote.typ": *
#import "../math.typ": *
#import "../moremath.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style
#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#import fletcher.shapes: house, hexagon
#let blob(pos, label, tint: white, width: auto, ..args) = node(
	pos, align(center, label),
	width: width,
	fill: tint.lighten(60%),
	stroke: 1pt + tint.darken(20%),
	corner-radius: 5pt,
	..args,
)

== Scaling properties for TTS-for-ASR <07_scaling>

#q(
  [#citep(<sutton_bitter_2019>)],
  [#emph[The Bitter Lesson]],
  [The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.]
)

The performance of deep learning models has been consistently observed to improve predictably with increasing computational resources and data. This phenomenon is formally described as neural scaling laws, which posit a power-law relationship between a model's test error and critical factors such as the number of model parameters, the allocated compute budget, and, most pertinent to this work, the size of the training dataset @kaplan_scaling_2020@bahri_scaling_2024. These empirical laws have been robustly verified across a diverse range of machine learning domains, encompassing natural language processing @kaplan_scaling_2020, computer vision @fan_images_2024, and spoken language modeling @cuervo_language_2024.

Understanding how these fundamental principles of scalability apply specifically to the #abbr.a[TTS]-for-#abbr.a[ASR] task is of paramount importance. As #abbr.a[TTS] models continue to advance and are increasingly trained on ever-larger speech corpora @casanova_xtts_2024@chen_vall-e_2024, their inherent capability to generate vast quantities of high-quality synthetic speech expands proportionally. This chapter delves into the theoretical underpinnings of scaling laws and their adaptation to model the complex dynamics of training an #abbr.a[ASR] system exclusively on synthetic speech. We investigate how the choice of #abbr.a[TTS] training objective impacts the scalability of synthetic data for #abbr.a[ASR] training, specifically comparing the widely used Mean Squared Error (#abbr.a[MSE]) against Denoising Diffusion Probabilistic Models (#abbr.pla[DDPM]).

=== Neural Scaling Laws in Speech Recognition

For a broad spectrum of machine learning tasks, the relationship between the test error and the size of the training dataset ($D$) can be accurately modeled by a power-law function. In the domain of speech recognition, this scaling behavior has been empirically validated for discriminative #abbr.a[ASR] rescoring models @gu_discriminativespeech_2023. Under the assumption of a sufficiently large model capacity and computational budget, the Word Error Rate (#abbr.a[WER]) of an #abbr.a[ASR] system is primarily limited by the quantity of training data. This relationship can be expressed formally as:
$ text("WER")(D) = (D_c/D)^alpha $
where $D_c$ represents a characteristic dataset size (a positive constant specific to the task) and $alpha$ is a positive constant that empirically determines the rate of performance improvement with increasing data. This power law implies a consistent and predictable improvement in performance as more training data is provided, assuming the data distribution remains fixed and representative of the target task.

=== Challenges in Scaling with Synthetic Data

While the standard neural scaling law provides a robust framework for modeling performance when training on real, naturally occurring data, its direct application to the #abbr.a[TTS]-for-#abbr.a[ASR] task introduces unique complexities. In this paradigm, the training data for the #abbr.a[ASR] system is entirely synthetic, which violates the core assumption of a fixed, static data distribution. The quality, characteristics, and crucially, the diversity of the synthetic data are not constant; they are dynamic properties that depend directly on the generative #abbr.a[TTS] model used to produce them. The performance of this #abbr.a[TTS] model, in turn, is itself a function of the data it was trained on.

This dynamic interplay is particularly relevant in the low-data regimes often encountered in #abbr.a[TTS]-for-#abbr.a[ASR] research and practical applications, such as for low-resource languages. In such scenarios, the behavior of generative models can deviate significantly from the true target data distribution, and this deviation is profoundly influenced by the generative model's training objective.

==== Mean Squared Error (MSE) and Oversmoothing

As previously discussed in @03_tts[Chapter 3], #abbr.a[TTS] models trained to minimize Mean Squared Error (#abbr.a[MSE]) implicitly assume that the conditional probability distribution of the target speech feature (e.g., Mel spectrograms) given the input text is unimodal. Given that speech generation is an inherently "one-to-many" problem – where a single text can correspond to myriad valid acoustic realizations (as explained in @01_intro[Chapter 1]) – this unimodal assumption leads to the well-documented issue of oversmoothing. The model, when faced with multiple plausible outputs for a given input, learns to predict the statistical average of these possibilities @ren_revisiting_2022. The resulting synthetic speech often exhibits low spectral variance and lacks the fine-grained, natural diversity and intricate details present in real speech, contributing to the "distributional gap" highlighted in @05_ttsasr[Chapter 5].

==== Denoising Diffusion Probabilistic Models (DDPM) and Stochasticity

In contrast to #abbr.a[MSE]-based models, Denoising Diffusion Probabilistic Models (#abbr.pla[DDPM]) are explicitly designed to model the entire, potentially multi-modal, data distribution by learning to reverse a stochastic noising process @ho_denoising_2020. This inherent stochasticity and ability to capture a broader range of variability should, in theory, allow #abbr.pla[DDPM]-based #abbr.a[TTS] models to generate more diverse and representative synthetic speech, thus better approximating the complexity of real human speech. However, the effectiveness of #abbr.pla[DDPM] in capturing this greater variability is highly dependent on the availability of sufficient training data. In low-data scenarios, #abbr.pla[DDPM] may struggle to learn the complete and accurate reverse diffusion process, potentially leading to noisy or even mode-collapsed outputs @lin_common_2024.

This initial mismatch between the synthetic and real distributions, which varies with the scale of the training data and the specific model objective, is not adequately captured by the standard single-term power law. To accurately model the performance of #abbr.a[ASR] systems trained exclusively on synthetic data, a more nuanced scaling framework is required. This framework must account for the dynamic quality and coverage of the synthetic data distribution, especially in the critical low-data regime where generative models may not yet fully capture the target distribution's complexities.

=== Methodology

To comprehensively investigate the distinct scaling properties of synthetic data when generated by #abbr.pla[DDPM] versus #abbr.a[MSE]-based #abbr.a[TTS] models, we designed a systematic experimental framework. Our primary focus is to evaluate how #abbr.a[ASR] performance, quantified by the Word Error Rate Ratio (#abbr.s[WERR]), evolves with increasing #abbr.a[TTS] training dataset size and speaker diversity.

The core #abbr.a[TTS] architecture employed in our experiments consists of two U-Net models, as conceptualized in @fig_tts_arch[Figure 1]:
A U-Net Encoder ($"U-Net"_"ENC"$) is responsible for generating a two-dimensional representation of prosody. This model leverages the Continuous Wavelet Transform (#abbr.a[CWT]) of pitch and energy, similar to FastSpeech @ren_fastspeech_2019, and additionally includes the #abbr.a[CWT] of phone duration. This prosody representation $P$ is conditioned on the input phone sequence and speaker identity (represented by a d-vector $s$):
    $ P = "U-Net"_"ENC" ("phone sequence", s) $
    This multi-resolution analysis of prosodic features is crucial for subsequent synthesis stages, capturing both temporal and frequency-domain characteristics.
A second U-Net Decoder ($"U-Net"_"DEC"$) then transforms this prosody representation $P'$ (which is an expanded version of $P$ repeated along the time axis according to predicted durations) into a Mel spectrogram ($M$). This U-Net is additionally conditioned on high-level semantic features derived from the text input using a pre-trained Flan-T5-Base language model @chung_scaling_2024. The Mel spectrogram $M$ serves as the final acoustic output of the #abbr.a[TTS] system:
    $ M = "U-Net"_"DEC" (P', s, "text features") $
    Both U-Net models are trained using either the #abbr.a[MSE] objective or the #abbr.pla[DDPM] objective, allowing for a direct comparison of their scaling behaviors.

Our datasets are systematically derived from the large-scale LibriHeavy corpus @kang_libriheavy_2024. To ensure fair and controlled comparisons, we create distinct subsets varying along two dimensions:
   #emph[Dataset Size]: The size of the #abbr.a[TTS] training data is varied from 100 hours up to 5000 hours.
   #emph[Speaker Diversity]: For each dataset size, we create subsets with varying levels of speaker diversity: 'low' (25-1531 speakers), 'medium' (40-956 speakers), and 'high' (62-1531 speakers).

Within each experiment, three distinct subsets are created: one for #abbr.a[TTS] training, one for #abbr.a[ASR] training (using the synthetic output from the #abbr.a[TTS] model), and one for #abbr.a[ASR] evaluation (using real data). Crucially, there is no overlap in transcripts between these three sets, ensuring that #abbr.a[ASR] performance differences are solely attributable to variations in the #abbr.a[TTS] models and dataset conditions, not to data leakage or evaluation bias. The proportions of speakers are consistently maintained across all three subsets to ensure consistency in speaker representation.

The #abbr.a[TTS] models (both #abbr.a[MSE] and #abbr.pla[DDPM] configurations) are trained for 500,000 iterations using a batch size of 16 and a cosine learning rate schedule starting at $4 times 10^{-5}$. To stabilize training, an exponential moving average (#abbr.a[EMA]) decay rate of 0.9999 is applied to the model parameters. For inference with #abbr.pla[DDPM] models, a DDIM sampler is employed with 20 sampling steps, utilizing a classifier-free guidance weight of 7.5 and a rescale factor of 0.7 for optimal control over generation quality. In line with prior research @lin_common_2024, the training utilizes a rescaled noise schedule to ensure zero terminal signal-to-noise ratio (#abbr.a[SNR]), and inference commences from the last timestep, ensuring congruency between training and inference processes.

For #abbr.a[ASR] evaluation, a Conformer-CTC model is utilized. The primary metric for assessing the quality of the synthetic data is the #abbr.s[WERR], defined as the ratio of the #abbr.a[WER] achieved by an #abbr.a[ASR] model trained exclusively on synthetic speech to the #abbr.a[WER] achieved by the same #abbr.a[ASR] model trained exclusively on real speech (as detailed in @05_ttsasr[Chapter 5]). A lower #abbr.s[WERR] indicates that the synthetic speech more closely matches the characteristics of real speech in terms of its utility for #abbr.a[ASR] training.

#figure(
  diagram(
    spacing: 5pt, // Spacing between default cells
    cell-size: (15mm, 15mm), // Defines the base size for nodes, influencing layout
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    
    // U-Net Encoder (Prosody)
    blob((0, 3), [Phone Sequence], width: 5em),
    edge((0, 3), (0, 2), "-|>"),
    blob((0, 2), [Speaker (d-vector)], width: 5em),
    edge((0, 2), (0, 1), "-|>"),
    blob((0, 1), [U-Net Encoder], width: 5em), // U-NetENC
    edge((0, 1), (0, 0), "-|>"),
    blob((0, 0), [Prosody Representation (P)], width: 5em), // P = CWT(Pitch, Energy, Phone Duration)

    // U-Net Decoder (Mel Spectrogram)
    blob((2.5, 3), [Text Features (Flan-T5-Base)], width: 5em),
    edge((2.5, 3), (2.5, 2), "-|>"),
    blob((2.5, 2), [Speaker (d-vector)], width: 5em),
    edge((2.5, 2), (2.5, 1), "-|>"), // Re-uses speaker d-vector
    blob((2.5, 1), [U-Net Decoder], width: 5em), // U-NetDEC
    edge((2.5, 1), (2.5, 0), "-|>"),
    blob((2.5, 0), [Mel Spectrogram (M)], width: 5em),

    // Connections between U-Net Encoder and Decoder
    edge((0, 0), (2.5, 1), "--|>"), // Prosody Representation to U-Net Decoder, maybe P' expanded here

    // Implicit Conditioning Lines
    // Add lines to clarify conditioning for U-Net Encoder
    edge((0, 3), (0, 1), "--|>"), // Phone Sequence to U-Net Encoder
    edge((0, 2), (0, 1), "--|>"), // Speaker (d-vector) to U-Net Encoder

    // Add lines to clarify conditioning for U-Net Decoder
    edge((2.5, 3), (2.5, 1), "--|>"), // Text Features to U-Net Decoder
    edge((2.5, 2), (2.5, 1), "--|>"), // Speaker (d-vector) to U-Net Decoder
  ),
  placement: top,
  caption: [Simplified #abbr.a[TTS] model architecture utilizing two U-Net models for prosody representation and Mel spectrogram generation. The #abbr.a[U-Net] Encoder generates a prosody representation conditioned on phone sequence and speaker identity. The #abbr.a[U-Net] Decoder produces Mel spectrograms, conditioned on the prosody representation, speaker identity, and text features.],
) <fig_tts_arch>

@fig_tts_arch illustrates the two-U-Net architecture employed. The U-Net Encoder processes phone sequences and speaker identity to output a prosody representation. This representation, along with speaker identity and text features, then conditions the U-Net Decoder to generate the Mel spectrogram.

=== Results

Our comprehensive experiments reveal distinct and informative scaling behaviors for #abbr.a[MSE]-based and #abbr.pla[DDPM]-based #abbr.a[TTS] models when used for #abbr.a[ASR] training. These findings are summarized in @tab_scaling_results, which presents #abbr.s[WERR] values across varying #abbr.a[TTS] training dataset sizes and speaker diversity levels.

The #abbr.a[MSE] model consistently demonstrated strong performance in low-data regimes, specifically up to approximately 100 hours of #abbr.a[TTS] training data. In these initial phases, #abbr.a[MSE] models often outperformed #abbr.pla[DDPM] models, regardless of the level of speaker diversity. For example, at 100 hours, the #abbr.a[MSE] model achieved #abbr.s[WERR] values of 3.62, 3.66, and 3.92 for low, medium, and high speaker diversity respectively, compared to 8.07, 8.33, and 7.44 for the #abbr.pla[DDPM] model under the same conditions . 
// ref table 1
This initial advantage for #abbr.a[MSE] models can be attributed to their more deterministic nature, which may be beneficial when the training data is scarce and the model needs to quickly learn basic speech generation patterns without excessive stochasticity. However, a significant limitation of the #abbr.a[MSE] model became evident with increasing dataset size. It showed very limited improvement beyond the initial gains, reaching a plateau in performance. This behavior aligns with the hypothesis that #abbr.a[MSE]-based models are intrinsically biased towards generating oversmoothed outputs, thereby failing to leverage the richer variability present in larger datasets. This suggests that further scaling of #abbr.a[MSE] models offers diminishing returns, rendering them less suitable for scenarios requiring vast amounts of diverse training data.

In contrast, the #abbr.pla[DDPM] model exhibited a notably different scaling behavior. While initially underperforming in smaller data regimes (approximately 300 hours and less), the #abbr.pla[DDPM] model showed significant and continuous improvements as the #abbr.a[TTS] training dataset size increased. It progressively closed the gap with #abbr.a[MSE] models and eventually outperformed them in larger data regimes. For instance, at 2500 hours, the #abbr.pla[DDPM] model achieved the best reported #abbr.s[WERR] of 1.46 for high speaker diversity, substantially lower than the #abbr.a[MSE] model's 2.59 #abbr.s[WERR] at the same scale and diversity.
// ref table 1
This indicates that the #abbr.pla[DDPM] model's inherent stochasticity and its ability to model complex, multi-modal distributions allow it to make more effective use of larger and more diverse datasets, translating into better #abbr.a[ASR] performance.

Speaker diversity further amplified the advantages of #abbr.pla[DDPM] models. While #abbr.a[MSE] models showed inconsistent benefits from increased speaker diversity (sometimes even degrading performance with more speakers), #abbr.pla[DDPM] models consistently performed better with higher speaker diversity at larger dataset sizes. At 5000 hours, the highest speaker diversity setting yielded a 4% better #abbr.s[WERR] compared to the lowest diversity setting for #abbr.pla[DDPM]. 
// ref table 1
However, it is important to note that even for #abbr.pla[DDPM] models, the relative difference between the lowest and highest diversity settings diminished as training data size increased (from 8% at 100 hours to 4% at 5000 hours). This suggests that a similar effect of diminishing returns, akin to that observed with overall dataset size, also applies to speaker diversity.

#let werr_value(wer_synth, wer_real) = calc.round(wer_synth / wer_real, digits: 2)
#let wer_cell(wer, std_dev) = "$#wer $ ± $#std_dev$"

#figure(
  table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: center,
  toprule(),
  table.header(
    [Hrs.],
    [Div.],
    [Spk.],
    table.cell(colspan: 2, [*WERR*]),
    table.cell(colspan: 2, [*WER↓*]),
    [*Real WER↓*]
  ),
  [], [], [], [*DDPM*], [*MSE*], [*DDPM*], [*MSE*], [],
  toprule(),
  [], [Low], [], [8.07], [3.62], [wer_cell(78.4, "-")], [wer_cell(35.2, "-")], [wer_cell(9.72, "-")],
  [], [Medium], [], [8.33], [3.66], [wer_cell(80.3, "-")], [wer_cell(35.3, "-")], [wer_cell(9.64, "-")],
  [], [High], [], [7.44], [3.92], [wer_cell(71.3, "-")], [wer_cell(37.6, "-")], [wer_cell(9.59, "-")],
  midrule(),
  [], [Low], [], [3.09], [2.58], [wer_cell(30.2, "-")], [wer_cell(25.2, "-")], [wer_cell(9.76, "-")],
  [], [Medium], [], [2.98], [2.66], [wer_cell(29.0, "-")], [wer_cell(25.9, "-")], [wer_cell(9.73, "-")],
  [], [High], [], [2.46], [2.56], [wer_cell(23.7, "-")], [wer_cell(24.7, "-")], [wer_cell(9.64, "-")],
  midrule(),
  [], [Low], [], [2.29], [2.39], [wer_cell(22.4, "-")], [wer_cell(23.4, "-")], [wer_cell(9.79, "-")],
  [], [Medium], [], [2.19], [2.58], [wer_cell(21.1, "-")], [wer_cell(24.9, "-")], [wer_cell(9.65, "-")],
  [], [High], [], [2.04], [2.65], [wer_cell(19.7, "-")], [wer_cell(25.5, "-")], [wer_cell(9.62, "-")],
  midrule(),
  [], [Low], [], [1.58], [2.45], [wer_cell(15.5, "-")], [wer_cell(24.0, "-")], [wer_cell(9.79, "-")],
  [], [Medium], [], [1.58], [2.55], [wer_cell(15.4, "-")], [wer_cell(24.9, "-")], [wer_cell(9.76, "-")],
  [], [High], [], [1.53], [2.48], [wer_cell(14.8, "-")], [wer_cell(24.1, "-")], [wer_cell(9.71, "-")],
  midrule(),
  [], [Low],[], [1.52], [2.42], [wer_cell(15.0, "-")], [wer_cell(23.9, "-")], [wer_cell(9.87, "-")],
  [], [Medium],[], [1.47], [2.52], [wer_cell(14.4, "-")], [wer_cell(24.8, "-")], [wer_cell(9.84, "-")],
  [], [High],[], [1.46], [2.59], [wer_cell(14.2, "-")], [wer_cell(25.2, "-")], [wer_cell(9.72, "-")],
  midrule(),
  [], [Low],[], [1.50], [2.42], [wer_cell(15.0, "-")], [wer_cell(24.2, "-")], [wer_cell(10.0, "-")],
  toprule(),
),
caption: [Results for different dataset sizes and speaker diversities. WER values have been converted from percentages to raw values for consistency with WERR.],
) <tab_scaling_results>

@tab_scaling_results presents the detailed #abbr.s[WERR] and #abbr.a[WER] results across varying #abbr.a[TTS] training dataset sizes and speaker diversity levels. The trends illustrate the distinct scaling behaviors of #abbr.pla[DDPM] and #abbr.a[MSE] models.

=== Proposed Scaling Law

To quantitatively model the observed scaling dynamics in #abbr.a[TTS]-for-#abbr.a[ASR], we propose a two-term power law that accounts for two distinct phases in performance improvement. This framework aims to capture both the initial rapid gains in low-data regimes and the eventual diminishing returns as dataset size increases, which are often indicative of underlying model limitations. The proposed scaling law for #abbr.s[WERR] as a function of dataset size ($D$) is given by:
$ text("WERR")(D) prop D^(-alpha) + D^(-gamma) $
Here, $alpha$ and $gamma$ are positive constants. The term $D^{-alpha}$ parametrizes the initial variance-limited phase, where additional data leads to rapid improvements in #abbr.a[ASR] performance as the #abbr.a[TTS] model learns to better approximate the underlying speech distribution and leverage the variability present in larger datasets. The term $D^{-gamma}$ parametrizes the resolution-limited phase, representing the diminishing returns observed as the dataset size further increases. In this phase, the #abbr.a[TTS] model's complexity or inherent biases (such as oversmoothing in #abbr.a[MSE]) limit its ability to extract further usefulness from additional synthetic data for #abbr.a[ASR].

By fitting this two-term power law to our experimental results, we derived approximate values for these parameters for both model types:
   For the #abbr.pla[DDPM] model, the fitted parameters were $alpha = 1.86$ and $gamma = 0.06$. The smaller $alpha$ value suggests a slower initial rate of improvement in the variance-limited phase compared to #abbr.a[MSE]. However, the larger $gamma$ value indicates that #abbr.pla[DDPM] exhibits a more sustained scaling behavior in the resolution-limited phase, leading to continuous, albeit slower, improvements with very large datasets.
   For the #abbr.a[MSE] model, the fitted parameters were $alpha = 2.93$ and $gamma = 0.01$. The larger $alpha$ value for #abbr.a[MSE] reflects a faster initial rate of improvement in the variance-limited phase, indicating that #abbr.a[MSE] models quickly learn the core speech generation patterns in low-data settings. However, the significantly smaller $gamma$ value (closer to zero) indicates that #abbr.a[MSE] models experience a much quicker stagnation in #abbr.s[WERR] as dataset size increases, confirming their earlier plateauing behavior observed in the results section.

These fitted power laws are visually represented in .
// ref figure 2
Even with the improved scaling properties of #abbr.pla[DDPM], our optimistic projections indicate that an enormous amount of synthetic data—on the order of at least one million hours—would be required for #abbr.pla[DDPM]-generated speech to match the performance of #abbr.a[ASR] systems trained on real data. This requirement far exceeds the size of currently available open datasets for #abbr.a[TTS] training @pratap_mls_2020@chen_gigaspeech_2021, highlighting the substantial persistent gap that remains.

#figure(
  [placeholder],
  caption: [Word Error Rate Ratio (WERR) as a function of #abbr.a[TTS] training dataset size for #abbr.pla[DDPM] and #abbr.a[MSE] models. This conceptual plot illustrates the distinct scaling behaviors: #abbr.pla[DDPM] shows better scalability over large datasets, while #abbr.a[MSE] plateaus due to oversmoothing.],
  placement: top,
) <fig_werr_scaling>

@fig_werr_scaling visually depicts the distinct scaling behaviors. It shows that while #abbr.a[MSE] models quickly reach a performance plateau, #abbr.pla[DDPM] models continue to improve with increasing data, albeit at a diminishing rate.

=== Discussion

These findings provide crucial insights into the effectiveness of different #abbr.a[TTS] training objectives for generating synthetic speech suitable for #abbr.a[ASR] training. The results indicate that the inherent limitations of #abbr.a[MSE]-based models, primarily stemming from their unimodal assumption and tendency towards oversmoothing, severely constrain their scalability and ability to leverage large, diverse datasets. This leads to a rapid plateau in #abbr.a[ASR] performance, making them less suitable for scenarios where vast amounts of synthetic data are desirable.

Conversely, #abbr.pla[DDPM] models, with their probabilistic nature and capability to model complex, multi-modal distributions, demonstrate significantly better scalability. They are more effective at utilizing large and diverse #abbr.a[TTS] training datasets, leading to sustained improvements in #abbr.a[ASR] performance. This makes #abbr.pla[DDPM]-based #abbr.a[TTS] models a more promising direction for future large-scale speech synthesis applications, particularly as access to massive unlabeled audio corpora continues to grow.

However, despite #abbr.pla[DDPM]'s superior scaling properties, our proposed two-term power law reveals that diminishing returns persist. Even with optimistic assumptions about the scaling behavior, reaching a #abbr.s[WERR] comparable to that achieved with real speech would require an unfeasibly large amount of synthetic data, potentially millions of hours, far exceeding the size of currently available open datasets. This indicates that while scaling is a powerful strategy, it alone may not be sufficient to fully bridge the performance gap between synthetic and real speech for #abbr.a[ASR] training. The persistent gap suggests that intrinsic limitations of current #abbr.a[TTS] models or the fundamental differences between synthetic and real speech distributions — such as unmodeled nuances of natural human variability or subtle artifacts — remain.

This observation highlights the necessity of exploring alternative and complementary approaches beyond merely scaling synthetic data. Future work should focus on methods that can directly address the persistent distributional dissimilarities, perhaps through novel #abbr.a[TTS] architectures, advanced training objectives, or hybrid approaches that combine the strengths of different generative paradigms. This persistent gap also underscores the critical need for robust and objective evaluation methodologies, like the #abbr.s[WERR] and other distributional distance metrics introduced in @05_ttsasr[Chapter 5] and @01_intro[Chapter 1], to accurately quantify progress and identify remaining challenges in creating truly human-like and functionally effective synthetic speech. The subsequent chapters of this thesis will delve further into such evaluation methodologies, emphasizing their role in guiding research towards closing this crucial gap.