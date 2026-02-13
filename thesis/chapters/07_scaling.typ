#import "../abbr.typ"
#import "../quote.typ": *
#import "../math.typ": *
#import "../moremath.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

== Scaling Properties of Synthetic Speech for ASR <07_scaling>

#q(
 [#citep(<sutton_bitter_2019>)],
  [#emph[The Bitter Lesson]],
  [The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.]
)

The principle that performance in machine learning improves with scale -- of data, computation, and model size -- is now a well-established fact. This phenomenon is often formalised by neural scaling laws, which describe a predictable and simple power-law relationship between a model's test error and these scaling factors @kaplan_scaling_2020. While these laws have been most documented for large language models, they have since been verified across a diverse range of domains, from computer vision to spoken language modelling @bahri_scaling_2024.
In the context of scaling for TTS-for-ASR, we pose the following research question: 

#emph[How does the choice of TTS training objective -- specifically Mean Squared Error versus a Denoising Diffusion Probabilistic Model -- and the scale of the TTS training data affect the utility of the generated synthetic speech for ASR training?

Can TTS-for-ASR behaviour with respect to TTS training data be modelled by neural scaling laws?]

The primary contribution of this chapter is the first comprehensive investigation into the neural scaling properties of synthetic speech for ASR training. We systematically compare the scaling behaviours of two distinct TTS training objectives: the conventional Mean Squared Error (MSE) and the probabilistic Denoising Diffusion Probabilistic Model (DDPM). To formally model the observed phenomena, particularly the significant performance disparity in low-data regimes, we propose a novel two-term power law that extends the standard neural scaling law to account for the initial distributional mismatch between synthetic and real data. Through our experiments, we quantify the scaling trajectories of both model types, linking their performance to the acoustic properties of the generated speech, specifically its spectral variance. These contributions were presented in the following publication:

- #cite(<minixhofer_oversmoothing_2025>, form: "full", style: "iso-690-author-date")

=== Neural Scaling Laws and Synthetic Data

The empirical study of neural scaling laws was prominently established by #citep(<kaplan_scaling_2020>), who demonstrated that language model performance, measured by the cross-entropy loss, scales smoothly and predictably as a power-law with the number of model parameters (N), the size of the training dataset (D), and the amount of compute used for training (C). A key insight from their work is that these factors are not independent; for optimal performance, they must be scaled in tandem. A particularly notable finding is that for a fixed compute budget, the optimal strategy involves training very large models on a relatively modest amount of data, stopping significantly short of convergence. This highlights the superior sample efficiency of larger models.

Subsequent research has extended these observations to the domain of synthetic data. Work by #citep(<fan_images_2024>) in computer vision confirmed that models trained on synthetic images also exhibit power-law scaling. However, their work revealed a crucial performance disparity: for supervised image classification, synthetic data scaled less effectively than real data. They identified two primary reasons for this gap: the inability of the generative models to accurately produce certain concepts and a lack of diversity in the generated images. This finding in the vision domain directly parallels the central problem of this thesis -- the synthetic-real gap in speech -- and provides a motivation for investigating how different generative modelling choices might affect these scaling properties.

=== Training Objectives and their Hypothesised Impact on Scaling

The standard neural scaling law provides a framework for modelling performance when training and testing on data from the same distribution. However, in the #abbr.a[TTS]-for-#abbr.a[ASR] paradigm, the training data is synthetic while the test data is real. The properties of the synthetic distribution are determined by the #abbr.a[TTS] model's training objective, which we hypothesise will have an impact on its scaling behaviour. There is limited formal theory linking specific generative objectives to scaling law parameters, thus this work provides an empirical investigation into this relationship.

The key connection we exploit is that the choice of objective constrains what aspects of the conditional distribution $p(S\mid T)$ a #abbr.a[TTS] model can represent at a given data scale. Objectives that encourage a single "best" prediction (e.g., per-frame regression) tend to produce mode-averaged outputs with reduced acoustic variability, whereas probabilistic objectives that explicitly model stochasticity can represent a wider range of plausible realisations. When synthetic speech is used as training data for #abbr.a[ASR] but evaluation remains on natural speech, this difference in distributional coverage manifests as a difference in how quickly downstream #abbr.a[ASR] performance improves as we scale the #abbr.a[TTS] training data. In other words: the training objective affects the synthetic distribution; the synthetic distribution determines the difficulty of the downstream transfer; and that transfer difficulty is what we observe as differing scaling curves.


==== Mean Squared Error and Oversmoothing

As discussed in @03_tts, #abbr.a[TTS] models trained to minimise #abbr.a[MSE] assume that the conditional probability distribution of the target speech feature (e.g., mel spectrograms) given the input text is unimodal. This assumption is fundamentally at odds with the "one-to-many" nature of speech, where a single text can have many valid acoustic realisations. This leads to the well-documented issue of oversmoothing, where the model learns to predict the statistical average of all plausible outputs @ren_revisiting_2022. The resulting synthetic speech often exhibits low spectral variance and lacks the fine-grained texture of real speech. We hypothesise that this inherent bias will cause #abbr.a[MSE]-based models to exhibit a distinct scaling pattern: they may perform relatively well in low-data regimes by quickly learning the average speech pattern, but their performance will quickly plateau as the lack of diversity becomes a bottleneck, preventing the #abbr.a[ASR] model from learning robust representations.

==== Denoising Diffusion Probabilistic Models (DDPMs) and Stochasticity

In contrast, #abbr.pla[DDPM] are explicitly designed to model the entire, potentially multi-modal, data distribution by learning to reverse a stochastic noising process @ho_denoising_2020. This class of generative models operates through two main phases: a forward diffusion process and a reverse denoising process.

The #emph[forward diffusion process] is a fixed Markov chain that gradually adds Gaussian noise to a clean data sample ($x_0$, typically a mel spectrogram in our case) over a series of $K$ timesteps. The process is defined by a variance schedule $beta_k in (0, 1)$ for $k=1, ..., K$:
$ q(x_k | x_(k-1)) = cal(N)(x_k; sqrt(1 - beta_k)x_(k-1),beta_k bold(I)) $
As $k$ increases, the data $x_k$ becomes progressively noisier, eventually approaching a standard Gaussian distribution at $x_K$.

The #emph[reverse denoising process] is where the model learns to invert this process. This is achieved by training a neural network, parameterised by $theta$, to approximate the conditional probability $p_theta (x_(k-1) | x_k)$, which is also assumed to be Gaussian:
$ p_theta (x_(k-1) | x_k) = cal(N)(x_(k-1); mu_theta (x_k, k), Sigma_theta (x_k, k)) $
In practice, the model is trained to predict the noise component $epsilon$ that was added at timestep $k$. The training objective is to minimise the difference between the true noise and the model's prediction, $epsilon_theta (x_k, k)$:
$ cal(L) = EE_(x_0, k, epsilon) [ ||epsilon - epsilon_theta (x_k, k)||^2_2] $
The inherent stochasticity of this process should, in theory, allow #abbr.pla[DDPM]-based models to generate more diverse speech that better approximates the complexity of the true distribution. However, accurately learning a complex, multi-modal distribution requires substantial data. We therefore hypothesise that #abbr.pla[DDPM]-based models will underperform #abbr.a[MSE] models in low-data regimes but will exhibit a more favourable scaling trajectory as data increases.

==== Challenges and Refinements in Diffusion Model Training
A common challenge in the training of #abbr.pla[DDPM] concerns the noise schedule and its implications for the training-inference mismatch @lin_common_2024. Many standard noise schedules do not enforce a zero Signal-to-Noise Ratio (#abbr.a[SNR]) at the final timestep $K$. A non-zero terminal #abbr.a[SNR] means that even at the highest noise level, $x_K$ still contains a small amount of residual signal from the original data $x_0$. During inference, however, the process starts from pure Gaussian noise, which has an #abbr.a[SNR] of zero. This discrepancy can lead to practical problems, such as a limited dynamic range in the generated outputs.

To mitigate this, we incorporate two refinements proposed by #citep(<lin_common_2024>): first, we rescale the noise schedule to ensure a terminal #abbr.a[SNR] of zero, making the training condition at timestep $K$ consistent with the inference starting point. Second, we ensure that the inference sampler always begins from the final timestep $K$. These corrections align the training and inference processes, enabling the model to generate samples that are more faithful to the learned data distribution.

=== A Framework for Evaluating Scaling Laws

To empirically test these hypotheses, we designed a controlled experimental framework to quantify the distinct scaling properties of synthetic data generated by #abbr.a[DDPM]- and #abbr.a[MSE]-based #abbr.a[TTS] models.

==== TTS and ASR Architectures
The core #abbr.a[TTS] architecture, shown in @fig:fig_tts_arch_scaling, consists of two U-Net models @ronneberger_u-net_2015. A U-Net Encoder ($"U-Net"_"ENC"$) generates a multi-resolution prosody representation from the input phone sequence and a speaker d-vector, using the #abbr.a[CWT] of pitch, energy, and duration.
$ H = "U-Net"_"ENC" ("G2P"(T), Z_"SPK") $
This prosody representation $H$ is then expanded according to predicted durations to create $hat(H)$. A second U-Net Decoder ($"U-Net"_"DEC"$) transforms this prosody representation into a mel spectrogram, additionally conditioning on high-level features from a pre-trained Flan-T5-Base language model @chung_scaling_2024:
$ cal(R)_"MEL" (tilde(S)) = "U-Net"_"DEC" (hat(H), Z_"SPK", f^"T5"_phi (T)) $
This U-Net architecture is crucial for #abbr.a[DDPM] training, as its structure, with its skip connections and hierarchical feature processing, is well-suited to the denoising task. For the #abbr.a[MSE] model, a FastSpeech 2-style architecture was also tested and found to perform similarly; however, we use the U-Net architecture for both objectives to ensure a direct and fair comparison of the training losses. For #abbr.a[ASR] evaluation, we use two distinct architectures: the hybrid HMM-TDNN model from our previous experiments (6-layer, 512 hidden units, LF-MMI objective), and a Conformer-CTC model (12-layer, 8 attention heads, CTC objective).

#figure(
  image("../figures/7/ddpm_architecture.png", width: 70%),
  placement: top,
  caption: [TTS model architecture utilising two U-Net models for prosody representation and mel spectrogram generation.],
) <fig_tts_arch_scaling>

==== Datasets and Experimental Design
All datasets are derived from the large-scale LibriHeavy corpus @kang_libriheavy_2024. To evaluate scaling, we created #abbr.a[TTS] training subsets of increasing size: 100, 250, 500, 1000, 2500, and 5000 hours. Following the controlled setup from @05_setup, for each data point, we created three disjoint sets of transcripts: one for training the #abbr.a[TTS] model, one for generating the synthetic speech for #abbr.a[ASR] training, and one for the real-speech test set for #abbr.a[ASR] evaluation. Speaker proportions were kept consistent across all subsets. The #abbr.a[TTS] models were trained for 500,000 iterations with a batch size of 16 and a cosine learning rate schedule. For #abbr.a[DDPM] inference, a #abbr.a[DDIM] sampler with 20 sampling steps was used, along with a classifier-free guidance weight of 7.5 and a rescale factor of 0.7. We also applied a random uniform masking of 0-50% of the input phone sequence during training as a regularisation technique. For the primary scaling law analysis presented in @fig:fig_scaling_lines and @fig:fig_scaling_werr, we fix the #abbr.a[ASR] training data size to 10 hours. This allows for a direct comparison with the diversity enhancement experiments in Chapter 6, isolating the effect of scaling the #abbr.a[TTS] model's training data.

This fixed-size #abbr.a[ASR] setting should be read as a controlled diagnostic slice rather than a claim that 10 hours is a particularly "natural" operating point. Holding the #abbr.a[ASR] training size constant removes a major confound and makes the figures easier to interpret: changes in downstream #abbr.a[WER] can be attributed primarily to the #abbr.a[TTS] model's training scale and objective. We additionally report results across a wider range of #abbr.a[ASR] training set sizes in @tbl:tab_synthetic_asr_wer to show how the observed trends change when both #abbr.a[TTS] and #abbr.a[ASR] scale.

==== A Two-Term Scaling Law for TTS-for-ASR
The standard neural scaling law describes performance on in-distribution data as a single power law: $
text("WER")(D) = (D_c/D)^alpha
$ 

Where $D_c$ and $alpha$ are parameters fit to the empirical data. To model the more complex behaviour observed in #abbr.a[TTS]-for-#abbr.a[ASR], where a distributional mismatch exists, we propose a two-term power law. This framework aims to capture both the initial challenges in low-data regimes and the eventual convergence to a standard scaling trend:
$ text("WERR")(D) = (D_v/D)^alpha + (D_r/D)^gamma $
Here, the first term $(D_v/D)^alpha$ parametrises an initial phase, dominated by the distributional mismatch between synthetic and real data. The second term $(D_r/D)^gamma$ parametrises the subsequent phase, which represents the diminishing returns as the synthetic distribution more closely approximates the real one, following a more conventional scaling behaviour.

In this formulation, $D$ denotes the #abbr.a[TTS] training data size (in hours); $text("WERR")(D)$ is the word error rate ratio measured by evaluating #abbr.a[ASR] models on a fixed, real-speech test set after training them on either real speech or synthetic speech generated by a #abbr.a[TTS] model trained on $D$ hours. $D_v$ and $D_r$ are fitted scale parameters (with the same units as $D$) that determine the relative weight of the two regimes. $alpha$ and $gamma$ are fitted exponents that control how quickly each term decays with increasing data. In the empirical analysis below, we fit $D_v, D_r, alpha, gamma$ to the measured scaling curves.

=== Empirical Scaling Behaviour and Analysis

#figure(
  table(
    columns: (20pt, 40pt, 100pt, 100pt),
    align: center,
    toprule(),
    table.header(
      [],
      [],
      [Hybrid],
      [Conformer],
    ),
    toprule(),
    table.cell(
    rowspan: 6,
    align: horizon,
      rotate(-90deg, reflow: true)[
        *WERR*
      ],
    ),
    table.vline(),
    [100], [3.69], [3.19],
    [250], [2.78], [2.66],
    [500], [1.83], [1.80],
    [1000], [1.29], [1.49],
    [2500], [#emph[N/A]], [1.34],
    [5000], [#emph[N/A]], [1.25],
    toprule(),
  ),
  caption: [Word Error Rates (WER) of ASR models trained on real speech.],
  placement: top
) <tab_real_asr_werr>

Our experiments reveal distinct scaling behaviours for the two training paradigms, confirming our initial hypotheses when the #abbr.a[ASR] training set is held constant (10 hours in @fig:fig_scaling_lines and @fig:fig_scaling_werr). The results are presented as #abbr.a[WER] achieved by #abbr.a[ASR] models trained on synthetic data (@fig:fig_scaling_lines) and as the #abbr.s[WERR] relative to models trained on real data of the same size (@fig:fig_scaling_werr). The baseline real-data #abbr.a[WER] values and the broader sweep over #abbr.a[ASR] training sizes are shown in @tbl:tab_synthetic_asr_wer.

==== Divergent Scaling Trajectories
As predicted, the #abbr.a[MSE] model demonstrates stronger performance in low-data regimes (below 500 hours). At 100 hours, the #abbr.s[WERR] for the #abbr.a[MSE] model is 3.66, whereas for the #abbr.a[DDPM] model it is 8.33. The deterministic nature of #abbr.a[MSE] allows it to quickly learn a stable, albeit averaged, representation of speech. However, its performance rapidly plateaus, showing minimal improvement beyond 1000 hours of training data. In contrast, the #abbr.a[DDPM] model, while initially struggling, exhibits a much more favourable scaling trajectory. Its performance improves continuously with more data, overtaking the #abbr.a[MSE] model at around 1000 hours and continuing to improve, achieving a best #abbr.s[WERR] of 1.46 at 2500 hours for this 10-hour #abbr.a[ASR] setup.

==== Variance as an Indicator of Distributional Coverage
The underlying reason for these divergent scaling behaviours is demonstrated by analysing the acoustic properties of the generated speech. @fig:fig_std_dev plots the average frame-wise standard deviation of the synthetic mel spectrograms, relative to that of the equivalent real data. The #abbr.a[MSE] model consistently produces under-varied, or oversmoothed, speech, with a spectral standard deviation up to 30% lower than real speech at 100 hours. This lack of variance is the direct cause of its performance plateau. Conversely, the #abbr.a[DDPM] model initially produces over-varied speech at 100 hours, which may correspond to noisy or unstable outputs when the model is undertrained. As the training data increases, the variance of the #abbr.a[DDPM]-generated speech converges towards that of real speech, aligning with its improving #abbr.s[WERR]. This shows a direct link between the model's ability to match the distributional properties of real data and the utility of its output for a downstream task.

#figure(
  image("../figures/7/std_dev.png", width: 80%),
  caption: [The average frame-wise standard deviation of the mel spectrograms, relative to the standard deviation of the equivalent real data.],
  placement: top,
) <fig_std_dev>

#figure(
  image("../figures/7/ddpm_mse_scaling.png", width: 80%),
  placement: top,
  caption: [WER scaling behaviour of DDPM and MSE models based on amount of TTS training data, with a fixed 10-hour ASR training set.],
) <fig_scaling_lines>

#figure(
  image("../figures/7/ddpmvsmse_werr.png", width: 80%),
  placement: top,
  caption: [WERR scaling behaviour of DDPM and MSE models based on amount of TTS training data across models, with a fixed 10-hour ASR training set.],
) <fig_scaling_werr>

==== Synthetic Speech Performance
While fixing the #abbr.a[ASR] training data allows for controlled analysis, exploring the full results in @tbl:tab_synthetic_asr_wer reveals further insights. When both the #abbr.a[TTS] and #abbr.a[ASR] training set sizes are increased, the #abbr.s[WERR] can be reduced even further. The lowest #abbr.s[WERR] achieved for the hybrid model is 1.07, obtained when training the #abbr.a[TTS] model on 5000 hours and the #abbr.a[ASR] model on 1000 hours of the resulting synthetic speech. For the Conformer model, the lowest #abbr.s[WERR] is 1.25,  as shown in @tbl:tab_real_asr_werr, achieved when training both the #abbr.a[TTS] and #abbr.a[ASR] systems on 5000 hours of data.

Interestingly, under certain conditions, synthetic data can outperform its real equivalent, resulting in a #abbr.s[WERR] below 1.0. For instance, when a Conformer #abbr.a[ASR] model is trained on just 100 hours of synthetic data generated by a #abbr.a[TTS] model that was itself trained on 5000 hours, the resulting #abbr.a[WER] is 5.88. This is substantially better than the 6.98 #abbr.a[WER] achieved when training on the corresponding 100 hours of real data, yielding a #abbr.s[WERR] of 0.84. We hypothesise that the large-scale #abbr.a[TTS] model learns a rich and diverse representation of speech from its vast 5000-hour training set. When generating a smaller 100-hour subset, it produces speech that is both highly varied and acoustically canonical, having averaged out some of the idiosyncratic speaker behaviours, disfluencies, or recording artefacts present in the smaller real dataset. This cleaner, yet still diverse, data can be a more efficient training signal for an #abbr.a[ASR] model with a limited data budget.

#figure(
  table(
    columns: (.5fr, 1fr, 1fr, 1fr, 1fr, 1fr, 
    1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center,
    toprule(),
    table.header(table.cell(colspan: 12)[*ASR Training Data*]),
    toprule(),
    [],[],
    table.cell(colspan: 4)[*Hybrid WER*],
    table.cell(colspan: 6)[*Conformer WER*],
    [],[],
    table.vline(),[~100~], [~250~], [~500~],[1000],table.vline(),
    [~100~], [~250~], [~500~],[1000],[2500],[5000],
    toprule(),
    table.cell(
    rowspan: 6,
    align: horizon,
      rotate(-90deg, reflow: true)[
        *TTS Training Data*
      ],
    ),
    [100], [*25.56*], [26.29], [26.25], [25.79], [*22.31*], [22.51], [21.89], [26.36], [23.80], [23.26],
    [250], [19.81], [18.33], [18.65], [*17.41*], [17.95], [17.8], [17.81], [17.72], [15.48], [*15.32*],
    [500], [13.18], [11.87], [*11.39*], [11.92], [11.95], [12.14], [11.23], [11.64], [*9.75*], [10.51],
    [1000], [8.23], [7.08], [*7.00*], [7.01], [7.16], [6.84], [6.71], [6.57], [*5.54*], [5.61],
    [2500], [7.44], [7.27], [*6.52*], [6.56], [6.67], [6.69], [5.95], [6.30], [5.59], [*5.60*],
    [5000], [6.10], [*5.71*], [5.78], [5.83], [5.88], [5.66], [5.2], [5.24], [*4.54*], [4.55],
    toprule(),
    table.cell(colspan: 2)[*Real*], [6.92], [6.57], [6.21], [5.41], [6.98], [6.70], [6.25], [4.40], [4.15], [3.64],
    toprule(),
  ),
  caption: [Word Error Rates (WER) of ASR models trained on synthetic speech, generated by Text-to-Speech (TTS) models.],
  placement: top,
) <tab_synthetic_asr_wer>


==== Fitting the Scaling Law
To formally capture the observed behaviours, we fit the two-term power law to the empirical WERR data from the 10-hour #abbr.a[ASR] setup. For the #abbr.a[DDPM] model, the fitted parameters were $alpha = 1.86$ and $gamma = 0.06$. The smaller $alpha$ value reflects its slower initial improvement, while the larger $gamma$ value indicates a more sustained scaling behaviour. For the #abbr.a[MSE] model, the parameters were $alpha = 2.93$ and $gamma = 0.01$. The large $alpha$ reflects a rapid initial improvement, while the near-zero $gamma$ confirms its quick stagnation. As shown in @fig:fig_scaling_werr, these fitted laws accurately model the empirical data, confirming the validity of our proposed two-term model.

=== The Limits of Scaling

Our findings provide clear insights into the effectiveness of different #abbr.a[TTS] training objectives for generating scalable synthetic speech. The results show that the inherent limitations of #abbr.a[MSE]-based models, due to their tendency towards oversmoothing, severely constrain their ability to benefit from large, diverse datasets. This leads to a rapid performance plateau, making them unsuitable for large-scale synthetic data generation. In contrast, #abbr.pla[DDPM], with their ability to model complex, multi-modal distributions, demonstrate far superior scalability, leading to sustained improvements in #abbr.a[ASR] performance. This positions #abbr.pla[DDPM]-based models as a more promising direction for the future of large-scale speech synthesis.

However, despite this superior scalability, our proposed two-term power law reveals that diminishing returns persist. Extrapolating the fitted curve for the #abbr.a[DDPM] model indicates that an enormous amount of synthetic data -- on the order of one million hours -- would be required to consistently reach a #abbr.s[WERR] of 1.0. This requirement vastly exceeds the size of any currently available speech corpus @pratap_mls_2020@chen_gigaspeech_2021, highlighting that scaling alone, even with a powerful generative model, is unlikely to be a complete solution for bridging the synthetic-real gap. Our best empirical result of a #abbr.s[WERR] of 1.07 required a significant mismatch in data size between the #abbr.a[TTS] and #abbr.a[ASR] systems. This persistent gap suggests that fundamental differences between the current models' capabilities and the true complexity of the real speech distribution remain, underscoring the need for continued research into both novel architectures and more robust, objective evaluation methodologies. In the following chapters, we address the latter, introducing a way to measure the distributional distance between synthetic and real speech beyond TTS-for-ASR.