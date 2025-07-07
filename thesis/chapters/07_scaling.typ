#import "../abbr.typ" 
#import "../quote.typ": *

== Scaling properties for TTS-for-ASR <07_scaling>

// #q(
//   [#citep(<sutton_bitter_2019>)],
//   [#emph[The Bitter Lesson]],
//   [The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.]
// )


The performance of deep learning models has been shown to improve predictably with scale. This phenomenon, formalized as *neural scaling laws*, describes a power-law relationship between a model's performance and factors such as the number of model parameters, the amount of compute, and, most crucially for this work, the size of the training dataset @kaplan_scaling_2020@bahri_scaling_2024. These laws have been empirically verified across numerous domains, including natural language processing @kaplan_scaling_2020, computer vision @fan_images_2024, and spoken language modeling @cuervo_language_2024.

Understanding how these principles apply to the TTS-for-ASR task is vital. As #abbr.s[TTS] models improve and are trained on ever-larger corpora @casanova_xtts_2024@chen_vall-e_2024, their potential to generate vast amounts of high-quality synthetic data grows. This chapter explores the theoretical background of scaling laws and how they can be adapted to model the dynamics of training an #abbr.s[ASR] system entirely on synthetic speech.

=== Neural scaling laws in speech recognition

For many machine learning tasks, the relationship between the test error and the training dataset size ($D$) can be modeled by a power-law function. In the context of speech recognition, this law has been shown to hold for discriminative #abbr.s[ASR] rescoring models @gu_discriminativespeech_2023. For a sufficiently large model and compute budget, the Word Error Rate (#abbr.s[WER]) is primarily limited by the amount of data, and this relationship can be expressed as:

$ text("WER")(D) = (D_c/D)^alpha $

where $D_c$ and $alpha$ are positive constants that are empirically determined for a given task and model architecture. This law implies that, for a given data distribution, performance will continue to improve as more training data is provided, following a predictable trajectory.

=== Challenges in scaling with synthetic data

While the standard scaling law provides a robust model for training on real data, its application to the TTS-for-ASR task, where the training data is entirely synthetic, requires special consideration. The core assumption of a fixed data distribution is violated, because the quality and characteristics of the synthetic data are not constant; they depend directly on the generative model that produced it. This model's performance, in turn, is a function of the data it was trained on.

This is particularly relevant in the low-data regimes often encountered in TTS-for-ASR research and applications. The behavior of generative models can deviate significantly from the target distribution when training data is scarce, and this deviation is heavily influenced by the model's training objective.

==== Mean squared error (MSE) and oversmoothing

As discussed in Section 3.1, #abbr.s[TTS] models trained to minimize MSE implicitly assume a unimodal distribution for the target speech. Given that speech generation is an inherently one-to-many problem, this assumption leads to the well-documented issue of *oversmoothing*, where the model learns to predict the average of the possible outputs @ren_revisiting_2022. The resulting speech exhibits low spectral variance, lacking the fine-grained diversity of real speech.

==== Denoising diffusion and stochasticity

In contrast, Denoising Diffusion Probabilistic Models (#abbr.pla[DDPM]) are designed to model the entire data distribution by learning to reverse a stochastic noising process @ho_denoising_2020. While this allows them to capture greater variability, their ability to do so faithfully is highly dependent on sufficient training data. In low-data scenarios, #abbr.pla[DDPM] may struggle to learn the complete reverse process, potentially leading to noisy or mode-collapsed outputs @lin_common_2024.

This initial mismatch between the synthetic and real distributions, which varies with data scale and model objective, is not captured by the standard power law. Therefore, to accurately model the performance of #abbr.s[ASR] systems trained on synthetic data, a more nuanced framework is required that can account for the quality and coverage of the synthetic data distribution, especially in the critical low-data regime.