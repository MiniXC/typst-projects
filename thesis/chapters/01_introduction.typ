#import "../abbr.typ" 
#import "../quote.typ": *
#import "../math.typ": *
#import "../moremath.typ": *

== Introduction <01_intro>

#q(
  [#citep(<taylor_tts_2009>)], 
  [#emph[Text-to-Speech Synthesis]],
  [… it is normally fairly
  easy to tell that it is a computer talking rather than a human, and so substantial progress is still to
  be made.]
)

Synthetic speech refers to any speech that is artificially generated, without the direct use of a human's vocal tract. This technology has found applications in diverse fields such as assistive technology, entertainment, and education for several decades. The most prevalent paradigm within synthetic speech generation is #abbr.l[TTS], the primary objective of which is to automatically convert a sequence of text into an utterance that sounds natural. However, the space of possible utterances is inherently vast. For instance, even a brief utterance can exhibit an immense number of variations in prosody, intonation, and recording conditions, a complexity that far exceeds the current capabilities of any single TTS system. A given sentence can be realised in a multitude of ways, contingent upon factors such as the speaker's emotional state, the surrounding environment, or their specific communicative intent, thereby defining a high-dimensional space of potential acoustic outputs. Consequently, TTS systems, by their fundamental design, must approximate a subset of this extensive space, frequently prioritising the naturalness of the output over an exhaustive coverage of all conceivable realisations.

=== Text-to-Speech (TTS)

Formally, let $cal(T)$ denote the space of all text sequences and $cal(S)$ represent the space of all possible utterances. The core objective of TTS is to model the true conditional probability distribution $Q(S|T)$ for any arbitrary pair $(T, S)$, where $T in cal(T)$ and $S in cal(S)$. This is achieved by training a generative model, denoted as $f^"TTS"_theta (dot)$, where the parameter set $theta$ is learned through training on pairs $(T,S)$ sampled from this true distribution. The model is consequently optimised to acquire an approximation of this distribution, $Q_theta$, which can then be utilised to sample a synthetic utterance $tilde(S)$ from an arbitrary input text $T$. In its simplest form, assuming speaker independence for initial conceptualisation, this relationship can be expressed as:

$
tilde(S) ~ Q_theta (S | T)
$ <tts_formula>

While @eqt:tts_formula represents the simplest case of #abbr.a[TTS], the resulting utterance $tilde(S)$ is typically not conditioned solely on the text input $T$. Additional conditioning variables, collectively denoted as $Z$, are often incorporated. These variables are generally derived by applying a transformation function $cal(R)(S)$ to a reference speech utterance. Some of these functions are inherently reductive, in which case neither the original speech $S$ nor its text $T$ can be perfectly reconstructed from the resulting representation. Nevertheless, these representations are designed to retain salient, yet abstracted, information pertinent to the speech. A common instance of $Z$ is the identity of the speaker, often encoded as a learned vector, a simple numeral, or derived from a reference utterance; a scenario referred to as voice-cloning TTS. However, TTS systems can be conditioned on an arbitrary number and variety of $cal(Z)$ values, constituting a set $cal(Z) = {Z_1, dots, Z_n}$, each derived using some transformation function $R(S)$. In practical real-world TTS applications, particularly when addressing the multitude of possible speech realisations, conditioning on speaker representations is frequently integrated.

The inherent complexity of modeling all possible real speech realisations without a perfect model of each speaker often leads to fundamental differences between real and synthetic speech. For example, some synthetic speech might exhibit characteristics that could not be produced by a human, such as the robotic quality sometimes introduced by certain vocoders. This is often characterised by a buzzy or metallic timbre resulting from insufficient fidelity in the synthesis of Mel spectrograms, as noted by #citep(<hu_syntpp_2022>) and #citep(<morise_world_2016>). Conversely, these differences could be more subtle and imperceptible to human listeners, as is the explicit goal in applications such as speech watermarking @nematollahi_watermarking_2013.

Given these difficulties, the evaluation of TTS systems typically relies on human judges. This methodology involves synthesising a number of utterances from a test set such that for each synthetic pair $(T,tilde(S))$ there exists a matching reference $(T,S)$. Human judges are then requested to rate both the synthetic and real speech (without prior knowledge of which is which) based on attributes such as #emph[naturalness] or #emph[quality]. This approach is intended to emphasise differences in speech that are perceptually relevant to listeners, while de-emphasising those that are perceptually irrelevant. Recent research has indicated that with advancements in TTS, the observation made by #citep(<taylor_tts_2009>) -- that it is normally fairly easy to tell that it is a computer talking -- no longer consistently applies. Contemporary synthetic utterances frequently receive ratings that are statistically indistinguishable from those of real human speech, as evidenced by #citep(<chen_vall-e_2024>), #citep(<tan_naturalspeech_2024>), and #citep(<eskimez_e2_2024>). These recent improvements have prompted investigations into whether such high-quality synthetic data can be leveraged for tasks where large quantities of speech data are advantageous.

=== Automatic Speech Recognition (ASR)

The inverse of #abbr.a[TTS] is #abbr.l[ASR], which aims to automatically transcribe a speech utterance into its corresponding text sequence. Therefore, the objective of ASR is to model the conditional probability distribution $P(T|S)$. This is typically framed as a prediction task, where a model $f^"ASR"_phi$ with parameters $phi$ is trained on $(S,T)$ pairs to learn an approximation of this distribution. This model can then be used to infer the most probable text sequence $asrT$ for a given input utterance $S$:

$ asrT = argmax_(T in cal(T)) P_phi (T|S) $

Given that augmenting datasets is a well-established method for improving the robustness and performance of machine learning models, and synthetic data is often easier to acquire and label than real data @pomerleau_alvinn_1988, the high subjective ratings of #abbr.a[TTS]-generated speech have inspired a number of works to utilise synthetic data for training #abbr.l[ASR] systems #citep(<li_synthaug_2018>), #citep(<rosenberg_speechaug_2019>), #citep(<thai_lrasr_2019>). This approach is referred to as TTS-for-ASR, where TTS-generated data is employed to train or augment ASR models, leveraging the controllability of speech synthesis to address data scarcity in domains such as low-resource languages or domains.

=== The Distribution Gap

However, in this work, we find that the high human ratings of synthetic speech do not directly translate to its suitability for #abbr.a[ASR] training. In alignment with prior works such as #citep(<li_synthaug_2018>), #citep(<hu_syntpp_2022>), and #citep(<rossenbach_duration_2023>), we observe that while human evaluations often place synthetic speech within a few percentage points of real speech in terms of naturalness @wang_tacotron_2017, real speech typically outperforms synthetic speech by a factor of 2 or more for ASR training @casanova_cross_2023. This persistent discrepancy between human perception and downstream utility constitutes a central focus of this thesis. We propose that the primary reason for this performance gap lies in the inability of the TTS model's learned distribution $Q_theta$ to adequately approximate the true distribution of real speech, $Q$, especially concerning the subtle variability that ASR models leverage for robust performance. Motivated by this hypothesis, we introduce a framework and methodology to empirically measure the distance between these distributions and demonstrate that this distributional approach can predict human judgments across diverse datasets, systems, domains, and languages.

=== Publications

This thesis is based on the following publications, which have been adapted and extended to form its core chapters.

In our earlier works, we addressed TTS-for-ASR and the gap between synthetic and real speech for ASR model training:
- #cite(<minixhofer_evaluating_2023>, form: "full")
- #cite(<minixhofer_oversmoothing_2025>, form: "full")

This work inspired us to investigate synthetic speech distributions more holistically using distributional distance measures, which lead to the Text-to-Speech Distribution Score (TTSDS):
- #cite(<minixhofer_ttsds_2024>, form: "full")
- #cite(<minixhofer_ttsds2_2025>, form: "full")
Additionally, the Masked Prosody Model (MPM) which serves as a component of TTSDS was published in the following joint first-author publication.
- #cite(<wallbridge_mpm_2025>, form: "full")
An extended pre-print of TTSDS including multilingual results is under review at NeurIPS 2025.
- #cite(<minixhofer_ttsds2arxiv_2025>, form: "full")

=== Structure

This manuscript is divided into three parts, with background on speech and its representations, TTS and ASR covered in Part I, our TTS-for-ASR contributions in Part II, and TTSDS in Part III.
===== Part I
-  *@02_factors[Chapter]*: This chapter provides an examination of various factors and representations of speech, classifying them based on their properties and purposes within speech technology.
-   *@03_tts[Chapter]*: This chapter offers an overview of Text-to-Speech (TTS) systems, detailing their evolution, core architectures, training objectives, and evaluation.
-   *@04_asr[Chapter]*: This chapter describes Automatic Speech Recognition (ASR), its underlying principles, the challenges posed by speech variability, and an overview of various ASR architectures, modeling and training techniques.
===== Part II
-   *@05_ttsasr[Chapter]*: This chapter investigates the synthetic-real gap in ASR training, quantifying the performance disparity between models trained on real versus synthetic speech using the Word Error Rate Ratio (WERR).
-   *@06_diversity[Chapter]*: This chapter explores methodologies for enhancing the diversity of synthetic speech through explicit attribute conditioning and post-generation data augmentation.
-   *@07_scaling[Chapter]*: This chapter analyses the scaling properties of synthetic data for ASR training, comparing the effectiveness of Mean Squared Error (MSE)-based and Denoising Diffusion Probabilistic Model (DDPM)-based TTS models with increasing training data size.
===== Part III
-   *@08_dist[Chapter]*: This chapter outlines the measurement of distributional distance for synthetic speech, and detailing the development and validation of the Text-to-Speech Distribution Score (TTSDS).
-   *@09_dist[Chapter]*: This chapter covers improvements to TTSDS which increase its robustness and versatility, as well as additional validation of TTSDS across domains and languages.

*@10_conclusions[Chapter]* covers conclusions, limitations and future directions of our work.

=== Notation

To maintain clarity and consistency when discussing both speech synthesis and recognition, the following conventions are adopted throughout this thesis. A speech signal, representing a raw audio waveform recording, is denoted by $S$, while its corresponding linguistic content, a text sequence, is indicated by $T$. When a synthetic speech utterance is specifically referred to, which is one that has been artificially generated, the tilde notation $tilde(S)$ is employed. A transcribed text sequence, typically as predicted by an ASR model, is represented by $asrT$. True probability distributions, particularly when referring to an underlying distribution, are denoted by $Q (dot)$ or $P (dot)$. When a probability distribution is modeled by a system with learned parameters, it is represented as $Q_theta (dot)$ or $P_theta (dot)$, with the subscript $theta$ explicitly indicating these parameters. A generic model or function is denoted by $f (dot)$ or $f_theta (dot)$, where $theta$ explicitly signifies its learned parameters; specific models, such as a Text-to-Speech model or an Automatic Speech Recognition model, may be further indicated by superscripts like $f^"TTS"_theta (dot)$ or $f^"ASR"_phi (dot)$.

The concept of a general space or set is represented by $cal(X)$, where $X$ can be drawn from $cal(T)$ (text sequences), $cal(S)$ (utterances), or $cal(R)$ (transformation functions). A dataset of real speech-text pairs is denoted by $D$, while a synthetic dataset is indicated by $tilde(D)$. Superscripts such as $"train"$ or $"test"$ are used to specify the role of the dataset within experimental setups, for instance, $D^"TEST"$ denotes a held-out test dataset used for evaluation. A dataset or set of datasets comprising noise samples, utilised as distractors in distributional distance calculations, is denoted as $D^"NOISE"$. Additional conditioning variables in generative models, which might include speaker identity or prosodic attributes, are generally referred to collectively as $cal(Z)$.

In @02_factors[Chapter], we introduce some general factors of speech. To avoid redefining these throughout the thesis, as there are differing definitions for factors such as prosody in the literature, we refer to a specifically styled version of the term when using these definitions such as "#smallcaps[Prosody]".