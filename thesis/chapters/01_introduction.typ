#import "../abbr.typ" 
#import "../quote.typ": *
#import "../math.typ": *
#import "../moremath.typ": *
#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#import fletcher.shapes: house, hexagon
#let blob(pos, label, tint: white, width: 26mm, ..args) = node(
	pos, align(center, label),
	width: width,
	fill: tint.lighten(60%),
	stroke: 1pt + tint.darken(20%),
	corner-radius: 5pt,
	..args,
)

== Introduction <01_intro>

#q(
  [#citep(<taylor_tts_2009>)], 
  [#emph[Text-to-Speech Synthesis]],
  [… it is normally fairly
  easy to tell that it is a computer talking rather than a human, and so substantial progress is still to
  be made.]
)

Synthetic speech is any speech that is artificially generated, without the use of a human's vocal tract. It has been used in assistive technology, entertainment, and education for decades @taylor_tts_2009. The most common paradigm is #abbr.l[TTS], the goal of which is to automatically convert a sequence of text into a natural-sounding utterance. Formally, let $cal(T)$ be the space of all text sequences and $cal(S)$ be the space of all utterances. The objective of TTS is to model the true conditional probability distribution $P(s|t)$ for any given pair $(t, s)$ where $t in cal(T)$ and $s in cal(S)$. This is achieved by training a generative model $f^"TTS"$, parameterized by $theta$, on a pairs $(t,s)$ sampled from this true distribution. The model is thereby optimized to learn an approximation of this distribution $P_theta$ which can then be used to sample a synthetic utterance $tilde(s)$ from an arbitrary input text $tilde(t)$:

$
tilde(s) ~ P_theta (s|tilde(t))
$ <tts_formula>

While @tts_formula is the simplest case of #abbr.a[TTS], $tilde(t)$ is usually not the only variable $s$ is conditioned on. These are derived by applying a reductive transformation function $r(s)$ from the the space of all such transforms $cal(R)$. Since the function is reductive, neither $s$ nor $t$ can be reconstructed from the representation -- however, they should hold salient, yet abstracted, information of the speech.

$
tilde(s) ~ P_theta (s | tilde(t), z) " where " z = r(s) " for some " r in cal(R)
$

The most comm on example of $z$ is the identity of the speaker in form of a learned vector, simple numeral or reference utterance, in which case we refer to voice-cloning TTS. However, TTS can be conditioned on any number and kinds of $z$.

A frequently-mentioned difficulty in TTS is the one-to-many problem @ren_fastspeech_2019 -- one combination of speaker identity and text can have many possible realisations. The fundamental impossibility of modelling all possible realisations without having a perfect model of each speaker leads to some fundamental differences between real and synthetic speech: some speech might be produced which could never be uttered by a human -- think the classic robotic sound caused by certain vocoders @hu_syntpp_2022, but these could also be more subtle and imperceptible by listeners such as is explicitly the goal for speech watermarking @nematollahi_watermarking_2013. Due to these difficulties evaluation of TTS usually relies on human judges: A number of utterances from a test set is synthesised such that for each $(tilde(s),tilde(t))$ pair there is a matching $(s,t)$ and then said human judges are asked to rate both synthetic and real speech (without knowing which is which), based on attributes such as #emph[naturalness] or #emph[quality]. Recent works have confirmed that with this method, #citea(<taylor_tts_2009>)'s leading quote from 2009 does no longer apply, with synthetic utterances now often receiving ratings statistically inseparable from real ones @chen_vall-e_2024@tan_naturalspeech_2024@eskimez_e2_2024.
The inverse of #abbr.a[TTS], is #abbr.l[ASR], which aims to automatically transcribe a speech utterance into its corresponding text sequence. Therefore, the objective of ASR is to model the conditional probability distribution $P(t|s)$
This is typically framed as a prediction task, where a model $f^"ASR"$ with parameters $Phi$ is trained on $(t,s)$ pairs to learn an approximation of this distribution. This model can then be used to infer the most probable text sequence $hat(t)$ for a given input utterance $hat(s)$:

$ hat(t) = argmax_(t in cal(T)) P_Phi (t|hat(s)) $

Since augmenting datasets is a tried-and-tested way to improve robustness and performance, and synthetic data is easier to acquire and label than real data @pomerleau_alvinn_1988, the high ratings of #abbr.a[TTS]-generated speech inspired a number of works to use synthetic data for training #abbr.l[ASR] systems @li_synthaug_2018@rosenberg_speechaug_2019@thai_lrasr_2019, which we refer to as TTS-for-ASR.

In this work, we first continue TTS-for-ASR research, and, in line with prior works @li_synthaug_2018@hu_syntpp_2022@rossenbach_duration_2023 find that the aforementioned high human ratings do not directly translate to suitability for #abbr.a[ASR] training. Instead of being within a few percentage points of the real speech as in human evaluation @wang_tacotron_2017 real speech usually outperforms synthetic by a factor of 2 or more for #abbr.a[ASR] training @casanova_singlespeaker_2022.
We make explaining this gap in performance the initial objective for this thesis, and introduce methodologies to both evaluate and reduce the gap. However, we hypothesise that the main reason for the gap is the inability of $P_theta$ to approximate $P$ in a way that captures the true distribution of real speech. Motivated by this we introduce a framework and methodology to measure the distance between these distributions empirically and show this distributional approach can predict human judgements across datasets, systems, domains and languages.

=== Contributions

The main contributions of this work are organized into two parts:

In Part II, we investigate the performance gap between real and synthetic speech when used for ASR training:

- We demonstrate that the gap in ASR performance is more significant than what subjective listening tests would suggest.

- We explore methods to enhance the diversity of synthetic speech through conditioning and scaling, finding that while these methods offer improvements, they do not fully close the performance gap, indicating a potential ceiling.

In Part III, we introduce and validate a new evaluation metric for synthetic speech:

- We propose the Text-to-Speech Distribution Score (TTSDS), an objective measure that quantifies the distributional dissimilarities between real and synthetic speech across various speech representations.

- We validate the TTSDS by demonstrating its strong correlation with longitudinal subjective listening test data from 2008 to 2024 and its effectiveness in evaluating systems that have reached human parity across different domains.

=== Notation

To maintain clarity when discussing both speech synthesis and recognition, we adopt the following notational conventions throughout this thesis. Speech signals are denoted by $S$, and their corresponding text is denoted by $T$. When both real and synthetic data are discussed in the same context, a tilde ($tilde$) is used to indicate the synthetic counterpart (e.g. $tilde(s)$ for a synthetic speech signal in $tilde(S)$).