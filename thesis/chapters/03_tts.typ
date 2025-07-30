#import "../abbr.typ" 
#import "../quote.typ": *
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
#import "../comic.typ"

== Text-to-Speech <03_tts>

As outlined in @01_intro[Chapter], we constrain this work to multi-speaker voice-cloning #abbr.a[TTS], in which there are two inputs; a speaker representation derived from a reference recording, which is most commonly a speaker embedding (see @02_speaker), but could also be a Mel Spectrogram or any other representation containing information about the given speaker @eskimez_e2_2024, like a text prompt describing their characteristics @lyth_parler_2024. Mapping these inputs to an acoustic realisation is a complex "one-to-many" problem @ren_revisiting_2022@blumstein_phonetic_1981. This chapter provides a comprehensive overview of TTS systems, drawing heavily on the survey by @tan_survey_2021, which categorizes neural TTS components and advanced topics. We expand on key methodologies, including historical context, frontends, and contrasts with related tasks like voice conversion, while adhering to the notation introduced in Chapter 1.

=== History of TTS

The evolution of TTS reflects advancements in signal processing, machine learning, and deep learning. Early systems relied on rule-based or concatenative methods, progressing to statistical parametric approaches, and finally to neural-based models. For a detailed historical survey, see #citep(<tan_survey_2021>).

Before the compute and data resources for #abbr.a[DNN]-based methods were available, three main approaches were used for #abbr.a[TTS].

*Concatenative synthesis* constructs speech by selecting and concatenating pre-recorded units of speech from a large database recorded by a single speaker @taylor_tts_2009. The most common form of this was #emph[unit selection synthesis]. For a given input text, the system would first determine a target sequence of phonetic units with associated prosodic features (e.g., pitch, duration). A search algorithm, typically Viterbi search, would then find the optimal path through the speech database to retrieve a sequence of waveform units. The "optimality" was determined by a cost function that balanced a #emph[target cost] (how well a database unit matches the target phonetic and prosodic features) and a #emph[join cost] (how smoothly two adjacent units can be concatenated) @hunt_unit_1996. While capable of producing high-quality and natural-sounding speech, these systems were limited by the contents of their database and could have audible concatenation points.

*Statistical parametric speech synthesis* emerged parallel to this, with #abbr.l[HMM]-based synthesis being the most prominent example @tukoda_hmm_2013. Instead of concatenating waveforms, #abbr.a[HMM]-based systems generate a smooth trajectory of acoustic parameters (like spectral features and fundamental frequency) from statistical models trained on speech data. Systems like Merlin @wu_merlin_2016 exemplified this era, using HMMs or DNNs for parameter prediction, often combined with vocoders for waveform generation.

*Hybrid models* represented the state-of-the-art prior to the deep learning revolution by combining the strengths of both approaches @ling_hybrid_2007. In a typical hybrid system, an #abbr.a[HMM]-based model would first generate the target acoustic parameter trajectories, providing flexible and natural prosody. Then, a unit selection component would search the database for waveform units that best matched these #abbr.a[HMM]-generated targets, rather than targets derived from simpler rules. This allowed for the high segmental quality of concatenative synthesis while leveraging the superior prosodic modelling of statistical methods.

#comic.comic((150mm, 80mm), "Comic timeline showing evolution from concatenative (pre-recorded units pieced together) to statistical parametric (parameter trajectories) to hybrid systems", blue) <fig_tts_history>

=== TTS Frontends

A critical but often overlooked component of TTS systems is the frontend, or text analysis module, which transforms raw text into linguistic features for synthesis @tan_survey_2021. This includes text normalization (converting non-standard words like "1989" to "nineteen eighty nine"), word segmentation (especially for languages like Chinese), part-of-speech tagging, prosody prediction, and grapheme-to-phoneme (G2P) conversion. G2P, for instance, maps written text to phonetic representations (e.g., "speech" to /s p iː tʃ/), essential for accurate pronunciation, particularly in languages with irregular orthography like English or polyphones in Chinese @yao_g2p_2015.

In neural TTS, frontends are simplified, often handling only normalization and G2P, as models directly process characters or phonemes. However, advanced systems incorporate neural frontends for end-to-end processing @hayashi_espnet-tts_2020. For details on frontend tasks, see @tan_survey_2021[Section 2.2].

=== Voice Conversion vs. TTS

While TTS maps text $T$ to speech $S$ (conditioned on variables like speaker embedding $z = r(S)$ as in Equation 1.2), voice conversion (VC) transforms an input speech signal $S$ to a target style while preserving content. Formally, VC conditions on $S$ rather than $T$: $tilde(S) tilde Q_theta (S | S, z)$, where $z$ might represent target speaker timbre or style. Unlike TTS, VC often disentangles content from prosody/timbre @sisman_vc_2020. This distinction is crucial, as VC enables unconditional generation or style transfer without text, complementing TTS in applications like dubbing.

=== Unconditional Generation

Beyond text-conditioned synthesis, unconditional generation models produce speech without explicit input, sampling from learned distributions. These often use autoregressive architectures to generate waveforms directly, as in early WaveNet @oord_wavenet_2016, which models $Q(S)$ without $T$ or $z$. Unconditional models are foundational for tasks like audio completion or creative sound design, but require careful handling of sequence length and variability to avoid artifacts @tan_survey_2021.

=== Hierarchical #sym.arrow.l.r #abbr.l[E2E] <03_hier_e2e>

Seeing #abbr.a[TTS] as a *hierarchical pipeline* breaks the problem into a series of steps and representations, moving from the utterance-level information such as speaker @stanton_speaker_2022 and lexical content, to phone level, frame level (i.e. mel spectrogram or MFCC frames) to sample level. The precise levels might differ in their definition and purpose between systems, but generally there is a gradual transformation from lower- to higher-resolution representations, ending in the raw waveform. The individual transformations might be accomplished using #abbr.pla[DNN], other learned modules or rule-based systems.

The second paradigm is the #abbr.a[E2E] approach in which a #abbr.a[DNN] directly predicts the output from the input. In many other domains, this approach has lead to consistently better results, however, for the high-resolution, continuous and highly inter-correlated nature of audio signals, this does not necessarily seem to be the case at the time of writing. Even the most recent #abbr.a[E2E] systems often use one or two components of the hierarchical approach, most commonly the #emph[vocoder], which converts mel spectrograms or other intermediate representations to raw waveforms @eskimez_e2_2024@chen_vall-e_2024, as well as the #abbr.a[g2p] conversion module @casanova_xtts_2024. Thus, TTS architectures exist on a spectrum, blending hierarchical and E2E elements for optimal performance @tan_survey_2021.

#comic.comic((150mm, 80mm), "Comic spectrum showing hierarchical TTS (step-by-step pipeline) blending into fully E2E TTS (direct text-to-waveform)", purple) <fig_hier_e2e>

=== #abbr.l[AR] #sym.arrow.l.r #abbr.l[NAR] <03_ar_nar>

If we let $T$ be the input lexical sequence (e.g., sub-word tokens), $bold(S) = (s_1, dots, s_n)$ be the target acoustic sequence (e.g., Mel-spectrogram frames), and $bold(e)_S$ be the speaker embedding vector for the target speaker, the goal of a multi-speaker conditional TTS model is to learn the distribution $p(bold(S)|bold(T),bold(e)_S)$.

*#abbr.l[AR] models* model this probability sequentially such that:

$ p(bold(S)|bold(T),bold(e)_S) = product_(i=1)^n p(s_i|s_1,dots,s_(i-1),bold(T),bold(e)_S) $

Here, $n$ represents the sequence length, determined by a stopping mechanism (e.g., an end-of-sequence token or predicted duration). #abbr.a[AR] can lead to better #abbr.a[TTS] performance, but usually shows less strict adherence to the lexical and speaker conditioning, since it is additionally conditioned on its own output, which can cause it to revert to unconditional generation in some cases @mehta_neuralhmm_2022.

In contrast *#abbr.l[NAR] models* assume conditional independence between output frames given the full conditioning information:

$ p(bold(S)|bold(T),bold(e)_S) = product_(i=1)^n p(s_i|bold(T),bold(e)_S) $

#comic.comic((150mm, 80mm), "Comic comparison of AR generation (sequential, step-by-step) vs. NAR (parallel, all-at-once)", green) <fig_ar_nar>

=== Objectives

When #link(<part_01>, [Part II]) of this work was conceptualised, the most common training objective for TTS was the #abbr.a[MSE] loss in use for the #abbr.a[NAR] FastPitch @lancucki_fastpitch_2021 as well as FastSpeech 1 @ren_fastspeech_2019 and 2 @ren_fastspeech_2021. On the #abbr.a[AR], Tacotron2 @wang_tacotron_2017 uses the same objective.

==== Mean Squared Error (MSE)

$ cal(L)_(text("MSE"))(theta)=EE[||bold(S)-f(bold(T),bold(e)_S;theta)||_2^2] $

Our exploration of different forms of conditioning for TTS-for-ASR systems in @06_attr[Chapter] utilises this objective. 

However, this can lead to oversmoothing of the output @ren_revisiting_2022 which caused the exploration of other objectives.

==== Denoising Diffusion Probabilistic Models (DDPM)

Among these is the *#abbr.a[DDPM] objective* @ho_denoising_2020 which we explore in detail in @07_scaling[Chapter]. In this approach speech is generated by learning to reverse a fixed forward process, $q$, which gradually adds Gaussian noise to the target speech $bold(S)$ over $N$ steps. This noising process is governed by a variance schedule $beta_n$. The model, parameterized by $theta$, learns the reverse denoising process, $p_theta(s_(n-1)|s_n)$, by predicting the parameters $(mu_theta, sigma_theta)$ to remove the noise at each step. It is trained by minimizing a loss function derived from the evidence lower bound (ELBO), which ensures the model can accurately reconstruct the original data from a noised state.

The simplified objective is to train a noise-prediction network, $epsilon_theta$, to predict the added noise $epsilon$ from a noised sample $bold(S)_t$ at any timestep $t$:

$ cal(L)_(text("DDPM"))(theta) = EE_(bold(S)_0, bold(epsilon), t, c) [|bold(epsilon) - bold(epsilon)_theta (bold(S)_t, t, c)\|_2^2] $

where $c$ represents the conditioning variables such as text and speaker information; $bold(S)_0$ is the original data; $bold(S)_t$ is the noised version at step $t$; and $bold(epsilon)$ is the noise added.

==== Negative Log-Likelihood (NLL)

Another way to generate synthetic speech is emerging, inspired by #abbr.pla[LLM] -- speech is converted into a discrete sequence of tokens and the task is to simply predict the next (speech) token @lyth_parler_2024. This can be used for unconditional speech generation, as well as conditioned on text tokens or a specific speaker to effectively make it a #abbr.a[TTS] or #abbr.a[VC] system. The training objective for such autoregressive, token-based models is the standard cross-entropy loss, which seeks to minimize the #abbr.a[NLL] of the ground-truth token sequence. This requires a preliminary step where the continuous speech waveform $bold(S)$ is encoded into a sequence of discrete tokens $bold(s) = (s_1, s_2, ..., s_L)$ using a neural audio codec.

The model is then trained to predict each token $s_i$ given the preceding tokens $bold(s)_(<i)$ and conditioning information $c$. The loss is the sum of the #abbr.pla[NLL] over the entire sequence:

$ cal(L)_(text("NLL"))(theta) = EE_(bold(s), c) [ - sum_(i=1)^L log p_theta(s_i | bold(s)_(<i), c) ] $

Here, $p_theta(s_i | bold(s)_(<i), c)$ is the probability assigned by the model to the correct token $s_i$ at timestep $i$. By minimizing this loss, the model learns the conditional probability distribution of the discrete speech representation. For unconditional generation, $c$ can be omitted, reducing to $p_theta(s_i | bold(s)_(<i))$, allowing free-form speech synthesis without input constraints @tan_survey_2021.

#comic.comic((150mm, 80mm), "Comic overview of TTS objectives: MSE (averaging errors), DDPM (denoising process), NLL (token prediction)", orange) <fig_tts_objectives>

=== Developments in modern TTS

While for #abbr.pla[LLM], a specific (decoder-only) architecture and training paradigm has emerged @naveed_llmoverview_2023, there is great diversity in the approaches to #abbr.a[TTS]. First, it is important to note that for #abbr.a[TTS] *Hierarchical* and #abbr.a[E2E] as well as #abbr.a[NAR] and #abbr.a[AR] exist on a spectrum, where systems blend elements of both to balance efficiency and quality @tan_survey_2021. For example, a model which generates Mel spectrograms and uses a separate vocoder to convert them to waveforms is #emph[mostly] #abbr.a[E2E], but still has the one hierarchical component of the vocoder. On the side of #abbr.a[NAR] and #abbr.a[AR], there is also the possibility of hybrid models which solve part of the #abbr.a[TTS] task in a #abbr.a[NAR] fashion while solving another part using an #abbr.a[AR] approach @wang_slmeval_2024. Modern #abbr.a[AR] approaches mostly predict tokens or speech codes and are often referred to as #abbr.pla[SLM]. These models greatly benefit from scale of data and model parameters, but can suffer in the areas of speaker identity and intelligibility, and sometimes hallucinate, meaning they generate speech not present in the transcript @wang_slmeval_2024. On the other hand purely #abbr.a[NAR] models have evolved as well, with the most popular recent approach abandoning both explicit alignment and #abbr.a[g2p] -- they do this by providing the transcript a sequence of characters as input to the model. Instead of phone- or word-level duration prediction, only the overall length of the utterance is predicted, and the input sequence padded to match this duration. The output representation of these models are usually Mel spectrograms, and they are trained using diffusion @eskimez_e2_2024@chen_f5_2024.

#comic.comic((150mm, 150mm), "A figure showing schematics of the most common TTS architectures, namely nonautoregressive-with-correlates, mel-autoregressive, token-based, token-based hierarchical, diffusion-nonautoregressive, ...", green) <fig_tts_arch>