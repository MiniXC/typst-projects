#import "../abbr.typ" 
#import "../quote.typ": *

== Modeling and Training TTS and ASR Models <03_modeling>

#q(
  [#citep(<sutton_bitter_2019>)],
  [#emph[The Bitter Lesson]],
  [The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.]
)

In this chapter, we introduce common architectures and training approaches for #abbr.l[TTS] and #abbr.l[ASR] as preliminaries for our contributions in later chapters. They are introduced here since TTS-for-ASR requires an understanding of both (@05_ttsasr[Chapter]).

=== Text-to-speech

As outlined in @01_intro[Chapter], we constrain this work to multi-speaker voice-cloning #abbr.a[TTS], in which there are two inputs; a speaker representation derived from a reference recording, which is most commonly a speaker embedding (see @02_speaker), but could also be a Mel Spectrogram or any other representation containing information about the given speaker @eskimez_e2_2024, like a text prompt describing their characteristics @lyth_parler_2024. Mapping these inputs to an acoustic realisation is a complex "one-to-many" problem @ren_revisiting_2022@blumstein_phonetic_1981.
There are two main paradigms for accomplishing this task:

==== Hierarchical

Seeing #abbr.a[TTS] as a *hierarchical pipeline* breaks the problem into a series of steps and representations, moving from the utterance-level information such as speaker @stanton_speaker_2022 and lexical content, to phone level, frame level (i.e. mel spectrogram or MFCC frames) to sample level. The precise levels might differ in their definition and purpose between systems, but generally there is a gradual transformation from lower- to higher-resolution representations, ending in the raw waveform. The individual transformations might be accomplished using #abbr.pla[DNN], other learned modules or rule-based systems.

// #inline-note[TODO: Expand this section and add a figure, the hierarchical approach is a good way to explain TTS in general.]

==== End-to-end
The second paradigm is the #abbr.a[E2E] approach in which a #abbr.a[DNN] directly predicts the output from the input. In many other domains, this approach has lead to consistently better results, however, for the high-resolution, continuous and highly inter-correlated nature of audio signals, this does not necessarily seem to be the case at the time of writing. Even the most recent #abbr.a[E2E] systems often use one or two components of the hierarchical approach, most commonly the #emph[vocoder], which converts mel spectrograms or other intermediate representations to raw waveforms @eskimez_e2_2024@chen_vall-e_2024, as well as the #abbr.a[g2p] conversion module @casanova_xtts_2024.

==== Large-language-model-based

A third way of synthetic speech generation is emerging, inspired by #abbr.pla[LLM] -- speech is converted into a discrete sequence of tokens and the task is to simply predict the next (speech) token @lyth_parler_2024. This can be used for unconditional speech generation, as well as conditioned on text tokens or a specific speaker to effectively make it a #abbr.a[TTS] or #abbr.a[VC] system.

==== Autoregressive and nonautoregressive

If we let $T$ be the input lexical sequence (e.g., sub-word tokens), $bold(S) = (s_1, dots, s_n)$ be the target acoustic sequence (e.g., Mel-spectrogram frames), and $bold(e)_S$ be the speaker embedding vector for the target speaker, the goal of a multi-speaker conditional TTS model is to learn the distribution $p(bold(S)|bold(T),bold(e)_S)$.

*#abbr.l[AR] models* model this probability sequentially such that:

$ p(bold(S)|bold(T),bold(e)_S) = product_(i=1)^n p(s_i|s_1,dots,s_(i-1),bold(T),bold(e)_S) $

#abbr.a[AR] can lead to better #abbr.a[TTS] performance, but usually shows less strict adherence to the lexical and speaker conditioning, since it is additionally conditioned on its own output, which can cause it to revert to unconditional generation in some cases @mehta_neuralhmm_2022.

In contrast *#abbr.l[NAR] models* assume conditional independence between output frames given the full conditioning information:

$ p(bold(S)|bold(T),bold(e)_S) = product_(i=1)^n p(s_i|bold(T),bold(e)_S) $


==== Objectives

When #link(<part_01>, [Part I]) of this work was conceptualised, the most common training objective for TTS was the #abbr.a[MSE] loss in use for the #abbr.a[NAR] FastPitch@lancucki_fastpitch_2021 as well as FastSpeech 1 @ren_fastspeech_2019 and 2 @ren_fastspeech_2021. On the #abbr.a[AR], Tacotron2 @wang_tacotron_2017 uses the same objective.

$ cal(L)_(text("MSE"))(theta)=EE[||bold(S)-f(bold(T),bold(e)_S;theta)||_2^2] $

Our exploration of different forms of conditioning for TTS-for-ASR systems in @06_attr[Chapter] utilises this objective. 

However, this can lead to oversmoothing of the output @ren_revisiting_2022 which caused the exploration of other objectives. Among these is the *#abbr.a[DDPM] objective* @ho_denoising_2020 which we explore in detail in @07_scaling[Chapter]. In this approach speech is generated by learning to reverse a fixed forward process, $q$, which gradually adds Gaussian noise to the target speech $bold(S)$ over $N$ steps. This noising process is governed by a variance schedule $beta_n$. The model, parameterized by $theta$, learns the reverse denoising process, $p_theta(s_(n-1)|s_n)$, by predicting the parameters $(mu_theta, sigma_theta)$ to remove the noise at each step. It is trained by minimizing a loss function derived from the evidence lower bound (ELBO), which ensures the model can accurately reconstruct the original data from a noised state.

The simplified objective is to train a noise-prediction network, $epsilon_theta$, to predict the added noise $epsilon$ from a noised sample $bold(S)_t$ at any timestep $t$:

$ cal(L)_(text("DDPM"))(theta) = EE_(bold(S)_0, bold(epsilon), t, c) [|bold(epsilon) - bold(epsilon)_theta (bold(S)_t, t, c)\|_2^2] $

where $c$ represents the conditioning variables such as text and speaker information.