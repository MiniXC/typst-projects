#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#import "../abbr.typ"
#import "../quote.typ": *
#import "@preview/drafting:0.2.2": inline-note

== TTS-for-ASR task <05_ttsasr>

#q(
  [#citep(<nikolenko_synthetic_2021>)],
  [#emph[Synthetic Data for Deep Learning]],
  [As soon as researchers needed to solve a real-world
computer vision problem with a neural network, synthetic data appeared.]
)

Using synthetic data for training has been around almost as long real training data -- its first recorded use was for training the self-driving neural network behind ALVINN in 1988, for which the authors "developed a simulated road generator which creates road images to be used as training exemplars for the network" since "changes in parameters such as
camera orientation would require collecting an entirely new set of road images" @pomerleau_alvinn_1988. 

The motivation behind using #abbr.a[TTS]-generated data for ASR in today's very different deep-learning landscape is similar -- if some parameters change it could be more efficient to generate said data, rather than go out and collect it. Instead of camera orientation, for ASR, these parameters could be speaker identity @du_speaker_2020@casanova_singlespeaker_2022, lexical content @rosenberg_speechaug_2019@fazelSynthASRUnlockingSynthetic2021 or even duration of the phones in the speech @rossenbach_duration_2023. The usual approach is to use real and synthetic data in conjunction (since it is rare that no real data is available) to gain a small improvement @li_synthaug_2018. When augmenting an existing real dataset in this way, a crucial consideration is the ratio of real to synthetic data. Several works have found a 50:50 split to be an effective and robust choice @li_synthaug_2018 @rosenberg_speechaug_2019 @wang_improving_2020. However, this is not a fixed rule; when the synthetic speech offers significantly increased style diversity, for example through a #abbr.a[VAE], a much smaller proportion of real data can be effective. For instance, @sun_generating_2020 demonstrated a 16% relative WER improvement with a split of only 9% real data to 91% synthetic data.

The data most commonly used for TTS-for-ASR is read audiobook speech, such as LibriSpeech @panayotov_librispeech_2015 and LibriTTS @zen_libritts_2019.

#figure(
  grid(
    columns: 1,
    row-gutter: 0mm,
    column-gutter: 0mm,
    image("../figures/3/tts_for_asr.svg"),
    diagram(
      spacing: 5pt,
      cell-size: (4mm, 10mm),
  
      // legend
      node((0,1.5), align(left)[1 #sym.arrow @li_synthaug_2018], width: 100mm),
      node((0,2), align(left)[2 #sym.arrow @rosenberg_speechaug_2019], width: 100mm),
      node((0,2.5), align(left)[3 #sym.arrow @rossenbach_synthattention_2020], width: 100mm), 
      node((0,3), align(left)[4 #sym.arrow @casanova_singlespeaker_2022], width: 100mm),
      node((0,3.5), align(left)[5 #sym.arrow @hu_syntpp_2022], width: 100mm), 
      node((0,4), align(left)[6 #sym.arrow @karakasidis_multiaccent_2023], width: 100mm), 
      node((0,4.5), align(left)[7 #sym.arrow @rossenbach_duration_2023], width: 100mm), 
      node((0,5), align(left)[8 #sym.arrow @yuen_adaptation_2023], width: 100mm),
      node((0,5.5), align(left)[9 #sym.arrow @rossenbach_model_2024], width: 100mm),
      node((0,6),[],width: 100mm),
    )
  ),
  caption: "TTS-for-ASR performance in terms of WERR over time.",
) <fig_werr>

=== Methods for synthetic data diversity

A variety of techniques have been developed to generate synthetic speech that is not just natural-sounding, but also diverse and robust enough for ASR training.

A primary approach is to introduce and control sources of variation through *latent variables*. Models learn a latent space that is ideally independent of the text content and can be sampled from to control the style of the generated speech. The two most common methods for this are #abbr.pla[GST] @wang_style_2018, which learn a set of discrete style tokens from reference speech, and #abbr.pla[VAE] @kingma_auto-encoding_2013, which learn a continuous latent distribution. VAEs have become the most common solution for improving controllabilty in TTS-for-ASR @casanova_yourtts_2022 @sun_generating_2020.

A more direct form of control is achieved through *explicit conditioning*. Here, specific attributes are extracted from reference audio and used as additional inputs to the TTS model. This commonly includes conditioning on speaker representations like d-vectors to control speaker identity @du_speaker_2020 @wang_improving_2020, as well as prosodic features like pitch, energy, and phoneme durations to control the speaking style @rossenbach_duration_2023.

Finally, the synthetic speech can be made more suitable for real-world ASR by applying *post-generation data augmentation*. This involves adding simulated background noise or acoustic reverberation to the clean synthetic output, which helps the ASR system become more robust to varied environmental conditions @rossenbach_synthattention_2020.

=== Training on synthetic data alone
However, a more fundamental question is how well synthetic speech can be used to train #abbr.a[ASR] on its own. It is natural to assume that if #abbr.pla[MOS], human ratings, which are explained in more detail in @07_subjective, are statistically indistinguishable between synthetic and real speech @chen_vall-e_2024@tan_naturalspeech_2024 so should be their usefulness as training data for #abbr.a[ASR] systems.

As can be seen in @fig_werr, this is not the case, with #abbr.pla[WERR], defined as the ratio between #abbr.pla[WER] when training on synthetic compared to real speech, tending towards $2$ rather than $1$ as would be expected if they were truly equivalent -- even with the methods outline in  To illustrate this, we can compute the ratio of Mean Opinion Scores (MOS) as well. Using this measure, Tacotron 2 @shen_natural_2018, which was published a year prior to the earliest system in @fig_werr, achieves a subjective score ratio of $approx 1.02$, still $0.65$ lower than the best reported TTS-for-ASR systems' #abbr.a[WERR].
 
This observation sets up much of the exploration in #link(<part_01>, [Part I]) of this work. Somehow, listeners seem to give favorable ratings to synthetic speech which consistently produces close to #emph[$2$ times] the #abbr.a[WER] of real speech. What is missing in the speech that explains this gap? We seek to answer this question in the following Chapters.

