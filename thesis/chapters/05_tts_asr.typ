#import "../abbr.typ"
#import "../quote.typ": *
#import "../math.typ": *
#import "../comic.typ"
#import "../moremath.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

== The Synthetic-Real Gap in ASR Training <05_ttsasr>

#q(
[#citep(<nikolenko_synthetic_2021>)],
[#emph[Synthetic Data for Deep Learning]],
[As soon as researchers needed to solve a real-world
computer vision problem with a neural network, synthetic data appeared.]
)

#ac("I need to break this chapter up more to make clear which parts are our contributions, specifically WERR and the initial TTS-for-ASR training experiments. A more detailed overview of the dataset splits is also still needed, as well as the information that we make sure to not resynthesise any data seen during training.")

Using synthetic data for training has been a used for machine learning since its early days, offering a practical solution to the challenges of real-world data collection. Its first recorded use dates back to 1988, when it was employed to train a neural network to steer a self-driving vehicle. The motivation was efficiency: "changes in parameters such as camera orientation would require collecting an entirely new set of road images" @pomerleau_alvinn_1988. In today's deep learning era, the rationale for leveraging Text-to-Speech (TTS)-generated speech in Automatic Speech Recognition (ASR) remains similar -- synthetic data can be generated with precise control over properties like #smallcaps[Speaker] identity, lexical content, or even #smallcaps[Prosody] features like phonetic durations, often more efficiently than gathering equivalent real data @rosenberg_speechaug_2019.

While synthetic data is frequently used to augment real datasets, in this work, we primarily treat it as a proxy for real speech: If the distributions of real speech $S$ and synthetic speech $tilde(S)$ were fully equivalent, we would in turn expect equivalent ASR performance when training on either. Training an ASR model on TTS-generated data -- TTS-for-ASR -- serves as an objective way to quantify the distributional distance between real and synthetic speech. This approach probes how well synthetic speech captures the variability of real speech in a way that directly impacts a downstream task. As we will show, this is not the case, and the resulting performance gap leads to our core research questions for this part of the thesis.

=== Evolution of TTS-for-ASR

The use of synthetic speech to augment training data for ASR has evolved significantly, driven by advancements in TTS quality and the persistent challenges of data scarcity. Initially, researchers explored this concept as a means to improve ASR for low-resource languages, where human-transcribed speech is particularly scarce. Early works, such as that by @rygaard_hmmtts4asr_2015, demonstrated that even simple HMM-based TTS models could be used to generate new audio to improve ASR performance in these settings.

With the advent of neural TTS models like Tacotron and WaveNet, the quality and naturalness of synthetic speech improved dramatically, leading to increased interest in its application for ASR data augmentation. Pioneering efforts by #citep(<li_synthaug_2018>) and #citep(<rosenberg_speechaug_2019>) showed that augmenting real datasets with high-fidelity synthetic speech could yield measurable improvements in #abbr.a[WER]. These works often focused on "mixed data" scenarios, where synthetic speech supplemented existing real data, with studies consistently showing that a mixing ratio (e.g., 50:50) provided reliable gains @li_synthaug_2018. The aim was to introduce targeted diversity -- be it in speaker variability, prosodic patterns, or phonetic coverage -- that was otherwise difficult to acquire from limited real corpora @sun_generating_2020.

The field has since diversified, exploring various facets of TTS-for-ASR. This includes augmenting specific types of diversity, such as speaker characteristics @du_speaker_2020 and various prosodic attributes @rossenbach_duration_2023. Researchers also investigated the application to highly specific domains, such as children's speech @shahnawazuddin_children_2020. A distinct line of research has focused on overcoming the inherent "distributional gap" between synthetic and real speech, proposing sophisticated techniques like rejection sampling and separate batch normalization statistics to mitigate artifacts and ensure consistency when mixing data @hu_syntpp_2022. More recently, novel frameworks like MAC have leveraged concatenative synthesis to boost low-resource ASR, integrating prior pronunciation knowledge @min_mac_2023. These efforts highlight a continuous drive to make synthetic data not just more natural, but also more representative of the complex variability found in real speech, thereby improving its utility for downstream ASR tasks.

=== Synthetic-Only Training for ASR

To rigorously evaluate how well TTS-generated speech captures the essential acoustic and linguistic variability of its real counterpart, training an ASR model exclusively on synthetic data provides a controlled experimental setup. If the generative model of the TTS system perfectly learned the true distribution of speech, then an ASR model trained solely on its synthetic output should achieve performance comparable to one trained on real speech, when evaluated on a real test set. A persistent performance gap, however, indicates systematic limitations of the TTS system in capturing aspects of speech variability.

Formally, consider a training dataset of real speech-text pairs 
$
D = {(S_(i), T_(i))}_(i=1)^(N)
$ 
and a corresponding synthetic dataset 
$
tilde(D) = {(tilde(S)_(i), T_(i))}_(i=1)^(N)
$ 

where $tilde(S)_(i)$ is generated by a TTS model from the text $T_(i)$. Let the ASR model be $f^"ASR"$, parameterized by $theta$ derived by training on $(S,T)$ pairs. When evaluating a synthetic-only trained ASR model, $f^"ASR" (dot;theta_(tilde(D)))$, as shorthand for $theta_tilde(D)$. its performance is measured on a real test set 
$
D^"test" = {(S^"test"_(j), T^"test"_(j))}^(M)_(j=1)
$

This approach allows for the direct observation of how well synthetic speech generalizes to real acoustic conditions. Previous works exploring this synthetic-only training paradigm reveal a consistent gap. For instance, #citep(<li_synthaug_2018>) found that ASR models trained exclusively on their neural TTS-generated data yielded significantly higher WERs compared to those trained on real data. #citep(<rosenberg_speechaug_2019>) similarly reported a substantial performance gap when using purely synthetic speech, highlighting that despite high subjective quality, the synthetic data still lacked critical variations for ASR robustness. #citep(<casanova_cross_2023>) also demonstrated this discrepancy, with models trained on synthetic data from various TTS systems exhibiting notably higher WERs on real test sets compared to models trained on human speech, even when using advanced cross-lingual voice conversion techniques for data augmentation. These findings suggest that human perception of "naturalness" in TTS does not directly translate to its utility for ASR training, implying that ASR models are sensitive to distributional differences that human listeners may not be able to detect.

=== Distributional Gap

#figure(
image("../figures/5/speech_dist.jpg", width: 100%),
caption: [Conceptual synthetic and real speech distributions.],
placement: top,
) <fig_speech_dist>

The aforementioned performance gap observed points to a fundamental discrepancy between the distributions of real and synthetic speech.
The disparity between synthetic and true data distributions can be conceptualized by partitioning the joint distribution of speech and its corresponding text into distinct regions, as illustrated by @hu_syntpp_2022[Figure 1]. This figure highlights four critical regions that characterize the "gap" as illustrated in @fig_speech_dist:

*Unnatural:* Synthetic speech may contain systematic errors or artificial sounds (e.g., structured noise, unnatural co-articulation, or unrealistic speaking styles) that are not present in real data. These patterns can be detrimental to downstream models, but are usually also penalised by listeners.

*Natural:* The synthesis process will most accurately represent the acoustic or prosodic patterns that were dominant in the training data.

*Missing:* Conversely, certain valid variations inherent in real speech (e.g., specific speaking styles, rare phonetic combinations, or subtle emotional nuances) might be under-represented or entirely missing from the synthetic distribution.

These differences could explain why an ASR model, optimized to extract robust linguistic features from highly diverse real speech, struggles when presented with synthetic data that lacks this full spectrum of variability. Human listeners, have no concept of the underlying speech distribution, and can only rate individual samples. A listener may perceive a synthetic utterance as "natural" because it is free from major distortions and follows general phonetic and prosodic rules. Yet, the same utterance might occupy a very narrow, high-density region within the vast space of plausible speech, failing to represent the rich diversity required for robust ASR training. This divergence between human perception and downstream task utility underscores the necessity of quantifying this distributional gap using objective, task-oriented metrics.

==== The Recognizability of Synthetic Speech

#figure(
table(
columns: (1fr, 1fr, 1fr),
align: center,
toprule(),
table.header([*Training Data*], [*Evaluation Data*], [*WER #sym.arrow.b*]),
toprule(),
[Real], [Real], [13.3 ± 0.29],
[Synthetic], [Real], [48.6 ± 0.43],
[Real], [Synthetic], [11.4 ± 0.69],
[Synthetic], [Synthetic], [3.0 ± 0.02],
toprule()
),
caption: "Results when training on synthetic and evaluating on real and vice versa.",
placement: top
) <tab_cross_ttsasr>

Beyond the mere existence of a distributional gap, the inherent regularity of synthetic speech has distinct implications for ASR model performance, manifesting in predictable and sometimes counter-intuitive ways. This phenomenon becomes evident when examining ASR models trained and evaluated on different combinations of real and synthetic data, as summarized in @tab_cross_ttsasr, for which we train a FastSpeech 2 @ren_fastspeech_2021 model on LibriTTS @zen_libritts_2019 and synthesise a 10 hour set for ASR training. A 6-layer hybrid HMM-TDNN system with a hidden unit size of 512, is trained with
the LF-MMI @hadian_lfmmi_2018 objective for 4 epochs on this resulting data and a real dataset with equivalent transcripts.

Firstly, when the ASR model is trained on real speech but evaluated on synthetic speech, its Word Error Rate (WER) is often surprisingly low, even lower than its performance on real evaluation data. For example, as shown in @tab_cross_ttsasr, an ASR model trained on real data yields a WER of 13.3% on real evaluation data, but only 11.4% on synthetic evaluation data. This seemingly anomalous result occurs despite synthetic speech being out-of-distribution for a model trained solely on real speech. The explanation might lie in the inherent "regularity" or "simplicity" of synthetic speech. Unlike real human speech, which contains a vast array of natural variabilities, noise, and spontaneous disfluencies, synthetic speech tends to be cleaner, more consistent, and often exhibits an oversmoothed quality, as discussed in @ren_revisiting_2022. For an ASR model that has learned to cope with the full complexity of real-world speech, this more "idealized" and less noisy synthetic input can be unexpectedly easier to transcribe. The acoustic-phonetic mappings in synthetic speech are often less ambiguous, allowing a robust ASR model to achieve a lower error rate, essentially exploiting the synthetic data's lack of complex, real-world acoustic challenges.

Secondly is the phenomenon observed when an ASR model is trained exclusively on synthetic speech and then evaluated on synthetic speech. In this scenario, the WER plummets to an extremely low value, as low as 3.0% in @tab_cross_ttsasr, which is impressive given the small amount of training data used. This dramatic reduction in error rates suggests that the ASR model is not merely recognizing speech; it is effectively learning to transcribe the unique and highly consistent patterns the TTS system is generating. While this leads to excellent in-domain performance on synthetic data, it fundamentally limits the ASR model's ability to generalize to the unpredictable variability of real human speech, thereby leading to the substantial performance gap discussed in the preceding section when evaluated on real data. This highlights that for TTS-for-ASR to be truly effective, the synthetic speech must not only be natural-sounding but also emulate the full distributional diversity of real speech, rather than presenting a merely recognizable, yet simplified, acoustic pattern.

=== Word Error Rate Ratio (WERR)

To rigorously quantify the disparity between ASR performance when trained on synthetic versus real speech, we introduce the Word Error Rate Ratio (WERR). This metric directly compares the effectiveness of different training data types. The WERR is formally defined as the ratio of the Word Error Rate (WER) achieved by an ASR model trained exclusively on synthetic speech to the WER achieved by an ASR model trained exclusively on real speech, when both are evaluated on the same real test set:

$ "WERR"(tilde(D), D) = "WER"(f^"ASR" (S;theta_tilde(D)), D^"test") / "WER"(f^"ASR" (S;theta_(D)), D^"test") $ <eq_werr>

Here, $theta_tilde(D)$ denotes the ASR model parameters trained on synthetic data $tilde(D)$, and $theta_D$ denotes the parameters trained on real data $D$. A WERR of 1 indicates that synthetic speech is acoustically equivalent to real speech for ASR training purposes. A value greater than 1 signifies that training on synthetic speech yields a higher error rate, indicating a performance gap. Our methodology for evaluating WERR emphasizes acoustic modeling by using ASR setups with minimal language model interference, ensuring the WER differences primarily reflect differences in the acoustic properties of the training data.

#figure(
image("../figures/5/tts_for_asr.svg", width: 80%),
caption: [Development of WERR over time. #text(fill: red)[TODO: update with new values.]],
placement: top,
) <fig_werr_trajectory>

==== WERR of Previous Works

Previous studies, while not always explicitly reporting WERR with this exact terminology, provide results that allow for its calculation, revealing a consistent performance gap between synthetic-only and real-only ASR training. @tab_werr_summary summarizes representative findings from the literature: ASR models trained solely on synthetic speech consistently yield significantly higher WERs compared to those trained on real speech, with WERR values generally ranging from approximately 9 to 1.7 and progress seems to plateau, as can be seen in @fig_werr_trajectory. Tacotron 2 @shen_natural_2018, which was used for the earlier TTS-for-ASR works @li_synthaug_2018@rosenberg_speechaug_2019 achieved a #abbr.a[MOS] of 4.53 compared to #abbr.a[MOS] 4.58 for ground truth -- a ratio of #calc.round((4.58/4.53),digits:2). This substantial gap persists with advanced neural TTS systems @rossenbach_model_2024. This observation motivates a deeper investigation into the underlying causes of this persistent gap.

#let gain(s,r) = {
  [#calc.round((1-(s/r))*100,digits:2)%]
}

#let werr(s,r) = {
  calc.round((s/r),digits:2)
}

#figure(
  table(
  columns: (.5fr, 4.7fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: center,
  toprule(),
  table.header([*\#*], [*Paper*], table.cell(colspan: 3, [*WER*]), [*Gain*], [*WERR*]),
  // table.header([*Paper*], [$theta_tilde(D)$], [$theta_D$], [$theta_(D+tilde(D))$], [*WER (Mixed Data)*], [*Rel. Gain (Mixed vs. Real)*]),
  [], [], [$theta_tilde(D)$], [$theta_D$], [$theta_(D+tilde(D))$], [], [],
  toprule(),
  
  [1], [#citep(<li_synthaug_2018>)], [49.80], [5.10], [4.66], [#gain(4.66,5.10)], [#werr(49.80,5.10)],
  
  [2],[#citep(<rosenberg_speechaug_2019>)], [32.44], [4.77], [4.58], [#gain(4.58,4.77)], [#werr(32.44,4.77)],

  
  [4], [#citep(<hu_syntpp_2022>) (Regular)], [7.0], [3.7], [3.2], [#gain(3.2,3.7)], [#werr(7.0,3.7)],
  [5], [#citep(<hu_syntpp_2022>) (Large)], [5.4], [2.9], [2.6], [#gain(2.6,2.9)], [#werr(5.4,2.9)],

  [6], [#citep(<casanova_cross_2023>)$crossmark$],  [56.84], [21.50], [20.39], [#gain(20.39,21.50)], [#werr(56.84,21.50)],
  
  [7], [#citep(<karakasidis_multiaccent_2023>)$crossmark$], [60.58], [8.42], [8.07], [#gain(8.07,8.42)], [#werr(60.58,8.42)],

  [8], [#citep(<rossenbach_duration_2023>)], [13.4], [3.4], [6.4], [#gain(3.4,6.4)], [#werr(13.4,3.4)],

  [9], [#citep(<yuen_adaptation_2023>)$crossmark$], [16.5], [10.3], [-], [-], [#werr(16.5,10.3)],

  [10],
  [#citep(<rossenbach_model_2024>) (AR)], [8.8], [5.1], [-], [-], [#werr(8.8,5.1)],

  [11],
  [#citep(<rossenbach_model_2024>) (NAR)], [11.0], [5.1], [-], [-], [#werr(11.0,5.1)],
  toprule(),
),
caption: [Comparison of ASR WERs purely synthetic, purely real, and mixed data, as well as WERR. Non-LibriSpeech works are indicated by $crossmark$.]
) <tab_werr_summary>

==== WERR and Vocoders

While the choice of vocoder is important in TTS, it does not seem to be as impactful in TTS-for-ASR. This can be quantified by computing WERR on re-synthesised data, meaning the vocoder representation is extracted from the original and then synthesied using said vocoder. #citep(<rossenbach_duration_2023>) do this using Griffin-Lim, which results in a WERR of $1.04$, at most accounting for approximately $5%$ of the gap. Additionally, we find in preliminary experiments that using Hifi-GAN @kong_hifigan_2020 consistently leads to WERR of approximately $1.00$, with no statistically significant difference between WER when training on resynthesised and real data. This is unsurprising, as Mel spectrograms are frequently used for ASR training in general @jurafsky_slp_2008, indicating that information important to ASR is not discarded when using this lossy-reconstructible transformation.

==== WERR and Symmetricity

While the WERR provides a quantitative measure of the performance disparity between ASR models trained on synthetic versus real speech, its interpretation as a formal statistical distance metric requires careful consideration. For $"WERR"$ to be a true metric, it would need to satisfy non-negativity ($"WERR"(X,Y)=0$), identity ($"WERR"(X,X)=0$) and symmetry ($"WERR"(X,Y)="WERR"(Y,Y)$). The former two are trivial to satisfy, however it is not immediately the clear if $"WERR"$ behaves in a symmetric way. For this to be the case, training on synthetic data and evaluating on real data would have to lead to similar results as vice versa. As shown in @tab_cross_ttsasr, we find that while the ratios are similar, with $"WERR"(tilde(D),D)=3.66$ and $"WERR"(D,tilde(D))=3.75$ (see @tab_cross_ttsasr) they are not equal due to the inherent stochasticity in model training.
To enforce symmetry, we define Mean Word Error Rate Ratio (MWERR) as the average of forward and reverse WERR:

$
"MWERR"(tilde(D), D) = 1/2 times ("WERR"(tilde(D), D) + "WERR"(D, tilde(D))
$ <eq_mwerr>

Augmentation experiments further highlight limitations: Techniques like SpecAugment improve ASR but distort distributions away from real speech @park_specaugment_2019. Thus, MWERR is a task-specific heuristic for dissimilarity, not a true metric. The WER itself is the outcome of a complex, non-linear optimization process during ASR model training and decoding, not a direct comparison of underlying data distributions. Therefore, while WERR gives us an empirical way to compare previous works and conclude that a significant gap exists, it cannot precisely quantify the specific causes of this gap. Differences in model architectures, training procedures, dataset splits, and evaluation conditions across various studies make it challenging to isolate the exact attributes of synthetic speech that contribute to its suboptimal performance for ASR training. This inherent lack of direct control over experimental variables in prior work necessitates a more controlled and systematic empirical investigation, which forms the basis for the subsequent chapters of this thesis.