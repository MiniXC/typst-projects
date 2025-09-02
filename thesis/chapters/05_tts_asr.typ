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

Using synthetic data for training has been instrumental for machine learning since its early days, offering a practical solution to the challenges of real-world data collection. Its first recorded use dates back to 1988, when it was employed to train a neural network to steer a self-driving vehicle. The motivation was efficiency: "changes in parameters such as camera orientation would require collecting an entirely new set of road images" @pomerleau_alvinn_1988. In today's deep learning era, the rationale for leveraging Text-to-Speech (TTS)-generated speech in Automatic Speech Recognition (ASR) remains similar -- synthetic data can be generated with precise control over properties like #smallcaps[Speaker] identity, lexical content, or even #smallcaps[Prosody] features like phonetic durations, often more efficiently than gathering equivalent real data @rosenberg_speechaug_2019.

While synthetic data is frequently used to augment real datasets, in this work, we primarily treat it as a proxy for real speech: If the distributions of real speech $S$ and synthetic speech $tilde(S)$ were fully equivalent, we would in turn expect equivalent ASR performance when training on either. Training an ASR model on TTS-generated data -- TTS-for-ASR -- serves as an objective way to quantify the distributional distance between real and synthetic speech. This approach probes how well synthetic speech captures the variability of real speech in a way that directly impacts a downstream task. Relating to TTS-for-ASR, we ask the following research questions in this chapter: 

#emph[How can we reliably compare and quantify TTS-for-ASR performance?]

#emph[How well has previous work approximated real speech for the TTS-for-ASR task?] 

#emph[Are there any obvious explanations for any arising gaps in performance, such as limitations of the vocoder or regularity of the synthetic speech?] 


To this end, we introduce a controlled TTS-for-ASR setup, minimising bias and ensuring fair comparison. We also introduce the #abbr.a[WERR] heuristic to compare previous efforts in TTS-for-ASR. Using WERR, we find that while TTS-for-ASR performance has increased, there seems to be a plateau, with synthetic speech performing by a factor of around 1.7 worse than real speech in the most recent works. We also evaluate WERR in our own experimental setup, and find that the vocoder does not play a significant role. Additionally, we find that synthetic speech is highly regular, leading to much lower WER when evaluating ASR performance on synthetic speech. While most of these contributions are preliminaries for later work which were not published outside of this thesis, the controlled TTS-for-ASR setup is covered in the following publication:

- #cite(<minixhofer_evaluating_2023>, form: "full")

=== Evolution of TTS-for-ASR

The use of synthetic speech to augment training data for ASR has evolved significantly, driven by advancements in TTS quality and the persistent challenges of data scarcity. Initially, researchers explored this concept as a means to improve ASR for low-resource languages, where human-transcribed speech is particularly scarce. Early works, such as that by #citep(<rygaard_hmmtts4asr_2015>), demonstrated that even simple HMM-based TTS models could be used to generate new audio to improve ASR performance in these settings.

With the advent of neural TTS models like Tacotron and WaveNet, the quality and naturalness of synthetic speech improved dramatically, leading to increased interest in its application for ASR data augmentation. Pioneering efforts by #citep(<li_synthaug_2018>) and #citep(<rosenberg_speechaug_2019>) showed that augmenting real datasets with high-fidelity synthetic speech could yield measurable improvements in #abbr.a[WER]. These works often focused on "mixed data" scenarios, where synthetic speech supplemented existing real data, with studies consistently showing that a mixing ratio (e.g., 50:50) provided reliable gains @li_synthaug_2018. The aim was to introduce targeted diversity -- be it in speaker variability, prosodic patterns, or phonetic coverage -- that was otherwise difficult to acquire from limited real corpora @sun_generating_2020.

The field has since diversified, exploring various facets of TTS-for-ASR. This includes augmenting specific types of diversity, such as speaker characteristics @du_speaker_2020 and various prosodic attributes @rossenbach_duration_2023. Researchers also investigated the application to highly specific domains, such as children's speech @shahnawazuddin_children_2020. A distinct line of research has focused on overcoming the inherent "distributional gap" between synthetic and real speech, proposing sophisticated techniques like rejection sampling and separate batch normalisation statistics to mitigate artefacts and ensure consistency when mixing data @hu_syntpp_2022. More recently, novel frameworks like MAC have leveraged concatenative synthesis to boost low-resource ASR, integrating prior pronunciation knowledge @min_mac_2023. These efforts highlight a continuous drive to make synthetic data not just more natural, but also more representative of the complex variability found in real speech, thereby improving its utility for downstream ASR tasks.

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

where $tilde(S)_(i)$ is generated by a TTS model from the text $T_(i)$. Let the ASR model be $f^"ASR"$, parameterised by $theta$ derived by training on $(S,T)$ pairs. When evaluating a synthetic-only trained ASR model, $f^"ASR" (dot;theta_(tilde(D)))$, as shorthand for $theta_tilde(D)$ its performance is measured on a real test set 
$
D^"test" = {(S^"test"_(j), T^"test"_(j))}^(M)_(j=1)
$

This approach allows for the direct observation of how well synthetic speech generalises to real acoustic conditions. Previous works exploring this synthetic-only training paradigm reveal a consistent gap. For instance, #citep(<li_synthaug_2018>) found that ASR models trained exclusively on their neural TTS-generated data yielded significantly higher WERs compared to those trained on real data. #citep(<rosenberg_speechaug_2019>) similarly reported a substantial performance gap when using purely synthetic speech, highlighting that despite high subjective quality, the synthetic data still lacked critical variations for ASR robustness. #citep(<casanova_cross_2023>) also demonstrated this discrepancy, with models trained on synthetic data from various TTS systems exhibiting notably higher WERs on real test sets compared to models trained on human speech, even when using advanced cross-lingual voice conversion techniques for data augmentation. These findings suggest that human perception of "naturalness" in TTS does not directly translate to its utility for ASR training, implying that ASR models are sensitive to distributional differences that human listeners may not be able to detect.

=== Distributional Gap

#figure(
image("../figures/5/speech_dist.png", width: 80%),
caption: [Conceptual synthetic and real speech distributions.],
placement: top,
) <fig_speech_dist>

The aforementioned performance gap observed points to a fundamental discrepancy between the distributions of real speech $Q$ and the distribution learned by a TTS model $Q_theta$.
The disparity between synthetic and true data distributions can be conceptualised by partitioning the joint distribution of speech and its corresponding text into distinct regions, as illustrated by @fig:fig_speech_dist:

*Unnatural:* Synthetic speech may contain systematic errors or artificial sounds (e.g., structured noise, unnatural co-articulation, or unrealistic speaking styles) that are not present in real data. These patterns can be detrimental to downstream models, but are usually also penalised by listeners.

*Natural:* The synthesis process will most accurately represent the acoustic or prosodic patterns that most common and regular (i.e. predictable) in the training data.

*Missing:* Conversely, certain valid variations inherent in real speech (e.g., specific speaking styles, rare phonetic combinations, or subtle emotional nuances) might be under-represented or entirely missing from the synthetic distribution.

These differences could explain why an ASR model, optimised to extract robust linguistic features from highly diverse real speech, struggles when presented with synthetic data that lacks this full spectrum of variability. Human listeners, have no concept of the underlying speech distribution, and can only rate individual samples. A listener may perceive a synthetic utterance as "natural" because it is free from major distortions and follows general phonetic and prosodic rules. Yet, the same utterance might occupy a very narrow, high-density region within the vast space of plausible speech, failing to represent the rich diversity required for robust ASR training. This divergence between human perception and downstream task utility underscores the necessity of quantifying this distributional gap using objective, task-oriented metrics.

=== Controlled TTS-for-ASR Setup <05_setup>

#figure(
image("../figures/5/tts4asr_splits.png", width: 80%),
caption: [Dataset splits and their use for TTS and ASR in our experiments.],
placement: top,
) <fig_tts4asr_splits>

We now introduce our controlled sequence of steps we follow to ensure our TTS-for-ASR results are as true a representation of the TTS model's capabilities as possible, shown in @fig:fig_tts4asr_splits: 

1. A TTS system is trained on a dataset $D^"TTS"_"train"$. A second dataset,#footnote[The second dataset is usually smaller than the first to emulate a low-resource scenario.] which does not share any text samples $T$ with this first dataset is constructed for ASR training, we name this $D^"ASR"_"train"$. 
2. The text samples $T^"ASR"_"train"$ of this dataset are used as inputs to the previously trained TTS system, and the pairs of these texts and the resulting waveforms become the synthetic $tilde(D)^"ASR"_"train"$ dataset. Importantly, $D^"TTS"_"train"$ and $D^"ASR"_"train"$ do not share any transcripts, making it impossible for given TTS system to simply reproduce samples in the training data. The distribution of speakers is kept identical for both the real and synthetic dataset to eliminate any bias in training.
3. Two equivalent ASR systems are trained with the real and synthetic data respectively. We generally hybrid HMM-DNN systems for this, as their reliance on an internal language model is less severe than for end-to-end systems, and we are mostly interested in the acoustic capabilities of TTS rather than its ability to reproduce the lexical content.
4. A held-out test set $D^"ASR"_"test"$, with no overlap in lexical content and speakers with any of the previous datasets, is used to derive the #abbr.a[WER] for the real and synthetically-trained ASR models with weights $theta$ and $tilde(theta)$ such that $"WER"(D_"test",theta)$ is the WER achieved using the model trained on real data and $"WER"(D,tilde(theta))$ is its equivalent trained using synthetic speech.

=== #("Word Error Rate Ratio (WERR)") <05_werr>

To quantify the disparity between ASR performance when trained on synthetic versus real speech, we also introduce the Word Error Rate Ratio (WERR). This heuristic directly compares the effectiveness of different training data types using ASR-derived #abbr.a[WER]. The WERR is defined as the ratio of the WER achieved by an ASR model trained exclusively on synthetic speech to the WER achieved by an ASR model trained exclusively on real speech, when both are evaluated on the same real test set:

$ "WERR"(D_"test", theta, tilde(theta)) = "WER"(D, tilde(theta)) / "WER"(D, theta) $ <eq_werr>

As $theta$ is derived using the real data $D$ and $tilde(theta)$ using the synthetic data $tilde(D)$, while the test set, TTS model and ASR models stays fixed, we refer to WERR as a heuristic of said datasets $"WERR"(D,tilde(D))$ for simplicity. A WERR of 1 indicates that the given synthetic speech is acoustically equivalent to real speech for ASR training purposes. A value greater than 1 signifies that training on synthetic speech yields a higher error rate, indicating a performance gap. Our methodology for evaluating WERR emphasises acoustic modeling by using ASR setups with minimal language model interference, ensuring the WER differences primarily reflect differences in the acoustic properties of the training data.

==== WERR and Symmetricity

While the WERR provides a quantitative measure of the performance disparity between ASR models trained on synthetic versus real speech, its interpretation as a formal statistical distance metric requires careful consideration. For $"WERR"$ to be a true metric, it would need to satisfy non-negativity ($"WERR"(X,Y)=0$), identity ($"WERR"(X,X)=0$) and symmetry ($"WERR"(X,Y)="WERR"(Y,Y)$). The former two are trivial to satisfy, however it is not immediately the clear if $"WERR"$ behaves in a symmetric way. For this to be the case, training on synthetic data and evaluating on real data would have to lead to similar results as vice versa. As shown in @tbl:tab_cross_ttsasr, we find that while the ratios are similar, with $"WERR"(tilde(D),D)=3.66$ and $"WERR"(D,tilde(D))=3.75$ (see @tbl:tab_cross_ttsasr) they are not equal due to the inherent stochasticity in model training.
To enforce symmetry, we define Mean Word Error Rate Ratio (MWERR) as the average of forward and reverse WERR:

$
"MWERR"(tilde(D), D) = 1/2 times ("WERR"(tilde(D), D) + "WERR"(D, tilde(D))
$ <eq_mwerr>

The fact that ASR data augmentation is frequently used shows a further limitation: Techniques like SpecAugment improve ASR but distort distributions away from real speech @park_specaugment_2019. Thus, MWERR is a task-specific heuristic for dissimilarity, not a true metric. The WER itself is the outcome of a complex, non-linear optimisation process during ASR model training and decoding, not a direct comparison of underlying data distributions. Therefore, while WERR gives us an empirical way to compare previous works and conclude that a significant gap exists, it cannot precisely quantify the specific causes of this gap. 

In the following sections, we analyse prior work and conduct a series of preliminary experiments to contextualise the synthetic-real gap and establish a baseline for further investigation. These experiments quantify the WERR of previous studies, establish a baseline for our own work, examine the impact of vocoders, and explore the implications of synthetic speech's inherent regularity.

=== WERR of Previous Works <05_werr_results>

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

  [], [], [$theta_tilde(D)$], [$theta_D$], [$theta_(D+tilde(D))$], [], [],
  toprule(),
  
  [1], [#citep(<li_synthaug_2018>)], [49.80], [5.10], [4.66], [#gain(4.66,5.10)], [#werr(49.80,5.10)],
  
  [2],[#citep(<rosenberg_speechaug_2019>)], [32.44], [4.77], [4.58], [#gain(4.58,4.77)], [#werr(32.44,4.77)],

  
  [3], [#citep(<hu_syntpp_2022>) (Regular)], [7.0], [3.7], [3.2], [#gain(3.2,3.7)], [#werr(7.0,3.7)],
  
  [4], [#citep(<hu_syntpp_2022>) (Large)], [5.4], [2.9], [2.6], [#gain(2.6,2.9)], [#werr(5.4,2.9)],

  [5], [#citep(<casanova_cross_2023>)$crossmark$],  [56.84], [21.50], [20.39], [#gain(20.39,21.50)], [#werr(56.84,21.50)],
  
  [6], [#citep(<karakasidis_multiaccent_2023>)$crossmark$], [60.58], [8.42], [8.07], [#gain(8.07,8.42)], [#werr(60.58,8.42)],

  [7], [#citep(<rossenbach_duration_2023>)], [13.4], [3.4], [6.4], [#gain(3.4,6.4)], [#werr(13.4,3.4)],

  [8], [#citep(<yuen_adaptation_2023>)$crossmark$], [16.5], [10.3], [-], [-], [#werr(16.5,10.3)],

  [9],
  [#citep(<rossenbach_model_2024>) (AR)], [8.8], [5.1], [-], [-], [#werr(8.8,5.1)],

  [10],
  [#citep(<rossenbach_model_2024>) (NAR)], [11.0], [5.1], [-], [-], [#werr(11.0,5.1)],
  toprule(),
),
caption: [Comparison of ASR WERs purely synthetic, purely real, and mixed data, as well as WERR. Non-LibriSpeech works are indicated by $crossmark$.],
placement: top,
) <tab_werr_summary>

Previous studies, provide results that allow for the calculation of WERR, revealing a consistent performance gap between synthetic-only and real-only ASR training. @tbl:tab_werr_summary summarises, to the best of our knowledge, all prior works for which synthetic-only TTS-for-ASR training was conducted in terms of WERR and relative improvement gained from combining real and synthetic speech. ASR models trained solely on synthetic speech consistently yield significantly higher WERs compared to those trained on real speech, with WERR values generally ranging from approximately 9 down to 1.7. As illustrated in @fig:fig_werr_trajectory, progress appears to plateau. 

#figure(
image("../figures/5/tts-for-asr.png", width: 110%),
caption: [Development of WERR over time achieved by prior work #linebreak()summarised in @tbl:tab_werr_summary.],
placement: top,
) <fig_werr_trajectory>

#pagebreak() For context, Tacotron 2 @shen_natural_2018, the architecture used in the earlier TTS-for-ASR works @li_synthaug_2018 @rosenberg_speechaug_2019, achieved a Mean Opinion Score (MOS) of 4.53 compared to 4.58 for ground truth speech, a perceptual quality ratio of only 1.01. The persistence of a substantial utility gap, even with TTS systems that are subjectively near-indistinguishable from real speech @rossenbach_model_2024, motivates a deeper investigation into the underlying causes of this discrepancy.


=== Baseline Experiments and Analysis

To establish a consistent baseline and explore the properties of the synthetic-real gap in a controlled manner, we conducted a series of our own experiments. This allows us to verify the findings of prior works and to perform targeted ablations.

==== Experimental Setup <05_hybrid>

Our baseline TTS model is a multi-speaker FastSpeech 2 @ren_fastspeech_2021 model trained for 40 epochs on the `train-clean-360` split of the LibriTTS corpus @zen_libritts_2019. For the ASR training data, we select a 10-hour subset of LibriTTS, ensuring no transcript overlap with the TTS training set, as detailed in the controlled setup in @05_setup. We use the trained TTS model to generate a synthetic version of this 10-hour set. The ASR system is a 6-layer hybrid HMM-TDNN model with a hidden unit size of 512, trained for 4 epochs using the Lattice-Free Maximum Mutual Information (LF-MMI) objective @hadian_lfmmi_2018. We train two separate ASR models, one on the real 10-hour dataset and one on its synthetic equivalent. Both are evaluated on the standard LibriSpeech `test-clean` set. This setup provides a baseline WERR and allows for cross-evaluation between models and data types.

==== Recognisability and Regularity of Synthetic Speech

The results of our baseline experiments, summarised in @tbl:tab_cross_ttsasr, confirm the existence of a significant distributional gap and reveal counter-intuitive phenomena related to the regularity of synthetic speech. The baseline WERR is 3.66, which is comparable to the values observed in prior work, as shown in @tbl:tab_werr_summary. This result indicates that the ASR model trained on synthetic data is over three times less effective than the one trained on real data.

Further analysis reveals the effects of synthetic speech's inherent regularity. When the ASR model trained on real speech is evaluated on synthetic speech, its WER is surprisingly low at 11.4%, which is better than its performance on real evaluation data (13.3%). This occurs despite the synthetic speech being out-of-distribution for a model trained solely on real speech. The explanation lies in the reduced complexity of synthetic speech; it tends to be cleaner, more consistent, and exhibits an oversmoothed quality @ren_revisiting_2022. An ASR model accustomed to the variability of real speech finds this "idealised" input easier to transcribe.

Conversely, when an ASR model is trained and evaluated exclusively on synthetic speech, its WER plummets to an extremely low value of 3.0%. This dramatic improvement suggests the model is not merely recognising speech but is effectively learning to transcribe the unique and highly consistent acoustic patterns of the TTS system. While this leads to excellent in-domain performance, it severely limits the model's ability to generalise to the unpredictable variability of real human speech. This highlights that for TTS-for-ASR to be truly effective, synthetic speech must emulate the full distributional diversity of real speech, not just present a simplified, recognisable acoustic pattern. These cross-evaluation experiments also allow for an examination of WERR's symmetricity. While $"WERR"(tilde(D),D)=3.66$, the reverse, $"WERR"(D,tilde(D))=3.75$, is similar but not identical, a discrepancy attributable to the inherent stochasticity of model training.

#figure(
table(
columns: (1fr, 2fr, 1fr),
align: center,
toprule(),
table.header([*Training Data*], [*Evaluation Data*], [*WER #sym.arrow.b*]),
toprule(),
[Real], [Real], [13.3 ± 0.29],
[Synthetic], [Real], [48.6 ± 0.43],
[Real], [Synthetic], [11.4 ± 0.69],
[Synthetic], [Synthetic], [3.0 ± 0.02],
toprule(),
[Real], [HiFi-GAN Resynthesised], [13.4 ± 0.24],
),
caption: "Results when training on synthetic and evaluating on real and vice versa.",
placement: top
) <tab_cross_ttsasr>

==== The Role of the Vocoder

While the choice of vocoder is critical for the perceptual quality of TTS output, its impact on the TTS-for-ASR utility gap appears to be minimal. This can be quantified by computing the WERR on re-synthesised data, where a vocoder representation is extracted from original audio and then used to synthesise a new waveform. An experiment by #citep(<rossenbach_duration_2023>) using the algorithmic Griffin-Lim vocoder resulted in a WERR of 1.04, suggesting the vocoder step accounts for at most 5% of the performance gap. Our own preliminary experiments using a modern, high-fidelity neural vocoder, Hifi-GAN @kong_hifigan_2020, yielded a WERR close to 1.00, as can be seen in @tbl:tab_cross_ttsasr. This indicates no statistically significant difference in WER between ASR models trained on original real data and those trained on data that was converted to Mel spectrograms and then back to waveforms. This result is not unexpected, as Mel spectrograms are a common input representation for ASR systems themselves @jurafsky_slp_2008. This implies that the lossy-reconstructible transformation to a Mel spectrogram preserves most of the information relevant to ASR, and the final vocoding step, if performed with sufficient fidelity, does not significantly contribute to the synthetic-real gap. The primary source of the gap must therefore lie in the acoustic model that generates the spectrograms, not the vocoder that inverts them. This chapter has established and quantified the synthetic-real gap in ASR training using the WERR heuristic. We have demonstrated through a review of prior work and our own baseline experiments that a significant performance disparity persists, and we have ruled out the vocoder as a primary cause. The evidence points to a more fundamental issue: the acoustic models of current TTS systems fail to capture the full distributional diversity of real speech.

This conclusion motivates the central question we address in subsequent chapters. #emph[If a lack of diversity is the core problem, can the synthetic-real gap be narrowed by explicitly introducing more variability into the synthetic data?] The next chapter addresses this question directly by exploring various methodologies for enhancing synthetic speech diversity, including explicit attribute conditioning and post-generation data augmentation. We systematically evaluate whether these techniques can produce synthetic speech that is not only natural-sounding but also more distributionally complete, with the goal of demonstrably reducing the WERR. Subsequently, we investigate the scaling properties of these approaches to understand the limits of bridging the gap through data volume alone.