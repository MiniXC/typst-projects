#import "../abbr.typ"
#import "../quote.typ": *
#import "../comic.typ"
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

== TTSDS Across Domains and Languages <09_dist>

#q(
[#citep(<moller_quality_2009>)],
[#emph("Quality prediction for synthesized speech: Comparison of approaches")],
[Each time a new TTS system is developed which potentially introduces new types of degradations, the validity and reliability of such a prediction algorithm has to be tested anew.]
)

In the preceding chapter, we introduced the Text-to-Speech Distribution Score (TTSDS), a factorised framework for quantifying the distributional dissimilarity between real and synthetic speech. The initial validation demonstrated its potential, showing strong correlations with human judgments across several decades of speech synthesis technology. However, that validation also highlighted several critical limitations. The reliance on pre-existing subjective datasets, such as the crowdsourced TTS Arena, introduced uncertainty regarding the listening conditions and experimental design; for instance, the comparison tests did not control for speaker identity, a potential confounding variable. Moreover, the evaluation was confined to a single domain of clean, read English speech, leaving the metric's robustness to more challenging, real-world conditions unevaluated. Finally, a more systematic comparison against a broader set of contemporary systems was required to fully assess its capabilities. These limitations directly inform the next stage of our investigation and lead to the following set of research questions for this chapter:

#emph[
How can the TTSDS framework be enhanced to achieve greater robustness and generalisability across diverse acoustic domains, speaking styles, and multiple languages?

Can this metric demonstrate consistent correlation with human perception when validated against a subjective dataset spanning various domains?

How does the performance of this distributional metric compare to a comprehensive suite of existing objective evaluation methods when benchmarked on the task of predicting human quality judgments for state-of-the-art TTS systems?

]
This chapter addresses these questions by presenting TTSDS2, an expansion of the original framework. The primary contributions herein are the targeted improvements to the metric's feature set and scoring mechanism to increase its robustness for application across diverse domains and languages. To validate these enhancements, we present a comprehensive evaluation across four distinct English-language domains and extend its application to 14 languages. A crucial component of this validation is the creation and public release of a large-scale listening test dataset, containing over 11,000 subjective ratings collected under controlled conditions. We also establish an automated, recurring pipeline and benchmark for the ongoing evaluation of #abbr.a[TTS] systems, ensuring up-to-date and uncontaminated comparisons. Finally, we conduct a systematic comparison of TTSDS2 against 16 other objective metrics, demonstrating its superior and more consistent correlation with human perception. An overview of the TTSDS2 pipeline and validation process is shown in @fig:fig_ttsds_pipeline. These contributions were covered in the following publication:

- #cite(<minixhofer_ttsds2_2025>, form: "full", style: "iso-690-author-date").

#figure(
  image("../figures/9/neurips_ttsds.png", width: 100%),
  placement: top,
  caption: [Overview of TTSDS2 -- both public (YouTube, LibriVox) and academic (Linguistic Data Consortium) sources were used for validating TTSDS2 as a metric, by showing robust correlations with listening test results across domains. A multilingual YouTube dataset is automatically scraped and synthesised quarterly, and with TTSDS2, provides ranking of TTS.],
) <fig_ttsds_pipeline>

=== Background on Speech Evaluation

To establish the context for a new distributional metric, it is necessary to first review the existing landscape of synthetic speech evaluation, which is broadly divided into subjective and objective methods.

==== Subjective Listening Tests

Subjective tests are the gold standard for synthetic speech evaluation as they directly measure human perception. However, their results are inherently variable and depend heavily on the experimental design. As discussed in @03_subjective, recent #abbr.a[TTS] work has reported near-parity subjective scores under particular protocols, but such results are highly protocol-dependent and should not be over-interpreted as universal "human parity".

A general best practice is to clearly define the experimental parameters before conducting a test, as recommended in guidelines for #abbr.a[TTS] evaluation @wester_listeners_2015@cooper_review_2024. This includes defining the listener pool, typically native speakers of the language being tested; the setting, preferably a quiet lab environment with headphones; and the specific instructions given to the listeners, which should clearly define the attribute being rated, such as #emph[naturalness]. The design of the stimuli presentation, such as the number of samples per page and whether re-listening is allowed, also influences the results @wells_bws_2024. Despite these best practices, subjective tests face challenges of standardisation, making results difficult to compare across studies, and they are resource-intensive.

#figure(
  image("../figures/9/mos_instructions.png", width: 75%),
  caption: [Initial instructions given for listening tests.],
  placement: top,
) <fig_mos_instructions_init>

The most common methodologies are the Mean Opinion Score (#abbr.a[MOS]), Comparison #abbr.a[MOS] (#abbr.a[CMOS]), and Speaker Similarity #abbr.a[MOS] (#abbr.a[SMOS]). For a #abbr.a[MOS] test, listeners rate isolated audio samples on a 5-point scale from 1 (bad) to 5 (excellent), as shown in @fig:fig_mos_instructions. #abbr.a[CMOS] tests present listeners with two samples, A and B, and ask them to rate their relative naturalness on a 7-point scale from -3 (A is much worse than B) to +3 (A is much better than B), which can be seen in @fig:fig_cmos_instructions. This is particularly useful for fine-grained comparisons when absolute scores may saturate. #abbr.a[SMOS] is used for voice cloning evaluation and operates similarly to #abbr.a[CMOS], but listeners rate the speaker similarity between two clips on a 5-point scale, as illustrated in @fig:fig_smos_instructions. We provide the full text for initial instructions given to listeners in @fig:fig_mos_instructions_init, and the survey can be accessed at #underline[#link("http://ttsdsbenchmark.com/survey",[ttsdsbenchmark.com/survey])].

#figure(
  image("../figures/9/mos.png", width: 75%),
  caption: [Interface for Mean Opinion Score (MOS) listening tests.],
  placement: top,
) <fig_mos_instructions>

#figure(
  image("../figures/9/cmos.png", width: 75%),
  caption: [Interface for Comparison MOS (CMOS) listening tests.],
  placement: top,
) <fig_cmos_instructions>

#figure(
  image("../figures/9/smos.png", width: 75%),
  caption: [Interface for Speaker Similarity MOS (SMOS) listening tests.],
  placement: top,
) <fig_smos_instructions>

==== Objective Metrics

Due to the resource-intensive nature of subjective tests, objective metrics are frequently used, especially for experimental iteration. These metrics can be categorised into several families. Signal-based reference metrics are the oldest group and consist of intrusive metrics that compare a synthetic utterance to a time-aligned real reference. Representatives include Perceptual Evaluation of Speech Quality (#abbr.l[PESQ]) @rix_pesq_2001, Short-Time Objective Intelligibility (#abbr.l[STOI]) @taal_stoi_2011, and Mel-Cepstral Distortion (#abbr.l[MCD]) @kominek_mcd_2008. These were often designed for telecommunications or enhancement scenarios and can struggle with modern #abbr.a[TTS] systems.

Model-based metrics involve training a neural network to predict subjective scores directly from an audio signal. This approach was introduced by MOSNet @lo_mosnet_2019 and has been followed by systems like UTMOS @saeki_utmos_2022, NISQA-MOS @mittag_nisqa_2021, and SQUIM-MOS @kumar_torchaudio-squim_2023. While effective in-domain, their performance often degrades on out-of-domain data, and they need to be continually re-validated as #abbr.a[TTS] technology evolves. Other metrics focus on specific attributes. Intelligibility is often proxied using Word Error Rate (#abbr.l[WER]) from an #abbr.a[ASR] system. Speaker similarity is measured by the cosine similarity of speaker embeddings, such as d-vectors @wan_generalized_2018 or from systems like ECAPA-TDNN @desplanques_ecapa_2020.

Finally, distributional metrics, inspired by the image domain's Fréchet Inception Distance (#abbr.l[FID]) @heusel_fid_2017, measure the distance between entire corpora of synthetic and real speech. Fréchet Audio Distance (#abbr.l[FAD]) @kilgour_fad_2019 adapts this principle for audio. These metrics do not require corresponding samples but often necessitate thousands of utterances, which may have limited their widespread adoption. This work builds upon this distributional approach.

=== TTSDS Robustness Improvements

Our methodology is founded on the principle of quantifying distributional dissimilarity. Here, we introduce the changes made to TTSDS to increase the robustness of the metric and its applicability to a wide range of domains.

The TTSDS framework provides a factorised evaluation of synthetic speech by measuring distributional distances across several perceptually motivated attributes.

The initial version, TTSDS, was validated primarily on clean, English audiobook speech and delineated five factors: #smallcaps[Generic], #smallcaps[Ambient], #smallcaps[Intelligibility], #smallcaps[Prosody], and #smallcaps[Speaker]. These were assessed using features like #abbr.a[SSL] embeddings, noise correlates, #abbr.l[WER], and speaker embeddings. While effective, its robustness to diverse conditions was limited. To verify this robustness prior to conducting correlations with subjective evaluation, we analysed the TTSDS scores of each potential factor on real data. Specifically, we split each real dataset into two halves and compared the score between them. Ideally, real speech compared against itself should yield a near-perfect score. Factors that scored below 95 on average or exhibited high standard deviation between datasets were excluded from the final metric. This factor selection process was finalised before running any correlation experiments.


#figure(
  table(
    columns: (auto, 0.6fr), // First column adjusts to content, second column takes 60% of remaining space
    stroke: (top: 1pt + black, bottom: 1pt + black), // Thicker top and bottom rules
    align: (left, left), // Left-align content in both columns

    // Table header row
    [#strong[Factor]], [#strong[Features]],

    // Generic Factor content
    [#smallcaps[Generic]], [
      WavLM #cite(<chen_wavlm_2022>) layer 11 activations (actv.) \
      HuBERT #cite(<hsu_hubert_2021>) (base) layer 7 actv. \
      wav2vec 2.0 #cite(<baevski_wav2vec_2020>) (base) layer 8 actv.
    ],

    // Speaker Factor content
    [#smallcaps[Speaker]], [
      d‑Vector embeddings #cite(<wan_generalized_2018>) \
      WeSpeaker embeddings #cite(<wang_wespeaker_2023>)
    ],

    // Prosody Factor content
    [#smallcaps[Prosody]], [
      Mean F0 (PyWORLD) #cite(<morise_world_2016>) \
      HuBERT #cite(<hsu_hubert_2021>) speaking‑rate \
      Allosaurus #cite(<li_allosaurus_2020>) speaking-rate \
      MPM (Ours)
    ],

    // Intelligibility Factor content
    [#smallcaps[Intelligibility]], [
      wav2vec 2.0 #cite(<baevski_wav2vec_2020>) ASR head actv. \
      Whisper #cite(<radford_robust_2023>) (small) ASR head actv.
    ],

    // General Factor content
    [#smallcaps[General]], [
      SSL Representations \
      HuBERT #cite(<hsu_hubert_2021>) \
      wav2vec 2.0 #cite(<baevski_wav2vec_2020>)
    ],
  ),
  caption: [Revised feature set used for each TTSDS2 factor.],
) <ttsds2_features>

TTSDS2 was developed to enhance robustness across domains and to add multilingual capabilities. This involved several key refinements to the feature set. For the #smallcaps[Intelligibility] factor, we moved from using direct #abbr.l[WER] values, to using the final-layer activations of the #abbr.a[ASR] model's head -- this comes with the additional advantage of not requiring the original transcriptions in this case. For the #smallcaps[Prosody] factor, the original feature based on the length of consecutive HuBERT tokens was replaced with an utterance-level speaking rate metric, which proved to be a more stable correlate of speech rhythm. The #smallcaps[Generic] factor was augmented by adding WavLM embeddings to the existing HuBERT and wav2vec 2.0 features to increase representational diversity. For multilingual support, the English-centric models were replaced with their multilingual counterparts, mHuBERT-147 and XLSR-53. The full feature set for TTSDS2 is detailed in @ttsds2_features. As in @08_dist[Chapter], the scoring mechanism for each feature computes a normalised similarity score yielding a value between 0 and 100. The aforementioned changes were informed by splitting datasets with solely real speech across the domains we introduce in @09_domains and aiming for high similarity (scores $>90$) of the speech with itself.


=== Experimental Validation <09_domains>


To validate the TTSDS framework, we conducted a large-scale experimental evaluation. This involved collecting new subjective ratings for a wide range of modern #abbr.a[TTS] systems across several challenging domains and comparing these human judgments against the scores produced by TTSDS2 and a comprehensive set of 16 other objective metrics.

==== Datasets and Systems for Evaluation 

The evaluation was designed to test the robustness of the metrics beyond clean, read speech. We constructed four distinct English-language datasets, each targeting a different acoustic or stylistic condition:
- #smallcaps[Clean]: This dataset served as the baseline and was sourced from the test split of the LibriTTS corpus @zen_libritts_2019. It contains high-quality, single-speaker audiobook recordings that have been filtered for a high signal-to-noise ratio.
- #smallcaps[Noisy]: To test robustness to environmental noise, this dataset was created by scraping LibriVox recordings from 2025 without applying any #abbr.a[SNR] filtering, thereby retaining the natural ambient noise present in the original recordings.
- #smallcaps[Wild]: To assess performance on conversational and in-the-wild speech, this dataset was compiled from recently uploaded YouTube videos. Utterances were extracted using Whisper diarisation @clapa_WhisperSpeech_2024>, capturing a wide variety of speaking styles, accents, and recording conditions, inspired by the Emilia dataset @he_emilia_2024.
- #smallcaps[Kids]: To evaluate generalisation to a different speaker population, this dataset was created from a subset of the My Science Tutor Corpus @pradhan_my_2024, which contains conversational speech from children interacting with a virtual tutor.

// Helper for superscript citations, mimicking the LaTeX \supercite command
#let supercite(ref_key) = {
  linebreak()
  text(0.7em, h(0.1em) + [*#cite(ref_key)*])
}

#let supercite2(ref_key1,ref_key2) = {
  linebreak()
  text(0.7em, h(0.1em) + [*#cite(ref_key1)#cite(ref_key2)*])
}

// Convenience macros for symbols in text mode as defined in the original LaTeX
#let OK = sym.checkmark
#let parityeq = "~" // near‑parity
#let paritygt = ">"     // better than GT
#let paritylt = "<"     // worse than GT

#context[
  #set text(size: 9pt)
#figure(
  placement: top,
  table(
    columns: (auto, .5fr, .7fr, .7fr, .7fr, 2fr), // Define 6 columns. Last column (ID) takes remaining space.
    
    // Top rule from booktabs
    toprule(),
    
    // Header row
    [#strong[System]], [#strong[Year]], [#strong[Obj.]], [#strong[Subj.]], [#strong[Parity]], [#strong[ID on `replicate.com`]],
    
    // Mid rule after header
    midrule(),

    // Content rows
    [Bark#supercite2(<bark_2023>,<BarkVC_2023>)], [2023], [], [], [], [#link("https://replicate.com/ttsds/e2")[`ttsds/e2`]],
    [†\*E2‑TTS#supercite2(<eskimez_e2_2024>,<chen_f5_2024>)], [2024], [#OK], [#OK], [#parityeq], [#link("https://replicate.com/ttsds/bark")[`ttsds/bark`]],
    [†F5‑TTS#supercite(<chen_f5_2024>)], [2024], [#OK], [#OK], [#paritygt], [#link("https://replicate.com/ttsds/f5")[`ttsds/f5`]],
    [FishSpeech 1.5#supercite(<liao_fishspeech_2024>)], [2024], [], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/fishspeech_1_5")[`ttsds/fishspeech_1_5`]],
    [GPT‑SoVITS~v2#supercite(<RVCBoss_gptsovits_2024>)], [2024], [], [], [], [#link("https://replicate.com/ttsds/gptsovits_2")[`ttsds/gptsovits_2`]],
    [†HierSpeech++~1.1#supercite(<lee_hierspeechpp_2023>)], [2023], [#OK], [#OK], [#parityeq], [#link("https://replicate.com/ttsds/hierspeechpp_1_1")[`ttsds/hierspeechpp_1_1`]],
    [†MaskGCT#supercite2(<wang_maskgct_2024>,<zhang_amphion_2024>)], [2024], [#OK], [#OK], [#paritygt], [#link("https://replicate.com/ttsds/maskgct")[`ttsds/maskgct`]],
    [MetaVoice‑1B#supercite(<sharma_metavoice_2024>)], [2024], [], [], [], [#link("https://replicate.com/ttsds/metavoice")[`ttsds/metavoice`]],
    [†\*NaturalSpeech~2#supercite2(<shen_natural_2018>,<zhang_amphion_2024>)], [2023], [#OK], [#OK], [#parityeq], [#link("https://replicate.com/ttsds/amphion_naturalspeech2")[`ttsds/amphion_naturalspeech2`]],
    [OpenVoice~v2#supercite(<qin_openvoice_2023>)], [2024], [], [], [], [#link("https://replicate.com/ttsds/openvoice2")[`ttsds/openvoice2`]],
    [†\*ParlerTTS~Large~1.0#supercite2(<lyth_parler_2024>,<lacombe_parlertts_2024>)], [2024], [#OK], [#OK], [#paritygt], [#link("https://replicate.com/ttsds/parlertts_large_1_0")[`ttsds/parlertts_large_1_0`]],
    [†Pheme#supercite(<budzianowski_pheme_2024>)], [2024], [#OK], [], [], [#link("https://replicate.com/ttsds/pheme")[`ttsds/pheme`]],
    [†SpeechT5#supercite(<ao_speecht5_2022>)], [2022], [], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/speecht5")[`ttsds/speecht5`]],
    [†StyleTTS~2#supercite(<li_styletts_2023>)], [2023], [#OK], [#OK], [#paritygt], [#link("https://replicate.com/ttsds/styletts2")[`ttsds/styletts2`]],
    [TorToiSe#supercite(<betker_tortoise_2023>)], [2022], [], [], [], [#link("https://replicate.com/ttsds/tortoise")[`ttsds/tortoise`]],
    [†\*VALL‑E#supercite2(<wang_valle_2023>,<zhang_amphion_2024>)], [2024], [#OK], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/amphion_valle")[`ttsds/amphion_valle`]],
    [†Vevo#supercite(<zhang_vevo_2025>)], [2024], [#OK], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/amphion_vevo")[`ttsds/amphion_vevo`]],
    [†VoiceCraft‑830M#supercite(<peng_voicecraft_2024>)], [2024], [#OK], [#OK], [#paritylt], [#link("https://replicate.com/ttsds/voicecraft")[`ttsds/voicecraft`]],
    [WhisperSpeech~Medium#supercite(<clapa_WhisperSpeech_2024>)], [2024], [], [], [], [#link("https://replicate.com/ttsds/whisperspeech")[`ttsds/whisperspeech`]],
    [†XTTS‑v1#supercite(<casanova_xtts_2024>)], [2023], [#OK], [], [], [#link("https://replicate.com/ttsds/xtts_1")[`ttsds/xtts_1`]],
    
    // Bottom rule from booktabs
    bottomrule(),
  ),
  caption: [
    Open-source #abbr.a[TTS] systems, prior evaluation, and results for each system relative to ground‑truth (GT) speech:
    $dagger$ = accompanied by publication;
    $ast$ = third‑party implementation; \
    Parity column: coarse summary based on reported #abbr.a[MOS]/#abbr.a[CMOS] relative to GT in the source evaluation. #parityeq indicates near‑parity (often within confidence intervals or without significant difference when reported); #paritygt indicates higher reported scores than GT under that specific protocol (not a general claim of being "better than human"), and #paritylt indicates lower reported scores.
  ],
) <tab_ttsds2_systems>
]

For evaluation, we selected 20 open-source, open-weight #abbr.a[TTS] systems released between 2022 and 2024, as detailed in @tbl:tab_ttsds2_systems. These systems were chosen based on their ability to perform voice cloning from a speaker reference utterance and a text transcript. The selection covers a wide range of modern architectures. For each system, we used the most recent publicly available checkpoint before January 1, 2025.

==== Subjective Data Collection and Ethical Considerations

A critical component of this validation was the collection of a new, large-scale set of subjective human judgments. This process was guided by rigorous logistical and ethical considerations.

#figure(
  image("../figures/9/real_fake_tts.png", width: 100%),
  placement: top,
  caption: [Ethical TTS evaluation through resynthesis. A reference utterance from a speaker is used to clone their voice, which then synthesises the text from a separate, distinct utterance by the same speaker. No novel content is generated.],
) <fig_ethical_tts>

Our data collection methodology was designed with a strong ethical framework to protect speaker privacy and prevent the misuse of voice cloning technology. As illustrated in @fig:fig_ethical_tts, we employed a two-utterance-per-speaker approach. For each of the 60 speakers selected per dataset, one utterance served as the reference to capture the vocal identity for the #abbr.a[TTS] system. The text content from a second, entirely distinct utterance from the same speaker was then used as the input for synthesis. This ensures that the generated speech, while in the target speaker's voice, only ever contains words that the speaker had actually spoken in a separate context, mitigating the risk of creating deepfakes with novel, unattributed content. This study was certified according to the Informatics Research Ethics Process, reference number 112246.

We recruited 200 annotators via the Prolific platform to provide ratings. Participants were screened to be native English speakers from the UK or US and were required to use headphones in a quiet environment. Each participant was assigned to one of the four datasets. The listening test, administered via Qualtrics, was structured into three parts, evaluating #abbr.a[MOS], #abbr.a[CMOS], and #abbr.a[SMOS], with the order randomised for each participant. We collected a total of 11,846 anonymised ratings, which have been made publicly available.

==== Compared Objective Metrics

To contextualise the performance of TTSDS2, we compared it against 16 other publicly available objective metrics using the VERSA evaluation toolkit @shi_versa_2024. This comparison set included signal-based metrics (#abbr.l[STOI], #abbr.l[PESQ], #abbr.l[MCD]), several model-based #abbr.a[MOS] predictors (UTMOS, UTMOSv2, NISQA-MOS, DNSMOS, SQUIM-MOS), speaker similarity metrics based on different embeddings (X-Vector, RawNet3, ECAPA-TDNN), distributional metrics (#abbr.l[FAD]), and multi-dimensional perceptual metrics (Audiobox Aesthetics sub-scores). These sub-scores consist of Content Enjoyment (AE-CE), Content Usefulness (AE-CU), and Production Quality (AE-PQ) @tjandra_meta_2025. This comprehensive set allows for a thorough analysis of the current state of objective evaluation for high-quality synthetic speech. As in @08_exp, 100 reference utterances were used.

=== Results and Discussion

// Define colors as per LaTeX original
#let negstrong = rgb("#F3C2C1") // –1 … –0.5
#let negweak = rgb("#F3E3E4")  // –0.5 … 0
#let posweak = rgb("#E2EAD5")  //  0 … 0.5
#let posstrong = rgb("#B8D78F") //  0.5 … 1

// Define a function for colored cells, with optional bolding
#let scorecell(value_str, bold: false) = {
  let value = float(value_str)
  let content = if bold { strong(value_str) } else { value_str }
  let color
  if value < -0.5 { color = negstrong }
  else if value < 0.0 { color = negweak }
  else if value < 0.5 { color = posweak }
  else { color = posstrong }
  table.cell(fill: color, content) // Added padding for visual spacing
}

#context[
  #show table.cell: c => {
    return text(10pt, c)
  }
  #figure(
    table(
      columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
      table.cell(rowspan: 2, [#strong[Metric]]), table.cell(colspan: 3, align: center, [#strong[Clean]]), table.cell(colspan: 3, align: center, [#strong[Noisy]]), table.cell(colspan: 3, align: center, [#strong[Wild]]), table.cell(colspan: 3, align: center, [#strong[Kids]]),
      table.hline(start: 1),
      [#smallcaps[mos]], [#smallcaps[cmos]], [#smallcaps[smos]], [#smallcaps[mos]], [#smallcaps[cmos]], [#smallcaps[smos]], [#smallcaps[mos]], [#smallcaps[cmos]], [#smallcaps[smos]], [#smallcaps[mos]], [#smallcaps[cmos]], [#smallcaps[smos]],
      [TTSDS2 (Ours)], scorecell("0.75", bold: true), scorecell("0.69", bold: true), scorecell("0.73", bold: true), scorecell("0.59"), scorecell("0.54"), scorecell("0.71"), scorecell("0.75"), scorecell("0.71"), scorecell("0.75"), scorecell("0.61"), scorecell("0.50"), scorecell("0.70"),
      [TTSDS (Ours)], scorecell("0.60"), scorecell("0.62"), scorecell("0.52"), scorecell("0.49"), scorecell("0.61"), scorecell("0.66"), scorecell("0.67"), scorecell("0.57"), scorecell("0.67"), scorecell("0.70"), scorecell("0.52"), scorecell("0.60"),
      [X‑Vector], scorecell("0.46"), scorecell("0.42"), scorecell("0.56"), scorecell("0.40"), scorecell("0.29"), scorecell("0.77"), scorecell("0.82", bold: true), scorecell("0.82", bold: true), scorecell("0.62"), scorecell("0.70"), scorecell("0.57"), scorecell("0.75", bold: true),
      [RawNet3], scorecell("0.36"), scorecell("0.26"), scorecell("0.52"), scorecell("0.44"), scorecell("0.37"), scorecell("0.82", bold: true), scorecell("0.85", bold: true), scorecell("0.80"), scorecell("0.64"), scorecell("0.73", bold: true), scorecell("0.61", bold: true), scorecell("0.77", bold: true),
      [SQUIM], scorecell("0.68"), scorecell("0.46"), scorecell("0.37"), scorecell("0.48"), scorecell("0.48"), scorecell("0.60"), scorecell("0.62"), scorecell("0.75"), scorecell("0.79", bold: true), scorecell("0.57"), scorecell("0.55"), scorecell("0.45"),
      [ECAPA‑TDNN], scorecell("0.36"), scorecell("0.29"), scorecell("0.47"), scorecell("0.29"), scorecell("0.22"), scorecell("0.72"), scorecell("0.81"), scorecell("0.78"), scorecell("0.58"), scorecell("0.69"), scorecell("0.60"), scorecell("0.72"),
      [DNSMOS], scorecell("0.41"), scorecell("0.37"), scorecell("0.22"), scorecell("0.57"), scorecell("0.36"), scorecell("0.22"), scorecell("0.35"), scorecell("0.28"), scorecell("0.03"), scorecell("0.31"), scorecell("0.10"), scorecell("0.28"),
      [AE-CE], scorecell("0.60"), scorecell("0.46"), scorecell("0.32"), scorecell("0.58"), scorecell("0.53"), scorecell("0.21"), scorecell("0.19"), scorecell("0.10"), scorecell("0.11"), scorecell("-0.02"), scorecell("-0.12"), scorecell("-0.10"),
      [AE-CU], scorecell("0.49"), scorecell("0.37"), scorecell("0.30"), scorecell("0.60", bold: true), scorecell("0.58", bold: true), scorecell("0.13"), scorecell("0.35"), scorecell("0.24"), scorecell("0.22"), scorecell("-0.09"), scorecell("-0.21"), scorecell("-0.13"),
      [AE-PQ], scorecell("0.49"), scorecell("0.33"), scorecell("0.21"), scorecell("0.55"), scorecell("0.48"), scorecell("0.04"), scorecell("0.21"), scorecell("0.16"), scorecell("0.12"), scorecell("0.03"), scorecell("-0.08"), scorecell("-0.05"),
      [UTMOSv2], scorecell("0.39"), scorecell("0.25"), scorecell("0.09"), scorecell("0.34"), scorecell("0.36"), scorecell("0.19"), scorecell("0.16"), scorecell("0.14"), scorecell("-0.04"), scorecell("0.05"), scorecell("0.03"), scorecell("-0.02"),
      [FAD (CLAP)], scorecell("-0.22"), scorecell("0.06"), scorecell("-0.01"), scorecell("0.45"), scorecell("0.30"), scorecell("0.16"), scorecell("-0.03"), scorecell("0.08"), scorecell("0.25"), scorecell("0.12"), scorecell("0.26"), scorecell("0.04"),
      [UTMOS], scorecell("0.51", bold: true), scorecell("0.30"), scorecell("0.31"), scorecell("0.47"), scorecell("0.29"), scorecell("0.00"), scorecell("-0.12"), scorecell("-0.12"), scorecell("-0.26"), scorecell("-0.02"), scorecell("-0.18"), scorecell("-0.04"),
      [STOI], scorecell("-0.11"), scorecell("0.01"), scorecell("0.02"), scorecell("-0.06"), scorecell("0.00"), scorecell("0.19"), scorecell("0.07"), scorecell("0.41"), scorecell("0.24"), scorecell("-0.32"), scorecell("-0.08"), scorecell("0.05"),
      [PESQ], scorecell("0.01"), scorecell("-0.16"), scorecell("0.27"), scorecell("-0.34"), scorecell("0.00"), scorecell("0.07"), scorecell("-0.14"), scorecell("0.01"), scorecell("-0.06"), scorecell("-0.08"), scorecell("-0.04"), scorecell("-0.38"),
      [NISQA], scorecell("0.05"), scorecell("0.00"), scorecell("0.06"), scorecell("0.05"), scorecell("-0.21"), scorecell("-0.53"), scorecell("-0.32"), scorecell("-0.33"), scorecell("-0.64"), scorecell("-0.29"), scorecell("-0.27"), scorecell("-0.46"),
      [MCD], scorecell("-0.46"), scorecell("-0.37"), scorecell("-0.27"), scorecell("-0.45"), scorecell("-0.58"), scorecell("-0.74"), scorecell("-0.33"), scorecell("-0.45"), scorecell("-0.51"), scorecell("-0.31"), scorecell("-0.13"), scorecell("-0.38"),
      [WER], scorecell("-0.19"), scorecell("-0.18"), scorecell("-0.17"), scorecell("-0.11"), scorecell("-0.30"), scorecell("-0.13"), scorecell("-0.28"), scorecell("-0.17"), scorecell("-0.22"), scorecell("-0.45"), scorecell("-0.26"), scorecell("-0.39"),
    ),
    caption: [Spearman rank correlations.#linebreak()Colours: #box(fill: negstrong, rect(width: 5pt, height: 5pt)) –1 … –0.5, #box(fill: negweak, rect(width: 5pt, height: 5pt)) –0.5 … 0, #box(fill: posweak, rect(width: 5pt, height: 5pt)) 0 … 0.5, #box(fill: posstrong, rect(width: 5pt, height: 5pt)) 0.5 … 1.],
    placement: top,
  ) <tab_ttsds2_spearman_correlation>
] 

==== Correlation with Human Judgments

The primary results of our study are summarised in the Spearman rank correlation matrix shown in @tbl:tab_ttsds2_spearman_correlation. TTSDS2 achieves the highest and most consistent performance, with an average correlation of 0.67 across all domains and subjective scores. Crucially, it is the only metric of the 16 compared to maintain a correlation coefficient greater than 0.5 in every single test condition. This indicates a high degree of robustness.

The next best performing category of metrics is speaker similarity, with RawNet3 and X-Vector achieving average correlations of 0.6. Among the #abbr.a[MOS] prediction networks, SQUIM-MOS is the strongest performer with an average correlation of 0.57. However, many other metrics, including several recent #abbr.a[MOS] predictors and the Audiobox Aesthetics sub-scores, show a performance drop when moving from in-domain clean and noisy audiobook data to the more challenging #smallcaps[Wild] and #smallcaps[Kids] domains. This highlights a common issue of overfitting or lack of generalisation in current learned objective metrics.

#figure(
  image("../figures/9/scatterplot_ttsds_combined.png", width: 100%),
  placement: top,
  caption: [Correlation of three representative objective metrics with human MOS across the four datasets. Each colour/marker denotes a domain. Solid line = overall least-squares fit; dashed/dotted lines = domain-specific fits; each with corresponding Pearson $r$.],
) <ttsds_correlations_plot>

The scatter plot in @fig:ttsds_correlations_plot further illustrates these findings. It shows that TTSDS2 scores form a continuous and well-distributed correlation with human #abbr.a[MOS] ratings. In contrast, both SQUIM-MOS and X-Vector exhibit some clustering behaviour, which may suggest that they are less discriminative for systems at the higher end of the quality spectrum or are potentially overfitting to specific system characteristics.

==== Factor-wise and Multilingual Analysis

#figure(
  placement: top,
  table(
    columns: (auto, auto, auto, auto, auto), // First column adjusts to content, remaining 4 columns adjust automatically
    
    // Top rule from booktabs
    toprule(),
    
    // Header row
    [#strong[Dataset]], [#strong[Generic]], [#strong[Speaker]], [#strong[Prosody]], [#strong[Intelligibility]],
    
    // Mid rule after header
    midrule(),

    // Content rows
    [Clean], [0.42], [#strong[0.84]], [0.38], [0.47],
    [Noisy], [0.59], [#strong[0.86]], [0.46], [0.59],
    [Wild], [0.53], [#strong[0.59]], [0.34], [0.58],
    [Kids], [#strong[0.80]], [0.70], [0.60], [0.63],
    
    // Bottom rule from booktabs
    bottomrule(),
  ),
  caption: [Pearson correlation ($r$) between each factor and #abbr.a[MOS].],
) <fig_ttsds2_factor_correlation>

The factorised nature of TTSDS2 provides interpretable insights into which aspects of speech quality are most salient in different domains. As shown in @tbl:fig_ttsds2_factor_correlation, the #smallcaps[Speaker] factor is the most dominant correlate of perceived quality in the #smallcaps[Clean] and #smallcaps[Noisy] audiobook domains. However, in the more complex #smallcaps[Wild] and #smallcaps[Kids] domains, its influence diminishes, and factors like #smallcaps[Intelligibility] and #smallcaps[Generic] similarity become equally important. Notably, the #smallcaps[Prosody] factor shows its highest correlation in the #smallcaps[Kids] dataset, highlighting its utility for evaluating the more expressive and dynamic speech of children.

#figure(
  image("../figures/9/ttsds_boxplot.png", width: 100%),
  placement: top,
  caption: [TTSDS2 scores across 14 languages.],
) <fig_language_scores>

The TTSDS2 framework has also been extended for multilingual and recurring evaluation. A fully automated pipeline, depicted in @fig:fig_ttsds_pipeline, scrapes new data from YouTube quarterly, processes it, and synthesises speech using the latest #abbr.a[TTS] models to maintain an up-to-date benchmark across 14 languages. The multilingual validity of TTSDS2 was confirmed by showing that its distances between ground-truth language datasets correlate significantly with established typological distances from Uriel+ @khan_uriel_2024. Uriel+ is a typological knowledge base that provides vector representations of languages based on syntactic, phonological, and inventory features, allowing for the calculation of linguistic distances between languages.

We find that when comparing the ground truth language datasets to each other using TTSDS2, the scores correlate with the distances with $rho = −0.39$ for regular and $rho = −0.51$ (both significant with $p < 0.05$). The negative correlations are expected since a higher score correlates with a smaller distance. @fig:fig_uriel visualises these relationships using Multidimensional Scaling (MDS) to project the pairwise distances into 2D space, illustrating that TTSDS2 captures linguistic similarities akin to typological classifications.

Additionally, @fig:fig_language_scores shows the distribution of TTSDS2 scores for the available #abbr.a[TTS] systems across the 14 supported languages (with $N$ TTS systems per language). While we do not have subjective ground truth for all these languages to strictly prove robustness, the range of scores combined with the typological correlation suggests the metric differentiates between systems across languages, while the objective ground truth data achieves consistent scores between 92 and 94.

#figure(
  image("../figures/9/uriel.png", width: 100%),
  placement: top,
  caption: [Multidimensional scaling (MDS) distance plots between languages (left to right) for i)
Uriel+ typological distances ii) TTSDS2 without multilingual changes iii) multilingual TTSDS2. The
three closest neighbors of each language are connected via lines.],
) <fig_uriel>



==== Conclusions

Despite its robust performance, TTSDS2 has several limitations. Its computation is more intensive than simpler metrics due to the extraction of multiple complex features. While it correlates strongly with human perception, it is not a replacement for subjective evaluation, as it cannot capture the full nuance of the human listening experience. Furthermore, it is not designed to detect certain specific failure modes, such as a model perfectly reproducing a reference transcript instead of the target text. Finally, the current implementation does not evaluate long-form speech generation, a growing area of interest in #abbr.a[TTS] research.

While the primary focus of this work is #abbr.a[TTS], the TTSDS framework is largely task-agnostic and could be applied to Voice Conversion (VC). However, careful construction of the datasets is required to ensure that the reference dataset matches the lexical content of the synthetic output. While TTSDS could potentially function with mismatched lexical content (as would be necessary for cross-lingual VC), we have restricted our validation here to matched content to control for as many variables as possible.

This chapter has detailed the methodology, development, and extensive validation of the TTSDS framework. The robust correlations observed with human judgments across diverse domains and languages establish its utility as a reliable objective metric for assessing and guiding the development of synthetic speech technology. This evaluation framework, together with the TTS-for-ASR analysis, provides a comprehensive set of tools to quantify and understand the persistent gap between real and synthetic speech.