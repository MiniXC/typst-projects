#import "../abbr.typ"
#import "../quote.typ": *

== Synthetic speech evaluation <07_eval>

#q(
  [#citep(<moller_quality_2009>)],
  [#emph("Quality prediction for synthesized speech: Comparison of approaches")],
  [Each time a new TTS system is developed which potentially introduces new types of degradations, the validity and reliability of such a prediction algorithm has to be tested anew.]
)

In this chapter, we outline evaluation methodology for synthetic speech, both #emph[subjective] (determined by raters opinions; potential different outcomes every time) and #emph[objective] (determined by a fixed algorithm, formula or model; same outcome every time with respect to data and parameters).

=== Subjective listening and preference tests <07_subjective>

Here we discuss the most common subjective listening test methodologies and best practices. Subjective tests are the gold standard for synthetic speech evaluation, however, there are drawbacks and trade-offs to any listening test since human behaviour can never be fully anticipated, especially across differing groups of listeners, spans of time, and sets of #abbr.a[TTS] systems.

==== Best practices & drawbacks <07_practices>

Before detailing the individual methods, we will outline best practice according to the literature, as well as commonly misunderstood aspects of listening tests, including their drawbacks.

A general *best practice* is clearly answer the following questions before conducting a test:
#enum(
[
  Who are the *listeners*? Usually, native speakers of the language of the  samples are preferred. Additionally, certain tests require a significant number of different speakers to be used @wester_listeners_2015.
],
[
  What is the *setting*? E.g. #sym.arrow In a lab or online? If possible, requiring use of headphones in a quiet environment is preferred.
],
[
  What are the *instructions*? Recently, #emph[naturalness] is the most commonly evaluated attribute of synthetic speech, and this should be phrased in an understandable way in the instructions.
],
[
  What is the *data*? E.g. #sym.arrow Is the reference or an anchor @schoeffler_mushra_2015 included in the set of samples to be evaluated?
],
[
  What are the concrete *stimuli*? E.g. #sym.arrow How many samples are presented at one time, can listeners re-listen to the same sample? How many points are on the evaluation scale, and are they labeled?
]
)

All these questions should be answered #emph[before] conducting a listening test, and reported in the work to frame the results.

There are also drawbacks to subjective evaluation. We will not go into detail on the advantages and disadvantages of every methodology here, but there are two main drawbacks to consider. Firstly, *lack of standardisation* means there is no standardised framework for most listening test methodologies, beyond the labels and values for particular scales. This means the questions above are answered differently (and often not reported) making it difficult or even impossible to compare results between studies. Subjective listening test results should only ever be compared to results obtained within the same study with the same participants and setup. Additionally, there is the *scale/comparison trade-off* -- many methodologies operate by presenting an ordinal scale to listeners on which they rate recordings on, which are then averaged, however this is not necessarily statistically meaningful. But when instead using a comparison-based task (in which one sample has to be rated over another), many more comparisons are needed @wells_bws_2024.

==== Mean opinion score

#abbr.a[MOS] is the most-commonly used and likely most-criticised listening test methodology @kirkland_mospit_2023. Raters are asked to rate recordings on a scale from 1 to 5 (Bad to Excellent) @ITU_P800 -- however, this is where the standardisation ends. The original recommendation was intended for evaluation of telecommunication systems and therefore asked listeners to rate #emph[quality], while most, but not all @kirkland_mospit_2023, #abbr.a[TTS] evaluations instead ask for #emph[naturalness] -- sometimes the former is referred to as Q-MOS and the latter as N-MOS. There is also no universal agreement for how many stimuli to present at a single time or many of the other questions listed above, and sometimes even the core definition in @ITU_P800 is altered to, for example, alter the number or type of labels @kirkland_mospit_2023.

==== Comparison MOS

This is an A/B-test-inspired variation of #abbr.a[MOS], in which two samples are presented and the listener is asked to compare and rate on a 7-point scale from -3 (A is much better than B) to +3 (A is much worse than B), with 0 indicating they are equivalent in naturalness (or quality). However, the same standardisation problem is present. E.g., sometimes a 13-point scale (from -6 to +6) is employed @li_styletts_2023 and even the name of the test is sometimes referred to as comparative instead of comparison #abbr.a[MOS].

==== Speaker similarity MOS

Sometimes we do not want to evaluate the quality or naturalness of the speech, but how closely the speakers in two samples resemble each other. In this case, #abbr.a[SMOS] is commonly used, which operates in the same way as #abbr.a[CMOS], but instead asks listeners to rate how closely the speakers resemble each other.

=== Objective metrics <07_distances>

Due to the considerable effort and resources required to conduct subjective evaluation, objective metrics are frequently used for TTS evaluation, especially for ablation studies @cooper_review_2024. We group existing metrics into several categories into the following categories, with the function of *distributional* metrics being outlined in further detail in @08_dist[Chapter]. In addition to these families we also distinguish #emph[intrusive] and #emph[non-intrusive] metrics. Intrusive metrics require some ground truth speech of the same speaker as a reference. Non-intrusive metrics are reference-free. When the reference does not need to contain the same lexical content, it is described as #emph[non-matching]. 

==== Signal-based reference metrics

The oldest group consists of intrusive metrics that compare each synthetic utterance to a matching reference. #abbr.l[PESQ] @rix_pesq_2001, #abbr.l[STOI] @taal_stoi_2011 and #abbr.l[MCD] @kominek_mcd_2008 are the best–known representatives. They were mostly designed for telephone or enhancement scenarios rather than TTS, and require access to the ground‑truth waveform.

==== Model-based

To predict scores directly, researchers train neural networks that map a single audio signal to an estimated MOS. #emph[MOSNet] @lo_mosnet_2019 introduced the idea, and was followed by #emph[UTMOS] @saeki_utmos_2022, its SSL‑based successor #emph[UTMOSv2] @baba_t05_2024, and #emph[NISQA‑MOS] @mittag_nisqa_2021. #emph[SQUIM-MOS] @kumar_torchaudio-squim_2023 additionally grounds its prediction by requiring a non-matching reference of the ground truth speech. These methods report in‑domain correlations; however, recent VoiceMOS challenges @huang_voicemos_2024 show that correlation with subjective ratings drops out-of-domain. The drawbacks listed in @07_distances make training these networks challenging, and generalisation to new TTS systems or domains unlikely -- as the quote from @moller_quality_2009 at the beginning of this chapter indicates, each time substantially different TTS systems are introduced, learned metrics will have to be adjusted and verified anew. A recent prediction system goes beyond #abbr.a[MOS] and argues that no single score can capture everything listeners care about. #emph[Audiobox Aesthetics] predicts four axes, Production Quality, Complexity, Enjoyment, and Usefulness, for arbitrary audio @tjandra_meta_2025 -- however, the challenges of learning from subjective labels remain the same.

==== Intelligibility

Often reported are also #abbr.l[WER] and #abbr.l[CER], computed on #abbr.a[ASR] transcripts.

==== Speaker similarity

Analogous to #abbr.a[SMOS], cosine similarity between the speaker embeddings (see @06_speaker) of a reference and target speech is frequently reported.

==== Distributional

Inspired by the image domain's #abbr.l[FID] @heusel_fid_2017, audio researchers proposed measuring entire corpora rather than single files. #abbr.l[FAD] @kilgour_fad_2019 compares embeddings and has since been adapted for TTS @shi_versa_2024. Distributional metrics require a set of references which do not need to correspond to the synthetic data. The authors of these metrics state the need for thousands of samples, which may be why they have not found more widespread adoption. We go into more detail into the workings of these metrics in the next Chapter.