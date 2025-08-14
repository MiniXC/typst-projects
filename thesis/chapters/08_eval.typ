#import "../abbr.typ"
#import "../quote.typ": *

== Synthetic speech evaluation <08_eval>

#q(
[#citep(<moller_quality_2009>)],
[#emph("Quality prediction for synthesized speech: Comparison of approaches")],
[Each time a new TTS system is developed which potentially introduces new types of degradations, the validity and reliability of such a prediction algorithm has to be tested anew.]
)

In this chapter, we outline evaluation methodology for synthetic speech, both #emph[subjective] (determined by raters opinions; potential different outcomes every time) and #emph[objective] (determined by a fixed algorithm, formula or model; same outcome every time with respect to data and parameters). As #abbr.a[TTS] systems have advanced to produce audio close to real speech @minixhofer_ttsds_2024 @minixhofer_ttsds2_2025, evaluation methods must evolve to capture nuances beyond basic perceptual quality. Subjective methods rely on human judgment to assess attributes like naturalness, while objective metrics provide automated, repeatable assessments, often correlating with human perception but facing challenges in generalization @cooper_review_2024. We draw from longitudinal data across TTS systems from 2008 to 2024 to highlight how evaluation priorities have shifted, with factors like prosody gaining prominence in modern systems @minixhofer_ttsds_2024.

=== Subjective listening and preference tests <07_subjective>

Here we discuss the most common subjective listening test methodologies and best practices. Subjective tests are the gold standard for synthetic speech evaluation, however, there are drawbacks and trade-offs to any listening test since human behaviour can never be fully anticipated, especially across differing groups of listeners, spans of time, and sets of #abbr.a[TTS] systems. These tests aim to quantify perceptual qualities that objective metrics may overlook, such as overall naturalness or speaker similarity, but their results are inherently variable due to listener biases and contextual factors @cooper_review_2024. Recent advancements in TTS, where synthetic speech often achieves human parity @chen_vall-e_2024, have made subjective evaluation more challenging, as listeners struggle to distinguish real from synthetic audio @minixhofer_ttsds2_2025.

==== Best practices & drawbacks <08_practices>

Before detailing the individual methods, we will outline best practice according to the literature, as well as commonly misunderstood aspects of listening tests, including their drawbacks.

A general best practice is clearly answer the following questions before conducting a test, as recommended in guidelines for TTS evaluation @wester_listeners_2015 @cooper_review_2024:
#enum(
[
Who are the listeners? Usually, native speakers of the language of the samples are preferred. Additionally, certain tests require a significant number of different speakers to be used @wester_listeners_2015, to mitigate biases from individual perceptual differences or fatigue effects.
],
[
What is the setting? E.g. #sym.arrow In a lab or online? If possible, requiring use of headphones in a quiet environment is preferred, to minimize external noise interference and ensure consistent audio playback @huang_voicemos_2024.
],
[
What are the instructions? Recently, #emph[naturalness] is the most commonly evaluated attribute of synthetic speech, and this should be phrased in an understandable way in the instructions, such as specifying that it includes prosody, intelligibility, and absence of artifacts @minixhofer_ttsds_2024.
],
[
What is the data? E.g. #sym.arrow Is the reference or an anchor @schoeffler_mushra_2015 included in the set of samples to be evaluated? Anchors, like degraded audio samples, can calibrate listener expectations and improve rating consistency @cooper_review_2024.
],
[
What are the concrete stimuli? E.g. #sym.arrow How many samples are presented at one time, can listeners re-listen to the same sample? How many points are on the evaluation scale, and are they labeled? Allowing re-listening reduces memory bias, and labeled scales enhance reliability @wells_bws_2024.
]
)

All these questions should be answered #emph[before] conducting a listening test, and reported in the work to frame the results. This transparency is crucial for reproducibility and comparability, as variations in setup can significantly affect outcomes @kirkland_mospit_2023.

There are also drawbacks to subjective evaluation. We will not go into detail on the advantages and disadvantages of every methodology here, but there are two main drawbacks to consider. Firstly, lack of standardisation means there is no standardised framework for most listening test methodologies, beyond the labels and values for particular scales. This means the questions above are answered differently (and often not reported) making it difficult or even impossible to compare results between studies. Subjective listening test results should only ever be compared to results obtained within the same study with the same participants and setup, as listener demographics, fatigue, and even the time of day can influence ratings @minixhofer_ttsds2_2025. Additionally, there is the scale/comparison trade-off -- many methodologies operate by presenting an ordinal scale to listeners on which they rate recordings on, which are then averaged, however this is not necessarily statistically meaningful due to inter-rater variability and non-linear perception of scales @wells_bws_2024. But when instead using a comparison-based task (in which one sample has to be rated over another), many more comparisons are needed to achieve statistical significance, increasing the test's duration and cost @wells_bws_2024. Furthermore, subjective tests are resource-intensive, prone to biases like the "halo effect" where overall impressions skew specific ratings, and may not generalize across languages or domains @cooper_review_2024.

// add info about qualtrics, listening test, screenshots, etc

==== Mean opinion score <08_mos>

#abbr.a[MOS] is the most-commonly used and likely most-criticised listening test methodology @kirkland_mospit_2023. Raters are asked to rate recordings on a scale from 1 to 5 (Bad to Excellent) @ITU_P800 -- however, this is where the standardisation ends. The original recommendation was intended for evaluation of telecommunication systems and therefore asked listeners to rate #emph[quality], while most, but not all @kirkland_mospit_2023, #abbr.a[TTS] evaluations instead ask for #emph[naturalness] -- sometimes the former is referred to as Q-MOS and the latter as N-MOS. There is also no universal agreement for how many stimuli to present at a single time or many of the other questions listed above, and sometimes even the core definition in @ITU_P800 is altered to, for example, alter the number or type of labels @kirkland_mospit_2023.

In practice, #abbr.a[MOS] involves presenting listeners with isolated audio samples and asking them to rate on the 5-point scale, often with intermediate labels like "Poor" (2), "Fair" (3), and "Good" (4) @ITU_P800. Scores are then averaged across listeners to yield a system-level #abbr.a[MOS]. This method is widely used in TTS challenges, such as the Blizzard Challenge @king_blizzard_2008, where it has been employed to compare systems over time. However, criticisms include its susceptibility to ceiling effects in high-quality TTS systems, where scores cluster near 5, reducing discriminative power @minixhofer_ttsds_2024. Moreover, #abbr.a[MOS] ratings can shift over time as listener expectations evolve; for instance, systems rated highly in 2008 may score lower today due to advancements in TTS @le_maguer_back_2022. Despite these issues, #abbr.a[MOS] remains a benchmark for validating objective metrics, as seen in correlations with TTSDS scores @minixhofer_ttsds_2024.

==== Comparison MOS <08_cmos>

This is an A/B-test-inspired variation of #abbr.a[MOS], in which two samples are presented and the listener is asked to compare and rate on a 7-point scale from -3 (A is much better than B) to +3 (A is much worse than B), with 0 indicating they are equivalent in naturalness (or quality). However, the same standardisation problem is present. E.g., sometimes a 13-point scale (from -6 to +6) is employed @li_styletts_2023 and even the name of the test is sometimes referred to as comparative instead of comparison #abbr.a[MOS].

#abbr.a[CMOS] is particularly useful for fine-grained comparisons between systems, as it directly elicits relative preferences rather than absolute ratings @cooper_review_2024. In TTS evaluations, one sample is often the ground truth (real speech), allowing assessment of how closely synthetic speech approaches human quality @minixhofer_ttsds2_2025. This method has gained traction in recent works, such as StyleTTS 2 @li_styletts_2023, where it helped demonstrate improvements over baselines. Unlike #abbr.a[MOS], #abbr.a[CMOS] can reveal subtle differences even when absolute scores saturate, but it requires more pairwise comparisons, increasing listener effort @wells_bws_2024. Studies like the VoiceMOS Challenge @huang_voicemos_2024 have used #abbr.a[CMOS] variants to benchmark systems, showing its robustness in crowdsourced settings.

==== Best-Worst Scaling (BWS) <08_bws>

Best-Worst Scaling (BWS) is a forced-choice method where listeners are presented with a set of samples (typically 4) and asked to select the best and worst according to a criterion like naturalness. This implicitly ranks the middle items, providing multiple pairwise comparisons from one judgment @wells_bws_2024. BWS has been applied to TTS to address MOS limitations, offering more consistent ratings with fewer annotations in tasks like sentiment analysis or summarization evaluation @wells_bws_2024.

In TTS, BWS is efficient for comparing multiple systems, as each tuple yields several implicit rankings (e.g., a 4-tuple gives 5/6 pairwise comparisons) @wells_bws_2024. It mitigates scale biases in MOS by focusing on relative judgments and has shown reliability in low-resource languages like Scottish Gaelic @wells_bws_2024. However, listeners may find it more taxing due to memory demands when comparing audio samples, unlike text-based BWS @wells_bws_2024. Experimental comparisons show BWS provides similar outcomes to MOS and AB tests but is perceived as less engaging and more complex @wells_bws_2024.

==== Speaker similarity MOS <08_smos>

Sometimes we do not want to evaluate the quality or naturalness of the speech, but how closely the speakers in two samples resemble each other. In this case, #abbr.a[SMOS] is commonly used, which operates in the same way as #abbr.a[CMOS], but instead asks listeners to rate how closely the speakers resemble each other.

Typically evaluated on a 5-point scale from 1 (definitely different speakers) to 5 (definitely the same speaker), #abbr.a[SMOS] is essential for voice-cloning TTS systems @casanova_xtts_2024. It assesses speaker fidelity in zero-shot scenarios, where a reference utterance guides synthesis @minixhofer_ttsds2_2025. For example, in evaluations of systems like XTTS @casanova_xtts_2024, #abbr.a[SMOS] correlates with objective speaker similarity metrics like cosine distance of embeddings @zhao_probing_2022. Drawbacks include sensitivity to prosodic variations that may mimic speaker changes, and it often requires non-matching references to avoid bias from identical content @cooper_review_2024. In multilingual benchmarks, #abbr.a[SMOS] has revealed challenges in low-resource languages @minixhofer_ttsds2_2025.

=== Objective metrics <08_distances>

Due to the considerable effort and resources required to conduct subjective evaluation, objective metrics are frequently used for TTS evaluation, especially for ablation studies @cooper_review_2024. We group existing metrics into several categories into the following categories, with the function of distributional metrics being outlined in further detail in @09_dist[Chapter]. In addition to these families we also distinguish #emph[intrusive] and #emph[non-intrusive] metrics. Intrusive metrics require some ground truth speech of the same speaker as a reference. Non-intrusive metrics are reference-free. When the reference does not need to contain the same lexical content, it is described as #emph[non-matching]. These metrics aim to approximate human perception without listener involvement, but their validity must be continually reassessed as TTS evolves @moller_quality_2009 @minixhofer_ttsds_2024.

*Signal-based reference metrics:*
The oldest group consists of intrusive metrics that compare each synthetic utterance to a matching reference. #abbr.l[PESQ] @rix_pesq_2001, #abbr.l[STOI] @taal_stoi_2011 and #abbr.l[MCD] @kominek_mcd_2008 are the best–known representatives. They were mostly designed for telephone or enhancement scenarios rather than TTS, and require access to the ground‑truth waveform.

#abbr.l[PESQ] estimates perceived quality by modeling auditory perception, accounting for distortions like delay and filtering @rix_pesq_2001. #abbr.l[STOI] focuses on intelligibility by correlating short-time spectral envelopes @taal_stoi_2011. #abbr.l[MCD] measures spectral differences using Mel-cepstral coefficients, often used in TTS to quantify timbre mismatch @kominek_mcd_2008. These metrics are intrusive, requiring time-aligned references, which limits their use in zero-shot TTS @minixhofer_ttsds2_2025. While effective for early systems, they struggle with modern TTS artifacts like prosodic inconsistencies @cooper_review_2024.

*Model-based:*
To predict scores directly, researchers train neural networks that map a single audio signal to an estimated MOS. #emph[MOSNet] @lo_mosnet_2019 introduced the idea, and was followed by #emph[UTMOS] @saeki_utmos_2022, its SSL‑based successor #emph[UTMOSv2] @baba_t05_2024, and #emph[NISQA‑MOS] @mittag_nisqa_2021. #emph[SQUIM-MOS] @kumar_torchaudio-squim_2023 additionally grounds its prediction by requiring a non-matching reference of the ground truth speech. These methods report in‑domain correlations; however, recent VoiceMOS challenges @huang_voicemos_2024 show that correlation with subjective ratings drops out-of-domain. The drawbacks listed in @08_distances make training these networks challenging, and generalisation to new TTS systems or domains unlikely -- as the quote from @moller_quality_2009 at the beginning of this chapter indicates, each time substantially different TTS systems are introduced, learned metrics will have to be adjusted and verified anew. A recent prediction system goes beyond #abbr.a[MOS] and argues that no single score can capture everything listeners care about. #emph[Audiobox Aesthetics] predicts four axes, Production Quality, Complexity, Enjoyment, and Usefulness, for arbitrary audio @tjandra_meta_2025 -- however, the challenges of learning from subjective labels remain the same, including domain shifts and the need for large, diverse training sets of rated audio @minixhofer_ttsds2_2025.

*Intelligibility:*
Often reported are also #abbr.l[WER] and #abbr.l[CER], computed on #abbr.a[ASR] transcripts. These metrics quantify how easily synthetic speech can be transcribed by an ASR system, serving as a proxy for human intelligibility @cooper_review_2024. Lower #abbr.l[WER] indicates clearer pronunciation, but in domains like children's speech, higher #abbr.l[WER] may be realistic @minixhofer_ttsds_2024. They are non-intrusive but require ASR models, and performance depends on the ASR's robustness @radford_robust_2023.

*Speaker similarity:* 
Analogous to #abbr.a[SMOS], cosine similarity between the speaker embeddings (see @02_speaker) of a reference and target speech is frequently reported. Systems like d-vectors @wan_generalized_2018 or ECAPA-TDNN @desplanques_ecapa_2020 compute embeddings, and similarity is the cosine of their angle @zhao_probing_2022. This non-intrusive metric correlates with #abbr.a[SMOS] but may overlook prosodic influences on perceived similarity @minixhofer_ttsds2_2025.

*Distributional:* Inspired by the image domain's #abbr.l[FID] @heusel_fid_2017, audio researchers proposed measuring entire corpora rather than single files. #abbr.l[FAD] @kilgour_fad_2019 compares embeddings and has since been adapted for TTS @shi_versa_2024. Distributional metrics require a set of references which do not need to correspond to the synthetic data. The authors of these metrics state the need for thousands of samples, which may be why they have not found more widespread adoption @minixhofer_ttsds_2024. We go into more detail into the workings of these metrics in the next Chapter, where we propose TTSDS as a factorized distributional metric that correlates robustly with subjective scores across domains and languages @minixhofer_ttsds2_2025.