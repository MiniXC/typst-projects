#import "../abbr.typ" 
#import "../quote.typ": *
#import "../comic.typ"

== Measuring distributional distance <09_dist>

#q(
  [#citep(<rubner_earth_2000>)], 
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval")],
  [… we want to define a consistent measure of distance, or dissimilarity, between two distributions of mass in a space that is itself endowed with a ground distance. … Practically, it is important that [such] distances between distributions correlate with human perception.]
)

As we have established throughout this work, it is useful to think of speech as a distribution. In this chapter, we formalize this further, and introduce a method to empirically measure how far real and synthetic speech distributions are apart across systems, domains and languages.

=== Audio & speech distributions

If we think of the set of all possible speech recordings with some specific constraints, the difficulty of matching the real speech distribution becomes clear. Even if we constrain utterances to never exceed 60 seconds, and allow each data point within an utterance to only be one of $2^16$ values (referred to as a bit depth of 16) and set the sampling rate to 16kHz, this results in $16,000*60=960,000$ values per recording. The resulting number of possible recordings is hard to fathom: $2^(16 times 960,000)$ -- however, to human listeners, the vast majority of these recordings would sound like meaningless noise.

When creating a system capable of producing synthetic speech, we should aim to model the real speech distribution "hidden" within this impossibly large possible recording space -- however, if we knew said distribution, we would not need to model it in the first place. We therefore usually settle for estimating the distribution from data, and verify our models learned something approximating the real distribution by asking listeners to quantify their subjective perceptions as outlined in @08_eval[Chapter]. However, we can also quantify how closely the synthetic distribution resembles the real distribution as outlined in the remainder of this Chapter.

=== Earth movers distance

#comic.comic((80mm, 40mm), "Comic illustration of kernel density estimates of X-Vector speaker embeddings in 2D PCA space for ground truth, synthetic, and noise data, with normalized density scaled by 10^{-5}", blue) <fig_xvector>

An inuitive way to measure the distance between two distributions is the #abbr.a[EMD], named and first introduced for the purpose of determining perceptual similarity for image retrieval by @rubner_earth_2000 and is derived from the Wasserstein distance @vaserstein_markov_1969 which in turn makes use of the Kantorovich–Rubinstein metric @kantorovich_planning_1939. Its motivation is explained as follows:

#q(
  [Yossi Rubner, Carlo Tomasi and Leonidas J. Guibas], 
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval"), 2000 @rubner_earth_2000],
  [Intuitively, given two distributions, one can be seen as a mass of earth properly spread in space, the other
as a collection of holes in that same space. Then, the #abbr.a[EMD] measures the least amount of work needed to fill the holes with earth. Here, a unit of work corresponds to transporting a unit of earth by a unit of ground
distance.]
)

This makes it a transport problem @hitchcock_transport_1941, which can be solved for the 2D case of image histograms as in the original work @rubner_earth_2000 -- however, speech representations (see @02_perceptual) are often high-dimensional, in which case this problem is prohibitively expensive.

=== Wasserstein metric

While the general #abbr.a[EMD] is computationally complex, a specific instance known as the *2-Wasserstein distance* offers tractable solutions in certain cases that are highly relevant for comparing distributions of speech representations.

Formally, the Wasserstein distance measures the distance between two empirical probability distributions, defined as the real distribution $P_R$ and the synthetic distribution $P_S$. It is defined as the minimum "cost" to transform one distribution into the other. The cost is calculated by multiplying the amount of "mass" moved by the distance it is moved. The set of all possible ways to move the mass is called the set of *transport plans*, $Pi(P_R, P_S)$. Each transport plan $gamma(x,y)$ is a joint distribution whose marginals are $P_R$ and $P_S$. Intuitively, $gamma(x,y)$ specifies how much mass to move from point $x$ to point $y$ to transform the distribution $P_R(x)$ into $P_S(y)$.

The $p$-Wasserstein distance is then the minimum cost over all possible transport plans:

$ W_p(P_R, P_S) = (inf_(gamma in Pi(P_R, P_S)) EE_((x,y)~gamma)[d(x,y)^p])^(1/p) $

where $d(x,y)$ is the distance between points $x$ and $y$. For our purposes, we will focus on the case where $p=2$ and $d(x,y)$ is the Euclidean distance $||x-y||_2$, rather than the #abbr.a[EMD] for which $p=1$. As with the general #abbr.a[EMD], computing this for arbitrary high-dimensional distributions remains difficult @kolouri_optimal_2017. However, two special cases provide efficient, closed-form solutions.

#strong[One-dimensional case]

In the one-dimensional case, the 2-Wasserstein distance has a simple closed-form solution that avoids searching over all possible transport plans. It can be calculated directly from the inverse #abbr.a[CDF] of the two distributions. Given the #abbr.a[CDF] for the real and synthetic distributions, $C_R$ and $C_S$, the squared 2-Wasserstein distance is simply the squared L2-distance between their inverse functions @kolouri_optimal_2017:

$ W_2^2(P_R, P_S) = integral_0^1(C_R^(-1)(z)-C_S^(-1)(z))^2d z $

This property is the foundation of the *Sliced-Wasserstein distance*, which computes the average Wasserstein distance between distributions over many random one-dimensional projections. However, there is another, form for the high-dimensional case which does not rely on slices.

#strong[High-dimensional case with gaussian assumption]

For sets of high-dimensional vectors, as is common for #abbr.a[DNN] features, computing the quantile functions is not feasible. However, as proposed by @heusel_fid_2017 in the context of image generation, we can make a simplifying assumption: that the embedding distributions can be approximated by multivariate Gaussians. This is a reasonable assumption for embeddings that have been projected into a well-behaved latent space @heusel_fid_2017. This approximation allows us to again compute the 2-Wasserstein distance in closed form using only the mean and covariance of the distributions.

Let the real and synthetic embedding distributions be modeled by multivariate Gaussians $N(mu_R, Sigma_R)$ and $N(mu_S, Sigma_S)$ respectively. The squared 2-Wasserstein distance, also known as the Fréchet distance @frechet_1925, between these two Gaussians is given by @dowson_frechet_1982:

$ W_2^2(P_R, P_S) = ||mu_R - mu_S||_2^2 + text("Tr")(Sigma_R + Sigma_S - 2(Sigma_R Sigma_S)^(1/2)) $

where:
- $mu_R$ and $mu_S$ are the mean vectors of the real and synthetic embeddings.
- $Sigma_R$ and $Sigma_S$ are the covariance matrices.
- $text("Tr")(dot)$ is the trace of a matrix.
- $(C_R C_S)^(1/2)$ is the matrix square root of the product of the covariance matrices.

The first term, $||mu_R - mu_S||_2^2$, measures the distance between the centers of the two distributions. The second term measures the difference in their spread and orientation (i.e., their covariances). In practice, we estimate the sample mean and covariance from a large number of real and synthetic embeddings, respectively, and then compute the distance using the formula above. This metric is the basis for the well-known *Fréchet Inception Distance* (#abbr.s[FID]) @heusel_fid_2017 in image generation, and the same principle can be applied to audio embeddings to create a *Fréchet Audio Distance* @kilgour_fad_2019.

=== Perceptually-motivated factorized evaluation

As we discussed in @08_distances, there are various ways to objectively evaluate if synthetic speech matches its real counterparts, and many are perceptually motivated, in-line with the representations presented in @02_perceptual.

There is no single ground truth for synthetic speech generation due to the one-to-many nature of the task. Instead, we frame TTS evaluation as a distributional similarity task. Let $D$ be an audio dataset and $X$ a feature extracted from it. We aim to quantify how well synthetic speech mirrors real speech by obtaining correlates of each factor and measuring their distance from both real speech datasets and noise datasets. TTSDS is the average of the resulting scores along the following perceptually motivated factors: (i) #smallcaps[#smallcaps[Generic]]: Overall distributional similarity, via SSL embeddings. (ii) #smallcaps[Speaker]: Realism of speaker identity. (iii) #smallcaps[#smallcaps[Prosody]]: Pitch, duration, and rhythm quality. (iv) #smallcaps[#smallcaps[Intellibility]]: Recognition of the speech, using ASR-derived features.

#comic.comic((80mm, 40mm), "table showing TTSDS factors", green) <ttsds2_features>

Each factor is measured using multiple features (see the above figure). The feature-wise scores within each factor are averaged, and in turn the averaged factor scores provide the overall TTSDS score. Thus, when compared to the metrics in @08_distances, TTSDS is both a Multi-dimensional perceptual metric due to the individual factor scores, and a distributional metric, as we compare distributions rather than individual samples.

Computing Wasserstein distances. We use the 2-Wasserstein distance $W_2$, also known as the Earth Mover’s Distance, to compare feature distributions, in line with the Fréchet Inception Distance (FID) @heusel_fid_2017 frequently used in computer vision. This distance measure has several desirable properties which make it well-suited for this task – it is symmetric (unlike Kullback-Leibler divergence @lin_jsd_1991) and it can be used to differentiate between differing non-overlapping distributions (unlike Jensen-Shannon divergence @lin_jsd_1991) @kolouri_optimal_2017.

Assuming approximate Gaussianity of SSL embeddings, the distance can be formulated as
$ W_2(hat(P)_1, hat(P)_2) = ||mu_1 - mu_2||_2^2 + text("Tr") (Sigma_1 + Sigma_2 - 2(Sigma_1/2_2 Sigma_1 Sigma^(1/2)_2)^(1/2)) $
where $mu$ and $Sigma$ are the mean and covariance matrices of the feature @heusel_fid_2017. In the one-dimensional case , it can be formulated as
$ W_2(hat(P)_1, hat(P)_2) = sqrt(1/n sum_(i=1)^n(x_i - y_i)^2) $ where ${x_i}$; ${y_i}$ are sorted samples of the distributions @kolouri_optimal_2017.

To allow us to average across features and factors, we derive a score, ranging from 0 (identical with a noise distribution) to 100 (identical with the real reference) for any synthetic speech feature distribution $hat(P)(X|D_"syn")$ – the lexical content of the reference and synthetic speech do not need to match. We first compute the 2-Wasserstein distance of $hat(P)(X|D_"syn")$ to each $hat(P)(X|D_"noise")$ for a set of "distractor" noise datasets $D_"noise"$ and let the minimum of said values be $W_"noise"|(X)$. The 2-Wasserstein distance to the real speech distribution $hat(P)(X|D_"real")$ is in turn $W_"real"(X)$ – using these terms, the normalized similarity score for a feature $X$ is simply defined as follows:
$ S(X) = 100 times W_"noise"(X) / (W_"real"(X) + W_"noise"(X)) $

#comic.comic((80mm, 40mm), "density plot of F0 distributions", blue) <fig_f0>

A worked example of this score for the one-dimensional pitch feature can be seen in the above figure. The equation yields scores between 0 and 100, where values above 50 indicate stronger similarity to real speech than noise. The final TTSDS score is the unweighted mean of factor scores, with each factor score in turn being the mean of its feature scores.

Updated factors and features. In this work, we modify the original TTSDS @minixhofer_ttsds_2024 to make it robust to several domains for each of its factors. #smallcaps[Intellibility] in TTSDS relies on Word Error Rate (WER). In preliminary experiments, these WER features resulted in low scores for real data across domains – to increase robustness, we use speech recognition models’ final-layer activations instead. For #smallcaps[Prosody], TTSDS originally used (a) the WORLD pitch contour @morise_world_2016, (b) masked-#smallcaps[Prosody]-model embeddings @wallbridge_mpm_2025, and (c) token lengths (in frames) extracted from HuBERT @hsu_hubert_2021. We found that the token-length features lead to low scores for real speech. We instead compute the utterance-level speaking rate by dividing the number of deduplicated HuBERT tokens in an utterance by the number of frames. We do the same for the multilingual phone recogniser Allosaurus @li_allosaurus_2020, also included in the original. #smallcaps[Generic] uses the same HuBERT @hsu_hubert_2021 and wav2vec 2.0 @baevski_wav2vec_2020 features as in the original, but we also add WavLM @chen_wavlm_2022 features for increased diversity. The factors and their features are shown in the above figure. For multilingual use, we replace HuBERT with mHuBERT-147 @boito_mhubert-147_2024, and wav2vec 2.0 with its XLSR-53 counterpart @conneau_xlsr_2021.

=== Correlations with listening tests across datasets

We now outline how we validate TTSDS to correlate with human scores across a variety of datasets, all of them in the English language.

#strong[Datasets:]
Since most systems are still trained using audiobook speech, and audiobook speech is easier to synthesize due to its more regular nature @he_emilia_2024, we use samples from the LibriTTS @zen_libritts_2019 test split as a baseline. Since LibriTTS is filtered by Signal-to-Noise Ratio (SNR), it only contains clean, read speech. In the remainder of this work, we refer to this as #smallcaps[Clean]. For all datasets, utterances between 3 and 30 seconds with a single speaker are selected. The remaining datasets alter this baseline domain in the following ways:
#smallcaps[Noisy] is created by scraping LibriVox recordings from 2025 (to avoid their occurrence in the training data) without SNR filtering. This tests how evaluation is affected by noise present in the recordings.
#smallcaps[Wild] is created by scraping recent YouTube videos and extracting utterances, which tests the metrics’ ability to generalize to diverse speaking styles and recording conditions. Its data collection and processing are inspired by Emilia @he_emilia_2024. We scrape 500 English-language YouTube videos uploaded in 2025 using 10 different search terms which emphasise scripted and conversational speech alike. We perform Whisper Diarization @radford_robust_2023 to isolate utterances.
#smallcaps[Kids] is a subset of the My Science Tutor Corpus @pradhan_my_2024 and contains children’s conversations with a virtual tutor in an educational setting. This tests if evaluation metrics can generalize to data rarely encountered during the training.
For all systems, we select 100 speakers at random, with two utterances per speaker. We then manually filter the data to exclude content which is (i) difficult to transcribe or (ii) potentially controversial or offensive. This leaves us with 60 speakers for each dataset. The first utterance by each speaker is used as the reference provided to the TTS system, while the transcript of the second utterance is used as the text to synthesize. This way, we can evaluate both intrusive and non-intrusive metrics. We use matching speaker identities to eliminate any possible preferences of listeners of one speaker over another, and to avoid systems scoring highly merely because of a set of speakers is closer to the reference than for other systems.

#strong[Listening Tests:]
We recruit 200 annotators using prolific.com which annotate the ground-truth and synthetic data for 20 TTS systems across the aforementioned datasets, in terms of MOS, CMOS and SMOS. Annotators are screened to be native speakers from the UK or the US and asked to wear headphones in a quiet environment. Any that fail attention checks are excluded. Each annotator is assigned to one dataset, resulting in 50 listeners per dataset. For MOS, there are 6 pages with 5 samples each, one of which is always the ground truth, while the others are selected at random. For CMOS and SMOS, 18 comparisons between ground truth and a randomly selected system’s sample are conducted. To avoid any learning or fatigue effects if a certain measure is always asked first or last, the order of the three parts of the test is varied from annotator to annotator. The median completion time was 32 minutes and the annotators were compensated with $10$, resulting in an hourly wage of $approx 19$. For both MOS and CMOS, we instruct annotators to rate the Naturalness of the speech. MOS and SMOS, in line with recommendations of @kirkland_mospit_2023, are evaluated on a 5-point scale ranging from Bad to Excellent. CMOS is evaluated on a full-point scale ranging from -3 (much worse) to 3 (much better). We collect a total of 11,846 anonymized ratings and utterances, of which we publish 11,282, excluding the ground truth utterances due to licensing. The ratings can be accessed at hf.co/datasets/ttsds/listening_test. While we use this data to validate if TTSDS aligns with human ratings, future work could use it for improving MOS prediction networks, since, to the best of our knowledge, all publicly available datasets of this size use TTS systems which have not reached human parity @huang_voicemos_2024 @tjandra_meta_2025 @cooper_review_2024 @lo_mosnet_2019.

#strong[Evaluated objective metrics:]
We use the VERSA evaluation toolkit @shi_versa_2024 for all compared objective metrics, except UTMOSv2, which was not included at the time of writing. For Audiobox Aesthetics we select their Content Enjoyment (AE-CE), Content Usefulness (AE-CU), and Production Quality (AE-PQ) subscores, which they show to correlate with MOS @tjandra_meta_2025. For distributional metrics, we evaluate Fréchet Audio Distance using Contrastive Language-Audio Pretraining latent representations @laion_clap_2023. For MOS prediction, we evaluate UTMOS @saeki_utmos_2022, UTMOSv2 @baba_t05_2024, NISQA @mittag_nisqa_2021, DNSMOS @reddy_dnsmos_2022, and SQUIM MOS @kumar_torchaudio-squim_2023, which is the only MOS prediction system we evaluate that requires a non-matching reference. For speaker embedding cosine similarity, which require non-matching reference samples as well, we use three systems included in ESPNet-SPK @jung_espnet-spk_2024: RawNet3, ECAPA-TDNN and X-Vectors. We also include some legacy signal-based metrics, which are STOI, PESQ, and MCD – these are the only ones to require matching references. In the next section, we compare these metrics with the subjective evaluation scores. For TTSDS, we evaluate both the original version outlined in our previous work @minixhofer_ttsds_2024 and our updated version described in this chapter. Both can be found at github.com/ttsds/ttsds.

#strong[Correlations:]
For each TTS of the 20 systems, we average human ratings for MOS, CMOS and SMOS for #smallcaps[Clean], #smallcaps[Noisy], #smallcaps[Wild] and #smallcaps[Kids]. These are the “gold” ratings. We now examine the Spearman correlation coefficients (since we deem ranking the systems most important) of these results with the aforementioned metrics across the four datasets. As the above figure shows, TTSDS shows the most consistent correlation across the datasets, with an average correlation of 0.67, surpassing the original by 10% relative. All correlations for TTSDS and TTSDS2 are statistically significant with p < 0.05. Speaker Similarity metrics come second, with average correlations of 0.6 for RawNet3 and X-Vector Speaker Similarities. Of the MOS Prediction networks, only SQUIM MOS performs well, with an average correlation of 0.57. Following the final Speaker Similarity tested, ECAPA-TDNN, there is a large drop in average correlation, with all remaining averages below 0.3. We note that many of the metrics, including Audiobox Aesthetics and UTMOSv2, still perform well on #smallcaps[Noisy] and #smallcaps[Clean], which only contains audiobook speech. Metrics seem to struggle most on #smallcaps[Kids], which is expected, as it is the furthest removed from the most common TTS domains.

#comic.comic((80mm, 40mm), "scatter plots showing correlations", green) <ttsds_correlations_plot>

We also investigate the top-performing TTSDS, X-Vector, and SQUIM MOS correlations. As the above figure shows, some behaviours are not shown in their correlation coefficients alone; TTSDS acts the most like a continuous scale; both SQUIM MOS and X-Vector Speaker Similarity show some clustering behaviour. Since both SQUIM and X-Vector are essentially black boxes, we cannot conclusively state what causes this behaviour, but it could indicate overfitting to specific systems.

=== Multilingual & recurring evaluation

While the previous section outlined robustness across datasets in a single language due to the ease of conducting listening tests in English, we extend TTSDS for multilingual use, and provide a public benchmark in 14 languages – this covers all languages synthesised by at least two systems, and is to be extended as more multilingual TTS systems are released.

As our benchmark should be updated frequently to avoid data leakage, and represent a wide range of recording conditions, speakers and environments, we decide to automate the creation of the #smallcaps[Wild] dataset described earlier. However, since manual filtering is not feasible in the long term for a growing set of languages and evaluation re-runs, we automate the collection process as can be seen in the above figure, and which we describe in detail in the following section.

#comic.comic((80mm, 40mm), "overview of TTSDS pipeline", blue) <fig_ttsds_pipeline>

#strong[Pipeline:]
The TTSDS pipeline, available at github.com/ttsds/pipeline is used to rebuild the multilingual dataset every quarter: (i) *Data Scraping*: First, 250 videos are collected per language. The 10 keywords outlined earlier are translated for each language, and for each term, a narrow YouTube search in the given language, and for the time range for the previous quarter of the year, is conducted using YouTube’s API. The results are sorted by views to avoid low-quality or synthetic results and only videos longer than 20 minutes are included. The videos are diarised using Whisper as before.
We use FastText @joulin_fasttext_2016 @bojanowski_fasttext_2016 language identification on the automatically generated transcripts to filter out videos not in the target language. (ii) *Preprocessing*: We then extract up to 16 utterances from the middle of the video – we only keep utterances from a single speaker as identified in the previous diarisation step. (iii) *Filtering*: Next, the utterances are filtered for potentially offensive or controversial content. In preliminary experiments, we find LLM-based filtering to be too resource-intensive for a recurring benchmark, while toxic comment classification work lets through potentially controversial but not explicitly toxic content, and often is not available in multiple languages. We find a solution by filtering the data using XNLI @conneau_xnli_2018 with potentially controversial topics as entailment. This leads to a large number of falsely filtered texts, which in our case is not a downside, since we only want a small number of “clean” samples. Finally, we use Pyannote speaker diarisation @bredin_pyannote_2023 to detect if there is any crosstalk, and Demucs @rouard_demucs_2022 source separation, to detect if there is any background music. Of the remaining utterances, 50 speaker-matched pairs are selected for each language, and split into the #smallcaps[Reference] and #smallcaps[Synthesis] set. (iv) *Synthesis*: For all systems in the benchmark, accessible at replicate.com/ttsds, we synthesise the speaker identities in #smallcaps[Reference] with the text in #smallcaps[Synthesis]. (v) *TTSDS*: We apply multilingual TTSDS to arrive at scores for each, and publish the results at ttsdsbenchmark.com. This is repeated each quarter, with systems published in the previous quarter, to avoid data contamination. We plan to expand to more systems and languages each evaluation round.

#strong[Multilingual validity of TTSDS:]
While collecting gold MOS labels for 14 languages is out of scope for this work, we verify TTSDS’s applicability to the multilingual case using Uriel+ @khan_uriel_2024, which supplies typological distances for 7970 languages. We pose that if TTSDS distances correlate to language distances found by linguists, it has some merit to be applied to this multi-lingual case. We also conduct analysis for each factor of TTSDS. We find that when comparing the ground truth language datasets to each other using TTSDS, the scores correlate with the distances with $rho = -0.39$ for regular and $rho = -0.51$ (both significant with p < 0.05) for multilingual TTSDS discussed earlier – the negative correlations are expected since a higher score correlates with a smaller distance, and the higher correlation of multilingual TTSDS scores is encouraging.

#comic.comic((80mm, 40mm), "Comic box plots of TTSDS scores across 14 languages, showing ground truth near 95 and synthetic scores varying by language, generally between 75 and 95", red) <fig_language_scores>

// === Limitations
// Since TTSDS extracts several features for each utterance, it uses more compute than other methods. While it robustly correlates with human evaluation, it never surpasses Spearman correlation coefficient of 0.8, indicating there is either a component of listening tests that is inherently noisy, or not predicted by any objective metric – TTSDS is not equivalent to, nor can it replace, subjective evaluation. Additionally, some modern TTS systems can have failure cases which are not identifiable as such by TTSDS - such as when the transcript given is not reproduced faithfully - in fact, exactly copying the reference transcript would yield a perfect score. To mitigate this shortcoming, we report the number of utterances with high Word Error Rates for each system at ttsdsbenchmark.com. TTSDS currently also does not capture the context the utterances were spoken in, and does not include long-form samples, as many systems do not support generation of utterances beyond 30 seconds.