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
#q(
  [#citep(<wang_tacotron_2017>)],
  [#emph[Tacotron: Towards End-to-End Speech Synthesis]],
  [TTS is a large-scale inverse problem: a highly compressed source (text) is "decompressed" into
audio. Since the same text can correspond to different pronunciations or speaking styles, this is a particularly difficult learning task …]
)

The primary goal of Text-to-Speech (TTS) is to automatically convert a sequence of text into a natural-sounding utterance. This chapter provides a comprehensive overview of the systems designed to solve this complex "one-to-many" problem, where a single text or other #smallcaps[Semantic] input can correspond to countless valid acoustic realisations depending on factors like #smallcaps[Speaker] and #smallcaps[Prosody] introduced in @02_factors[Chapter]. This chapter will illustrate how TTS systems are fundamentally designed to generate and manipulate representations related to these factors, while using yet more representations of speech for conditioning and internal representations.

=== Forms of Speech Synthesis

In our work, we focus on #abbr.a[TTS], however there are other paradigms for speech generation, the most common being #emph[Textless Generation] and #emph[Voice Conversion]. @fig:fig_tts_vc_textless visualises these differing approaches.

==== Textless generation

Beyond text-conditioned synthesis, these models produce speech without any explicit or implicit semantic input, sampling instead from a learned distribution of speech sounds. Thus, these models aim to learn the underlying probability distribution $Q(S)$ directly. Early neural approaches, such as the original WaveNet architecture, demonstrated this by using an autoregressive model to predict low-level #smallcaps[Generic] raw audio samples one at a time, conditioned only on previous samples @oord_wavenet_2016. Textless models are foundational for tasks like audio completion or infilling but they require careful handling of sequence length and long-term coherence to avoid devolving into unstructured noise @tan_survey_2021. Recently, this form of generation has seen increased interest due to #abbr.pla[SLM] @lakhotia_gslm_2021 where a #abbr.a[LLM]-like system is trained to predict high-level acoustic units, usually contextualised speech embeddings -- however, even the latest systems still lack semantic coherence, with the text extracted from speech produced by these models achieving $<20%$ "win rate" against real text @park_speechssm_2025. While they usually do not perform as well as TTS, such approaches can be necessary when there is no script for the target language @chen_hokkien_2023.

==== Voice Conversion

While TTS maps a #smallcaps[Semantic] representation ($T$) to a #smallcaps[Generic] acoustic one ($S$), the related task of #emph[#abbr.a[VC]] transforms an input speech signal from a source style to a target style while preserving the linguistic content. VC conditions on a source utterance $S^"SRC"$ rather than text $T$:

$
tilde(S) tilde Q_theta (S | S^"SRC", Z)
$

Here, the conditioning set $Z$ typically contains a high-level #smallcaps[Speaker] embedding representing the target voice. A key challenge in VC is to effectively disentangle the #smallcaps[Semantic] content of the speech from the #smallcaps[Speaker] and #smallcaps[Prosody] factors, often using specialised feature extractors or autoencoder-based architectures @sisman_vc_2020. VC enables applications like real-time voice modification and dubbing, where the goal is to change the voice characteristics without relying on a transcript.

#figure(
  fletcher.diagram(
    spacing: 5pt,
    cell-size: (5mm, 10mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 70%,
    
    // Unconditional Generation (NEW)
    fletcher.node((-2, 3.2), align(center)[$Z$]),
    fletcher.edge((-2, 3.2), (-2, 2), "--|>"), // Optional conditioning (e.g., random seed)
    blob((-2, 2), [Textless Generation], tint: orange, width: 42mm),
    fletcher.edge((-2, 2), (-2, 0.9), "-|>"),
    fletcher.node((-2, 0.9), $tilde(S)$),

    // voice conversion
    fletcher.node((-.37, 3.2), align(center)[$Z$]),
    fletcher.edge((-.37, 3), (0, 2), "--|>", bend: -15deg),
    fletcher.node((0, 3), $S^text("SRC")$),
    fletcher.edge((0, 3), (0, 2), "-|>"),

    blob((0, 2), [Voice Conversion], tint: orange, width: 42mm),
    fletcher.edge((0, 2), (0, 0.9), "-|>"),
    fletcher.node((0, 0.9), $tilde(S)$),

    // tts
    fletcher.node((1.6, 3.2), align(center)[$Z$]),
    fletcher.edge((1.6, 3.2), (2, 2), "--|>", bend: -15deg),
    fletcher.node((2, 3), $T$),
    fletcher.edge((2, 3), (2, 2), "-|>"),
    blob((2, 2), [Text-to-Speech], tint: orange, width: 42mm),
    fletcher.edge((2, 2), (2, 0.9), "-|>"),
    fletcher.node((2, 0.9), $tilde(S)$)
  ),
  placement: top,
  caption: "Comparison of Text-to-Speech (TTS), Voice Conversion (VC), and Textless Generation.",
) <fig_tts_vc_textless>

==== Text-to-Speech

For this thesis, we focus on TTS due to the abundance of openly available TTS models compared to #abbr.pla[VC] or #abbr.pla[SLM] as well as due to their controllability -- being able to condition on separate #smallcaps[Semantic], #smallcaps[Speaker] and optionally #smallcaps[Prosody] representations allows us to investigate the synthetic speech in more controlled experimental setups. Systems which condition on at least the former two are understood as multi-speaker, voice-cloning TTS systems, which make up the majority of modern TTS systems @tan_survey_2021. The #smallcaps[Semantic] representation is text or derived from text, such as contextualised text embeddings or phones. The #smallcaps[Speaker] representation is most commonly a high-level learned speaker embedding (see @02_speaker), but can also be another representation containing salient speaker information, such as a mid-level Mel spectrogram of a reference utterance @eskimez_e2_2024 or even a high-level text prompt describing their identifying characteristics @lyth_parler_2024.

=== Hierarchy of Text-to-Speech

Most TTS systems have a distinct #emph[frontend], transforming the input text in some speech representation and a #emph[vocoder], converting some speech representation to a waveform. These components are useful since the "raw" text $T$ and waveform $S$ are not ideal representations for modeling speech.

==== Frontend
Text can be ambiguous, for example in terms of pronunciation, or representation of numbers and dates.
This makes the frontend, or text analysis module, a critical component of many TTS systems. Its purpose is to transform raw, unstructured text into a clean, structured linguistic feature representation suitable for the acoustic model @taylor_tts_2009. This process is fundamentally about converting the highest-level #smallcaps[Semantic] representations into a format that can be more easily mapped to acoustics. This can involve the following:

*Text Normalisation:* This stage converts non-standard words, such as numbers, abbreviations, and symbols, into their full written form. For example, "1989" becomes "nineteen eighty-nine" and "Dr." becomes "Doctor." This step can help to ensure correct pronunciation.

*Linguistic Analysis:* This can include tasks like part-of-speech (POS) tagging to resolve pronunciation ambiguities (e.g., "read" as /riːd/ vs. /rɛd/) and word segmentation for languages like Chinese that do not use spaces.

*Grapheme-to-Phoneme (G2P) Conversion:* This is often the final step, converting the normalised text into a mid-level #smallcaps[Semantic] representation: the phone sequence. This mapping from orthography to a phonetic representation like /s p iː tʃ/ is essential for accurate pronunciation, particularly in languages with irregular orthography like English or with polyphones in Chinese @yao_g2p_2015.

*Text Tokenization:* While traditional systems used complex, multi-stage, rule-based frontends, modern neural TTS systems have simplified this process. Many models now operate directly on character sequences, only tokenizing the text @hayashi_espnet-tts_2020. A recent development is the use of #abbr.a[SSL] models to extract phone-like representations directly from audio @liu_semanticodec_2024, which can then be used to train TTS systems, bypassing traditional G2P altogether.

==== Vocoder

Raw waveforms are high-resolution, low-dimension representations, with typically 16- to 44-thousand values representing each second of audio. Generative models therefore are often tasked to predict a more low-resolution representation with higher dimensionality, usually a lossy-reconstructible mid-level #smallcaps[Generic] representation like a Mel spectrogram, which is then converted to a raw waveform using a vocoder. Legacy systems rely on algorithmic vocoders based on the source-filter model. LPC Vocoders @atal_lpc_1970, originally developed for telephony, provide a computationally cheap but robotic-sounding output. Source-Filter Vocoders like WORLD @morise_world_2016, represented an improvement by modeling the source (F0, aperiodicity) and filter (spectral envelope) to produce much more natural, albeit sometimes "buzzy" speech for parametric systems. Early end-to-end models like Tacotron, which were among the first to directly produce mel spectrograms successfully, used the algorithmic Griffin-Lim method @griffin_griffinlim_1984 to iteratively estimate phase and reconstruct a waveform.

With deep learning, neural vocoders were introduced, leading to a dramatic leap in quality. Among the first were Simple Autoregressive models like WaveNet @oord_wavenet_2016, which generate audio sample-by-sample, achieving human-level fidelity at the cost of extremely slow inference. To address this, parallel models were developed. Flow-based vocoders like WaveGlow @prenger_waveglow_2019 use normalising flows for fast, high-quality parallel generation. However, GAN-based vocoders have become the dominant approach for their exceptional balance of speed and quality. Systems like MelGAN @kumar_melgan_2019, HiFi-GAN @kong_hifigan_2020 or BigVGAN @lee_bigvgan_2023 use a generator to create audio from a Mel spectrogram and a discriminator to ensure its perceptual realism, enabling real-time, high-fidelity synthesis. Diffusion-based vocoders such as DiffWave @kong_diffwave_2021, are another high-quality parallel approach, learning to reverse a noising process, though they can be slower than GANs. The most recent development involves using the decoder of a pre-trained Neural Audio Codec as the vocoder. Models like EnCodec @defossez_encodec_2023 learn to tokenize audio into a discrete representation, and their decoders can perfectly reconstruct a waveform from these tokens, a technique leveraged by the latest generation of TTS models.

=== History of Text-to-Speech

#figure(
  image("../figures/3/tts_timeline.png", width: 100%),
  caption: [Non-exhaustive timeline of TTS systems, as well as frontend and vocoder technologies used.],
  placement: top,
) <fig_tts_timeline>

The evolution of TTS, as visualised in @fig:fig_tts_timeline, reflects a progression in architectures and the representations they employ, moving from complex pipelines to more integrated, end-to-end systems.

The legacy era is defined by modular systems. *Concatenative synthesis*, exemplified by the Festival toolkit @taylor_festival_1998, uses a frontend with full linguistic analysis to select raw waveform chunks from a database. As it concatenates existing audio, it does not require a vocoder. In parallel, *Statistical Parametric Speech Synthesis* (SPSS), seen in HTS @tokuda_hts_2013 and later the DNN-based Merlin toolkit @wu_merlin_2016, also uses a complex linguistic frontend but generates parameters for a Source-Filter vocoder like WORLD @morise_world_2016.

The first wave of end-to-end neural models simplifies the frontend to basic tokenization and focuses on generating spectrograms. Tacotron @wang_tacotron_2017 is a autoregressive model that used the algorithmic Griffin-Lim @griffin_griffinlim_1984 method as its vocoder. The major quality breakthrough comes with Tacotron 2 @shen_natural_2018, which pairs a similar spectrogram-prediction model with a powerful, though slow, Simple Autoregressive neural vocoder based on WaveNet @oord_wavenet_2016.

The subsequent drive for efficiency leads to a family of non-autoregressive models that generate Mel spectrograms in parallel. These systems, including FastSpeech, FastSpeech2, and GlowTTS, offload the final waveform generation to a separately trained, fast neural vocoder. As @fig:fig_tts_timeline shows, these models can be paired with various Mel Spectrogram Based vocoders, with GAN-based models like HiFi-GAN @kong_hifigan_2020 being the most common choice due to their speed and quality. Diffusion-based and Flow-based vocoders are also viable options for this architecture.

Recent developments have focused on integrated end-to-end approaches and output representations. VITS @kim_vits_2021 represents a true end-to-end approach, jointly learning to align text and generate a raw waveform directly within a single model, thus having its own integrated vocoder. Concurrently, diffusion models have been applied to the main TTS task in systems like NaturalSpeech2 @tan_naturalspeech_2024 and StyleTTS2 @li_styletts_2023, which typically generate a high-quality Mel spectrogram that is then rendered by a powerful GAN-based vocoder. The latest paradigm shift, represented by VALL-E @wang_valle_2023 and ParlerTTS @lyth_parler_2024, abandons spectrograms entirely. These systems use a simplified frontend, sometimes leveraging an SSL Phoneme Model, Phoneme-level BERT @li_plbert_2023, for robust phonetic representations, and their acoustic model predicts a sequence of discrete tokens. The final waveform is then synthesised using a Neural Audio Codec decoder as the vocoder.

=== Architectures and Training Objectives

The development of modern TTS has been driven by the exploration of diverse neural network architectures and the training objectives used to optimise them. While legacy systems were modular by necessity, current state-of-the-art approaches are predominantly #abbr.pla[DNN]. These systems typically take as input a #smallcaps[Semantic] representation in the form of a character or phoneme sequence, and are trained to produce a #smallcaps[Generic] acoustic representation as output, most commonly a Mel spectrogram or a sequence of discrete audio codec tokens. This section details the primary architectural paradigms that define how these models generate speech, and the training objectives used to guide their learning process.

==== Architectural paradigms <03_arch>

The architecture of a TTS model dictates how it transforms input text into an acoustic representation. The most fundamental distinction is whether the model generates its output autoregressively or non-autoregressively. The building blocks of these architectures have also evolved, with Transformers @vaswani_attention_2017 and Conformers @gulati_conformer_2020 becoming standard components.

*Autoregressive (AR) Models:* These models generate the output sequence one step at a time, conditioning each step on the previously generated outputs. If we let the input linguistic sequence be $T$ and the target acoustic sequence be $S = (s_1, ..., s_k)$, an AR model factorises the probability sequentially:
$ p(S|T) = product_(k=1)^K p(s_k|s_1,dots,s_(k-1),T) $
This sequential dependency allows AR models to capture complex, long-range correlations in the speech signal. For example, Tacotron 2 @shen_natural_2018, uses an attention mechanism to solve the alignment problem dynamically. At each decoder step $k$, the attention mechanism computes a set of alignment weights $alpha_k$, which determine how much "attention" to pay to each input embedding from the text encoder when generating the current audio frame. These weights are calculated based on the previous decoder state $d_(k-1)$ and the encoder outputs $h$:
$
alpha_k = "Attention"(d_(k-1), h)
$
The attention module calculates a context vector $c_k$ as the weighted sum of the encoder outputs, $c_k = sum_j alpha_(i,j) h_j$, which is then used to predict the output frame $s_k$. This process allows the model to dynamically learn the alignment between the text and the much longer audio sequence. While this approach produces high-quality speech, its sequential nature makes inference slow and can lead to error propagation. More recently, this paradigm has been applied to discrete token generation in models like VALL-E @wang_valle_2023, which use a Transformer architecture to autoregressively predict neural codec tokens.

*Non-Autoregressive (NAR) Models:* To overcome the slow inference of AR models, NAR models were developed to generate the entire output sequence in parallel. These models assume conditional independence between the output frames given the full input conditioning:
$ p(S|T) = product_(k=1)^K p(s_k|T) $
This parallel generation is extremely fast but presents a new challenge: aligning the variable-length input text sequence with the much longer output acoustic sequence. NAR models can be further subdivided based on how they solve this alignment problem.
Among non-autoregressive models, #emph[Explicit Duration-Based Models] address the alignment challenge directly by predicting the duration for each input unit (typically a phone). An early example, FastSpeech @ren_fastspeech_2019, introduced the Transformer architecture to NAR TTS. The Transformer's self-attention mechanism is highly effective at modeling long-range dependencies within the text sequence, which is crucial for producing coherent prosody. FastSpeech uses a "Length Regulator" to expand the input sequence according to predicted durations. The more robust and widely adopted approach, seen in FastSpeech 2 @ren_fastspeech_2021, integrates this into a Variance Adaptor module, which predicts low-level #smallcaps[Prosody] correlates like duration, pitch, and energy.
#emph[Implicit Alignment Models] represent a more advanced class of NAR architectures that learn the alignment end-to-end. Glow-TTS @kim_glowtts_2020 achieved this using normalising flows coupled with a Monotonic Alignment Search mechanism. A highly successful example is VITS @kim_vits_2021, which uses a Variational Autoencoder (VAE) framework to jointly learn a stochastic duration predictor and an alignment between the latent text and speech representations, guided by a connectionist temporal classification (CTC)-style loss. This allows the model to be trained end-to-end while discovering the alignment on its own. Recent works simplify this further with an alignment-free approach. These works only predict the total utterance length, making prediction of phoneme durations fully implicit @eskimez_e2_2024@chen_f5_2024.

#figure(
image("../figures/3/unet_transformer.png", width: 100%),
caption: [A side-by-side comparison of U-Net and Transformer/ Conformer neural network architectures.],
placement: top,
) <fig_arch_overview>

Underpinning many of these advanced architectures is the Conformer block, which has become a state-of-the-art building block adapted from ASR into TTS @gulati_conformer_2020. The Conformer architecture effectively combines the strengths of Transformers for capturing global context and convolutions for modeling local feature interactions. The convolution module, located between two feed-forward layers within a Transformer block, can model both the long-range semantic and prosodic dependencies and the local acoustic patterns of phonemes with high efficiency.
The U-Net architecture, is also commonly used when a bottleneck in the representation is desirable. Originally developed for biomedical image segmentation @ronneberger_u-net_2015, the U-Net is exceptionally well-suited for denoising tasks where the output must have the same resolution as the input while being informed by multi-scale contextual information. The architecture consists of a downsampling encoder path, which progressively reduces the spatial resolution of the input (e.g., the Mel spectrogram) to capture high-level contextual features, and a symmetric upsampling decoder path, which reconstructs the high-resolution output. The key innovation of the U-Net is the use of skip connections that link the feature maps from the encoder directly to the corresponding layers in the decoder. These connections allow the model to fuse low-level, high-resolution details from the encoder path with the high-level, semantic context from the decoder path. For the diffusion task, this means the model can simultaneously understand the overall structure of the speech it is trying to generate while precisely predicting the fine-grained noise that needs to be removed at each step, a capability that has made it a cornerstone of diffusion-based TTS models like Grad-TTS @popov_gradtts_2021.

==== Training objectives

The training objective, or loss function, guides the optimisation of the model's parameters. Modern TTS systems employ several different objectives, often in combination, to balance fidelity, diversity, and training stability. For simplicity, we omit conditioning $Z$ in this Section.

*Mean Squared Error (MSE) / L1 Loss:* For models that predict a continuous representation like a Mel spectrogram ($R^"MEL" (S)$), the simplest and most common objective is a reconstruction loss. This is typically the Mean Squared Error (L2 loss) or the Mean Absolute Error (L1 loss). The objective is to minimise the distance between the model's prediction $f(T;theta)$ and the ground-truth target:
$ cal(L)_(text("MSE"))(theta)=EE[||S-f(T;theta)||_2^2] $
While stable and easy to optimise, this loss is known to produce "oversmoothed" outputs. By penalising large errors heavily, it encourages the model to predict the average of all plausible outputs, resulting in a loss of fine-grained detail and texture in the final synthesised speech @ren_revisiting_2022. MSE is the primary objective for models like Tacotron 2 and the FastSpeech series.

*Negative Log-Likelihood (NLL) / Cross-Entropy Loss:* This is the standard objective for models that generate discrete sequences, such as the token-based Speech Language Models. After converting the continuous waveform $S$ into a discrete token sequence $s = (s_1, ..., s_K)$, an autoregressive model is trained to predict each token given the previous ones. The training objective is to minimise the NLL of the ground-truth sequence, which is equivalent to maximising the log-probability of the data:

$ cal(L)_(text("NLL"))(theta) = EE_(S) [ - sum_(k=1)^K log p_theta (s_k | bold(s)_(<k)) ] $
This is the fundamental loss used to train powerful token-based models like VALL-E.

*Adversarial Loss:* To improve the perceptual quality of the output, many systems incorporate adversarial training. This involves a discriminator network that is trained to distinguish real from synthesised audio, while the main generator model is trained to produce outputs that can "fool" the discriminator. If the discriminator is powerful enough, this pushes the generator to create outputs that are perceptually closer from real data, capturing fine details that reconstruction losses can miss. GAN training is a core component of vocoders like HiFi-GAN and is also integrated directly into end-to-end TTS models like VITS.

*Diffusion Loss:* The training of diffusion models relies on a specialised objective. The model, $epsilon_theta$, is optimised to predict the noise $epsilon$ that was added to a clean sample $S_0$ to create a noised version $S_k$ at timestep $k$.#footnote[not to be confused with the previous timesteps in a sequence used for AR model training] The simplified loss is typically an MSE loss at each timestep $k$:
$ cal(L)_(text("DDPM"))(theta) = EE_(S_0, bold(epsilon), k) [|bold(epsilon) - f (bold(S)_(k), k;theta)\|_2^2] $
By learning to predict the noise, the model effectively learns the score function of the data distribution. This allows it to generate high-fidelity samples that do not suffer from the oversmoothing issue of direct MSE prediction on the data itself. This is the core objective for diffusion-based TTS models like NaturalSpeech 2. Some state-of-the-art systems, such as StyleTTS 2 @li_styletts_2023, use a combination of these losses—for instance, a reconstruction loss (like L1), an adversarial loss, and perceptual feature matching losses—to achieve the best results.

=== Evaluation of Text-to-Speech Systems <03_eval_tts>

In this section, we outline evaluation methodology for synthetic speech, both #emph[subjective] (determined by raters opinions; potential different outcomes every time) and #emph[objective] (determined by a fixed algorithm, formula or model; same outcome every time with respect to data and parameters). As #abbr.a[TTS] systems have advanced to produce audio close to real speech @eskimez_e2_2024, evaluation methods must evolve to capture nuances beyond basic perceptual quality. Subjective methods rely on human judgment to assess attributes like naturalness, while objective metrics provide automated, repeatable assessments, often correlating with human perception but facing challenges in generalisation @cooper_review_2024.

==== Subjective listening and preference tests <03_subjective>

Here we discuss the most common subjective listening test methodologies and best practices. Subjective tests are the gold standard for synthetic speech evaluation, however, there are drawbacks and trade-offs to any listening test since human behaviour can never be fully anticipated, especially across differing groups of listeners, spans of time, and sets of #abbr.a[TTS] systems. These tests aim to quantify perceptual qualities such as overall naturalness or speaker similarity, but their results are inherently variable due to listener biases and contextual factors @zielinski_bias_2008. Recent advancements in #abbr.a[TTS], where synthetic speech often achieves human parity @chen_vall-e_2024, have made subjective evaluation more challenging, as listeners struggle to distinguish real from synthetic audio.

===== Best practices & drawbacks <03_best_practices>

Before detailing the individual methods, we will outline best practice according to the literature, as well as commonly misunderstood aspects of listening tests, including their drawbacks.

A general best practice is to clearly answer the following questions before conducting a test, as recommended in guidelines for #abbr.a[TTS] evaluation @wester_listeners_2015@kirkland_mospit_2023:
#enum(
[
*Who are the listeners?* Usually, native speakers of the language of the samples are preferred. Additionally, certain tests require a significant number of different speakers to be used, to mitigate biases from individual perceptual differences or fatigue effects @wester_listeners_2015. For example, in large-scale subjective evaluations, annotators are often screened to be native speakers from specific regions and are required to use headphones in a quiet environment, with attention checks embedded in the survey to exclude unreliable responses.
],
[
*What is the setting?* In a lab or online? If possible, requiring use of headphones in a quiet environment is preferred, to minimise external noise interference and ensure consistent audio playback @huang_voicemos_2024. For online crowd-sourced studies, clear instructions on environmental conditions are provided, although direct control over these external factors is inherently limited.
],
[
*What are the instructions?* Until recently, #emph[naturalness] was the most commonly evaluated attribute of synthetic speech, however #emph[quality] is now sometimes preferred @perrotin_blizzard_2025. This ensures listeners evaluate the intended dimensions of the speech.
],
[
*What is "range" of the data?* Is the reference or an anchor included in the set of samples to be evaluated @schoeffler_mushra_2015? Anchors, like degraded audio samples, can calibrate listener expectations and improve rating consistency -- however, they can also "compress" the upper range of ratings and make distinction between top systems difficult and are difficult to define @le_voting_2025. The inclusion of ground truth samples, which are actual human recordings, can allow for a direct comparison of synthetic speech against real speech.
],
[
*What are the concrete stimuli?* How many samples are presented at one time, can listeners re-listen to the same sample? How many points are on the evaluation scale, and are they labelled? Allowing re-listening reduces memory bias, and labelled scales enhance reliability @wells_bws_2024. Is the ground truth reference (if any) labelled as such?
]
)

All these questions should be answered #emph[before] conducting a listening test, and reported in the work to contextualise the results. This transparency is crucial for reproducibility and comparability, as variations in setup can significantly affect outcomes @kirkland_mospit_2023.

There are also drawbacks to subjective evaluation. We will not go into detail on the advantages and disadvantages of every methodology here, but there are two main drawbacks to consider. Firstly, lack of standardisation means there is no standardised framework for most listening test methodologies, beyond the labels and values for particular scales. This means the questions above are answered differently (and often not reported) making it difficult or even impossible to compare results between studies. Subjective listening test results should only ever be compared to results obtained within the same study with the same participants and setup, as listener demographics, fatigue, and even the time of day can influence ratings. Additionally, there is the scale/comparison trade-off -- many methodologies operate by presenting an ordinal scale to listeners on which they rate recordings on, which are then averaged, however this is not necessarily statistically meaningful due to inter-rater variability and non-linear perception of scales. However, when instead using a comparison-based task (in which one sample has to be rated over another), many more comparisons are needed to achieve statistical significance, increasing the test's duration and cost @wells_bws_2024. Furthermore, subjective tests are resource-intensive, prone to biases where overall impressions skew specific ratings, and may not generalise across languages or domains.

===== Mean Opinion Score <03_mos>

#abbr.a[MOS] is the most-commonly used and likely the most-criticised listening test methodology @kirkland_mospit_2023. Raters are asked to rate recordings on a scale from 1 to 5 (Bad to Excellent) @ITU_P800 -- however, this is where the standardisation ends. The original recommendation was intended for evaluation of telecommunication systems and therefore asked listeners to rate #emph[quality], while most, but not all @kirkland_mospit_2023, #abbr.a[TTS] evaluations instead ask for #emph[naturalness] -- sometimes the former is referred to as Q-MOS (Quality MOS) and the latter as N-MOS (Naturalness MOS). There is also no universal agreement for how many stimuli to present at a single time or many of the other questions listed above, and sometimes even the core definition in @ITU_P800 is adapted to, for example, alter the number or type of labels @kirkland_mospit_2023.

In practice, #abbr.a[MOS] involves presenting listeners with isolated audio samples and asking them to rate on the 5-point scale, with the labels being "Bad" (1) "Poor" (2), "Fair" (3), "Good" (4) and "Excellent" (5) @ITU_P800. Scores are then averaged across listeners to yield a system-level #abbr.a[MOS]. This method is widely used in #abbr.a[TTS] challenges, such as the Blizzard Challenge @king_blizzard_2008, where it has been employed to compare systems over time. However, criticisms include its susceptibility to ceiling effects in high-quality #abbr.a[TTS] systems, where scores cluster near 5, reducing discriminative power. Moreover, #abbr.a[MOS] ratings can shift over time as listener expectations evolve; for instance, systems rated highly in 2008 may score lower today due to advancements in #abbr.a[TTS] @le_maguer_back_2022. Despite these issues, #abbr.a[MOS] remains a benchmark for validating objective metrics. For a survey, listeners are typically presented with a set number of audio recordings, such as 6 sets each containing 5 samples of the same text. One of these samples is always a ground-truth recording, while the others are randomly selected synthetic samples.

===== Comparison MOS <03_cmos>

Comparison MOS is an A/B-test-inspired variation of #abbr.a[MOS], in which two samples are presented and the listener is asked to compare and rate on a 7-point scale from -3 (A is much better than B) to +3 (A is much worse than B), with 0 indicating they are equivalent in naturalness (or quality). However, the same standardisation problem is present. E.g., sometimes a 13-point scale (from -6 to +6) is employed @li_styletts_2023 and even the name of the test is sometimes referred to as comparative instead of comparison #abbr.a[MOS].

#abbr.a[CMOS] is particularly useful for fine-grained comparisons between systems, as it directly elicits relative preferences rather than absolute ratings as listeners are forced to make direct comparisons @cooper_review_2024. In #abbr.a[TTS] evaluations, one sample is often the ground truth (real speech), allowing assessment of how closely synthetic speech approaches human quality. This method has gained traction in recent works, such as StyleTTS 2 @li_styletts_2023, where it helped demonstrate improvements over baselines. Unlike #abbr.a[MOS], #abbr.a[CMOS] can reveal subtle differences even when absolute scores saturate, but it requires more pairwise comparisons, increasing listener effort @wells_bws_2024. Studies like the VoiceMOS Challenge @huang_voicemos_2024 have used #abbr.a[CMOS] variants to benchmark systems, showing its robustness in crowdsourced settings. For this methodology, listeners are typically presented with sets of audio samples, where each set consists of two audio recordings (A and B) with same lexical content. One of these is a ground-truth reference, and the other is a synthetic sample from a randomly selected system. The goal is to compare the naturalness of A relative to B, using a scale ranging from -3 (A is much better than B) to +3 (A is much worse than B).

===== Best-Worst Scaling (BWS) <03_bws>

Best-Worst Scaling (BWS) is a forced-choice method where listeners are presented with a set of samples (typically 4) and asked to select the best and worst according to a criterion like naturalness. This implicitly ranks the middle items, providing multiple pairwise comparisons from one judgment. BWS has been applied to #abbr.a[TTS] to address #abbr.a[MOS] limitations, offering more consistent ratings with fewer annotations in tasks like sentiment analysis or summarisation evaluation @wells_bws_2024.

In #abbr.a[TTS], BWS is efficient for comparing multiple systems, as each tuple yields several implicit rankings -- a 4-tuple yields 5 pairwise comparisons. BWS mitigates scale biases in #abbr.a[MOS] by focusing on relative judgments and has shown reliability in low-resource languages like Scottish Gaelic. However, listeners may find it more taxing due to memory demands when comparing audio samples, unlike text-based BWS. Experimental comparisons show BWS provides similar outcomes to #abbr.a[MOS] and AB tests but is perceived as less engaging and more complex.

===== Speaker Similarity MOS <03_smos>

Sometimes we do not want to evaluate the quality or naturalness of the speech, but how closely the speakers in two samples resemble each other. In this case, #abbr.a[SMOS] is commonly used, which operates in the same way as #abbr.a[CMOS], but instead asks listeners to rate how closely the speakers resemble each other.

Typically evaluated on a 5-point scale from 1 (definitely different speakers) to 5 (definitely the same speaker), #abbr.a[SMOS] is essential for voice-cloning #abbr.a[TTS] systems @casanova_xtts_2024. It assesses speaker fidelity in zero-shot scenarios, where a reference utterance guides synthesis. For example, in evaluations of systems like XTTS @casanova_xtts_2024, #abbr.a[SMOS] correlates with objective speaker similarity metrics like cosine distance of embeddings @zhao_probing_2022. Drawbacks include sensitivity to prosodic variations that may mimic speaker changes.

==== Objective Metrics <03_objective_metrics>

Due to the considerable effort and resources required to conduct subjective evaluation, objective metrics are frequently used for #abbr.a[TTS] evaluation, especially for experimental iteration @cooper_review_2024. We group existing metrics into several categories, with the function of distributional metrics being outlined in further detail in @09_dist[Chapter]. In addition to these families we also distinguish #emph[intrusive] and #emph[non-intrusive] metrics. Intrusive metrics require some ground truth speech of the same speaker as a reference. Non-intrusive metrics are reference-free. When the reference does not need to contain the same lexical content, it is described as #emph[non-matching]. These metrics aim to approximate human perception without listener involvement, but their validity must be continually reassessed as #abbr.a[TTS] evolves @moller_quality_2009.

*Signal-Based Reference Metrics:*
The oldest group consists of intrusive metrics that compare each synthetic utterance to a matching reference. #abbr.l[PESQ] @rix_pesq_2001, #abbr.l[STOI] @taal_stoi_2011 and #abbr.l[MCD] @kominek_mcd_2008 are the best–known representatives. They were mostly designed for telephone or enhancement scenarios rather than #abbr.a[TTS], and require access to the ground‑truth waveform.

#abbr.l[PESQ] estimates perceived quality by modeling auditory perception, accounting for distortions like delay and filtering @rix_pesq_2001. #abbr.l[STOI] focuses on intelligibility by correlating short-time spectral envelopes @taal_stoi_2011. #abbr.l[MCD] measures spectral differences using Mel-cepstral coefficients, often used in #abbr.a[TTS] to quantify timbre mismatch @kominek_mcd_2008. These metrics are intrusive, requiring time-aligned references, which limits their use in zero-shot #abbr.a[TTS]. While effective for early systems, they struggle with modern #abbr.a[TTS] systems and do not show significant correlation with subjective evaluation @cooper_review_2024.

*Model-Based:*
To predict scores directly, researchers train neural networks that map a single audio signal to an estimated #abbr.a[MOS]. #emph[MOSNet] @lo_mosnet_2019 introduced the idea, and was followed by #emph[UTMOS] @saeki_utmos_2022, its #abbr.a[SSL]‑based successor #emph[UTMOSv2] @baba_t05_2024, and #emph[NISQA‑MOS] @mittag_nisqa_2021. #emph[SQUIM-MOS] @kumar_torchaudio-squim_2023 additionally grounds its prediction by requiring a non-matching reference of the ground truth speech. These methods report in‑domain correlations; however, recent VoiceMOS challenges @huang_voicemos_2024 show that correlation with subjective ratings decreases in out-of-domain settings. The drawbacks listed in @03_objective_metrics[this section] make training these networks challenging, and generalisation to new #abbr.a[TTS] systems or domains unlikely -- each time substantially different #abbr.a[TTS] systems are introduced, learned metrics will have to be adjusted and verified anew. A recent prediction system goes beyond #abbr.a[MOS] and argues that no single score can capture everything listeners care about. #emph[Audiobox Aesthetics] predicts four axes, Production Quality, Complexity, Enjoyment, and Usefulness, for arbitrary audio @tjandra_meta_2025 -- however, the challenges of learning from subjective labels remain the same, including domain shifts and the need for large, diverse training sets of rated audio.

*Intelligibility:*
Often reported metrics also include #abbr.l[WER] and #abbr.l[CER], computed on #abbr.a[ASR] transcripts. These metrics quantify how easily synthetic speech can be transcribed by an #abbr.a[ASR] system, serving as a proxy for human intelligibility. Lower #abbr.l[WER] indicates clearer pronunciation, but in domains like children's speech, higher #abbr.l[WER] may be realistic. They are non-intrusive but require #abbr.a[ASR] models, and performance depends on the #abbr.a[ASR]'s robustness @radford_robust_2023.

*Speaker Similarity:*
Analogous to #abbr.a[SMOS], cosine similarity between the speaker embeddings (see @02_speaker) of a reference and target speech is frequently reported. Systems like d-vectors @wan_generalized_2018 or ECAPA-TDNN @desplanques_ecapa_2020 compute embeddings, and similarity is computed as the cosine of their angle @zhao_probing_2022. This non-intrusive metric correlates with #abbr.a[SMOS] but may overlook prosodic influences on perceived similarity.

*Distributional:* Inspired by the image domain's #abbr.l[FID] @heusel_fid_2017, audio researchers proposed measuring entire corpora rather than single files. #abbr.l[FAD] @kilgour_fad_2019 compares embeddings and has since been adapted for #abbr.a[TTS] @shi_versa_2024. Distributional metrics require a set of references which do not need to correspond to the synthetic data. The authors of these metrics state the need for thousands of samples, which may be why they have not found more widespread adoption.

=== Cross-Domain and Multilingual Synthesis

While early TTS research typically focused on a single domain -- one speaker reading prepared text in one language -- generalisation and robustness across diverse domains has become more and more important as systems improve. This involves building models that can handle multiple languages, accents, and speaking styles, moving TTS from only producing English, read speech to more versatile and adaptable systems. This also increases the challenges associated with subjective evaluation, as evaluators might be more difficult to come by for low-resource languages @pine_eval_2025 or expert knowledge might be required to rate speech from non-standard domains @fatemeh_enhancement_2024.

==== Multilingual and Cross-Lingual TTS
The challenge of building a single model that can speak multiple languages is substantial. Different languages have distinct phoneme inventories, syntactic structures, and prosodic rules @clark_language_1996. A naive approach of simply pooling data can lead to negative interference, where learning one language degrades performance on another. More sophisticated approaches use a universal, language-agnostic phonetic representation (like the International Phonetic Alphabet, IPA) and provide the model with a learned language ID embedding as part of the conditioning set $Z$ @casanova_yourtts_2022. This allows the model to share statistical strength across languages while still learning language-specific characteristics. The ultimate goal is not just multilingualism (speaking many languages) but also cross-lingual synthesis, where a model can, for instance, synthesise speech in Spanish using the voice characteristics of a monolingual English speaker @badlani_multilingual_2023. This requires models that can effectively disentangle the #smallcaps[Speaker] factor from the #smallcaps[Semantic] and #smallcaps[Prosody] factors tied to a specific language. Large, pre-trained multilingual SSL models like XLS-R @conneau_xlsr_2021 provide a powerful foundation for this, as they learn universal acoustic representations from vast amounts of unlabeled speech across many languages.

==== Cross-domain speaking styles
A parallel challenge is generalising across different speaking styles and acoustic conditions. The vast majority of benchmark TTS datasets, such as LibriTTS @zen_libritts_2019, consist of read speech from audiobooks. This data is acoustically clean, well-structured, and prosodically planned, making it an ideal but unrealistic training target. Real-world human speech is often conversational, a domain characterised by entirely different phenomena. Conversational speech datasets, such as the Emilia dataset @he_emilia_2024, feature filled pauses ("um", "uh"), disfluencies, self-corrections, overlapping speech between interlocutors, and highly dynamic, spontaneous prosody. Furthermore, the acoustic conditions are far more variable, with background noise and reverberation. Training a TTS model on such data is significantly more challenging. A model trained on read speech will sound unnaturally formal and robotic in a conversational context. Conversely, a model trained on conversational speech must learn to generate these "imperfections" in a natural way to be perceived as authentic. Cross-domain settings require not only more robust architectures but also a re-evaluation of objective evaluation correlation with subjective scores, if they were established for a different domain, which might otherwise penalise the phenomena that occur in the new domain.

Successfully modeling these diverse domains is a key step towards creating truly interactive and human-like synthetic voices.
While this diversity across languages and domains is crucial, in this work, we also investigate how diverse synthetic speech can be within the relatively constrained domain of English read speech, and how this diversity can be increased, as is detailed in the next chapters.