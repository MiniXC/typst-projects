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

=== Forms of speech synthesis

In our work, we focus on #abbr.a[TTS], however there are other paradigms for speech generation, the most common being #emph[Textless Generation] and #emph[Voice Conversion]. @fig_tts_vc_textless visualises these differing approaches.

==== Textless Generation

Beyond text-conditioned synthesis, these models produce speech without any explicit or implicit semantic input, sampling instead from a learned distribution of speech sounds. Thus, these models aim to learn the underlying probability distribution $Q(S)$ directly. Early neural approaches, such as the original WaveNet architecture, demonstrated this by using an autoregressive model to predict low-level #smallcaps[Generic] raw audio samples one at a time, conditioned only on previous samples @oord_wavenet_2016. Textless models are foundational for tasks like audio completion or infilling but they require careful handling of sequence length and long-term coherence to avoid devolving into unstructured noise @tan_survey_2021. Recently, this form of generation has seen increased interest due to #abbr.pla[SLM] @lakhotia_gslm_2021 where a #abbr.a[LLM]-like system is trained to predict high-level acoustic units, usually contextualised speech embeddings -- however, even the latest systems still lack semantic coherence, with the text extracted from speech produced by these models achieving $<20%$ "win rate" against real text @park_speechssm_2025.

==== Voice Conversion

While TTS maps a #smallcaps[Semantic] representation ($T$) to a #smallcaps[Generic] acoustic one ($S$), the related task of #emph[#abbr.a[VC]] transforms an input speech signal from a source style to a target style while preserving the linguistic content. VC conditions on a source utterance $S^"SRC"$ rather than text $T$:

$
tilde(S) tilde Q_theta (S | S^"SRC", Z)
$

Here, the conditioning set $Z$ typically contains a high-level #smallcaps[Speaker] embedding representing the target voice. A key challenge in VC is to effectively disentangle the #smallcaps[Semantic] content of the speech from the #smallcaps[Speaker] and #smallcaps[Prosody] factors, often using specialized feature extractors or autoencoder-based architectures @sisman_vc_2020. VC enables applications like real-time voice modification and dubbing, where the goal is to change the voice characteristics without relying on a transcript.

#figure(
  diagram(
    spacing: 5pt,
    cell-size: (5mm, 10mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 70%,
    
    // Unconditional Generation (NEW)
    node((-2, 3.2), align(center)[$Z$]),
    edge((-2, 3.2), (-2, 2), "--|>"), // Optional conditioning (e.g., random seed)
    blob((-2, 2), [Textless Generation], tint: orange, width: 42mm),
    edge((-2, 2), (-2, 0.9), "-|>"),
    node((-2, 0.9), $tilde(S)$),

    // voice conversion
    node((-.37, 3.2), align(center)[$Z$]),
    edge((-.37, 3), (0, 2), "--|>", bend: -15deg),
    node((0, 3), $S^text("SRC")$),
    edge((0, 3), (0, 2), "-|>"),

    blob((0, 2), [Voice Conversion], tint: orange, width: 42mm),
    edge((0, 2), (0, 0.9), "-|>"),
    node((0, 0.9), $tilde(S)$),

    // tts
    node((1.6, 3.2), align(center)[$Z$]),
    edge((1.6, 3.2), (2, 2), "--|>", bend: -15deg),
    node((2, 3), $T$),
    edge((2, 3), (2, 2), "-|>"),
    blob((2, 2), [Text-to-Speech], tint: orange, width: 42mm),
    edge((2, 2), (2, 0.9), "-|>"),
    node((2, 0.9), $tilde(S)$)
  ),
  placement: top,
  caption: "Comparison of Text-to-Speech (TTS), Voice Conversion (VC), and Textless Generation.",
) <fig_tts_vc_textless>

==== Text-to-Speech

For this work, we focus on TTS due to the abundance of openly available TTS models compared to #abbr.pla[VC] or #abbr.pla[SLM] as well as due to their controllability -- being able to condition on separate #smallcaps[Semantic], #smallcaps[Speaker] and optionally #smallcaps[Prosody] representations allows us to investigate the synthetic speech in more controlled experimental setups. Systems which condition on at least the former two are understood as multi-speaker, voice-cloning TTS systems, which make up the majority of modern TTS systems @cooper_review_2024. The #smallcaps[Semantic] representation is text or derived from text, such as contextualised text embeddings or phones. The #smallcaps[Speaker] representation is most commonly a high-level learned speaker embedding (see @02_speaker), but can also be another representation containing salient speaker information, such as a mid-level Mel spectrogram of a reference utterance @eskimez_e2_2024 or even a high-level text prompt describing their identifying characteristics @lyth_parler_2024.

=== Hierarchy of Text-to-Speech

Most TTS systems have a distinct #emph[frontend], transforming the input text and a #emph[vocoder], converting some speech representation to a waveform. These components are useful since the "raw" text $T$ and waveform $S$ are not ideal representations for modeling speech.

==== Frontend
Text can be ambiguous, for example in terms of pronunciation, or representation of numbers and dates.
Which makes the frontend, or text analysis module, a critical component of many TTS systems. Its purpose is to transform raw, unstructured text into a clean, structured linguistic feature representation suitable for the acoustic model @taylor_tts_2009. This process is fundamentally about converting the highest-level #smallcaps[Semantic] representations into a format that can be more easily mapped to acoustics. This can involve the following:

*Text Normalisation:* This stage converts non-standard words, such as numbers, abbreviations, and symbols, into their full written form. For example, "1989" becomes "nineteen eighty-nine" and "Dr." becomes "Doctor." This step can help to ensuring correct pronunciation.

*Linguistic Analysis:* This can include tasks like part-of-speech (POS) tagging to resolve pronunciation ambiguities (e.g., "read" as /riːd/ vs. /rɛd/) and word segmentation for languages like Chinese that do not use spaces.

*Grapheme-to-Phoneme (G2P) Conversion:* This is often the final step, converting the normalized text into a mid-level #smallcaps[Semantic] representation: the phone sequence. This mapping from orthography to a phonetic representation like /s p iː tʃ/ is essential for accurate pronunciation, particularly in languages with irregular orthography like English or with polyphones in Chinese @yao_g2p_2015.

*Text Tokenization:* While traditional systems used complex, multi-stage, rule-based frontends, modern neural TTS systems have simplified this process. Many models now operate directly on character sequences, only tokenizing the text @hayashi_espnet-tts_2020. A recent development is the use of #abbr.a[SSL] models to extract phonetic representations directly from audio, which can then be used to train TTS systems, bypassing traditional G2P altogether.

==== Vocoder

Raw waveforms are high-resolution, low-dimension representations, with typically 16- to 44-thousand values representing each second of audio. Generative models therefore are often tasked to predict a more low-resolution representation with higher dimensionality, usually a lossy-reconstructible mid-level #smallcaps[Generic] representation like a Mel spectrogram, which is then converted to a raw waveform using a vocoder. Legacy systems rely on algorithmic vocoders based on the source-filter model. LPC Vocoders @atal_lpc_1970, originally developed for telephony, provide a computationally cheap but robotic-sounding output. Source-Filter Vocoders like WORLD @morise_world_2016, represented an improvement by modeling the source (F0, aperiodicity) and filter (spectral envelope) to produce much more natural, albeit sometimes "buzzy" speech for parametric systems. Early end-to-end models like Tacotron, which were among the first to directly produce mel spectrograms successfully, used the algorithmic Griffin-Lim method @griffin_griffinlim_1984 to iteratively estimate phase and reconstruct a waveform.

With deep learning, neural vocoders got were introduced, leading to a dramatic leap in quality. Among the first were Simple Autoregressive models like WaveNet @oord_wavenet_2016, which generate audio sample-by-sample, achieving human-level fidelity at the cost of extremely slow inference. To address this, parallel models were developed. Flow-based vocoders like WaveGlow @prenger_waveglow_2019 use normalizing flows for fast, high-quality parallel generation. However, GAN-based vocoders have become the dominant approach for their exceptional balance of speed and quality. Systems like MelGAN @kumar_melgan_2019, HiFi-GAN @kong_hifigan_2020 or BigVGAN @lee_bigvgan_2023 use a generator to create audio from a Mel spectrogram and a discriminator to ensure its perceptual realism, enabling real-time, high-fidelity synthesis. Diffusion-based vocoders such as DiffWave @kong_diffwave_2021, are another high-quality parallel approach, learning to reverse a noising process, though they can be slower than GANs. The most recent development involves using the decoder of a pre-trained Neural Audio Codec as the vocoder. Models like EnCodec @defossez_encodec_2023 learn to tokenize audio into a discrete representation, and their decoders can perfectly reconstruct a waveform from these tokens, a technique leveraged by the latest generation of TTS models.

=== History of Text-to-Speech

#figure(
  image("../figures/3/tts_timeline.png", width: 100%),
  caption: [Non-exhaustive timeline of TTS systems, as well as frontend and vocoder technologies used.],
  placement: top,
) <fig_tts_timeline>

The evolution of TTS, as visualised in @fig_tts_timeline, reflects a progression in architectures and the representations they employ, moving from complex pipelines to more integrated, end-to-end systems.

The legacy era was defined by modular systems. *Concatenative synthesis*, exemplified by the Festival @taylor_festival_1998 toolkit, used a frontend with full linguistic analysis to select raw waveform chunks from a database. As it concatenated existing audio, it did not require a vocoder. In parallel, *Statistical Parametric Speech Synthesis* (SPSS), seen in HTS @tokuda_hts_2013 and later the DNN-based Merlin toolkit @wu_merlin_2016, also used a complex linguistic frontend but generate parameters for a Source-Filter vocoder like WORLD @morise_world_2016.

The first wave of end-to-end neural models simplifies the frontend to basic tokenization and focuses on generating spectrograms. Tacotron @wang_tacotron_2017 is a autoregressive model that used the algorithmic Griffin-Lim @griffin_griffinlim_1984 method as its vocoder. The major quality breakthrough comes with Tacotron 2 @shen_natural_2018, which pairs a similar spectrogram-prediction model with a powerful, though slow, Simple Autoregressive neural vocoder based on WaveNet @oord_wavenet_2016.

The subsequent drive for efficiency leads to a family of non-autoregressive models that generate Mel spectrograms in parallel. These systems, including FastSpeech, FastSpeech2, and GlowTTS, offload the final waveform generation to a separately trained, fast neural vocoder. As @fig_tts_timeline shows, these models can be paired with various Mel Spectrogram Based vocoders, with GAN-based models like HiFi-GAN @kong_hifigan_2020 being the most common choice due to their speed and quality. Diffusion-based and Flow-based vocoders are also viable options for this architecture.

Recent developments have focused on integrated end-to-end approaches and output representations. VITS @kim_vits_2021 represents a true end-to-end approach, jointly learning to align text and generate a raw waveform directly within a single model, thus having its own integrated vocoder. Concurrently, diffusion models have been applied to the main TTS task in systems like NaturalSpeech2 @tan_naturalspeech_2024 and StyleTTS2 @li_styletts_2023, which typically generate a high-quality Mel spectrogram that is then rendered by a powerful GAN-based vocoder. The latest paradigm shift, represented by VALL-E @wang_valle_2023 and ParlerTTS @lyth_parler_2024, abandons spectrograms entirely. These systems use a simplified frontend, sometimes leveraging an SSL Phoneme Model, Phoneme-level BERT @li_plbert_2023, for robust phonetic representations, and their acoustic model predicts a sequence of discrete tokens. The final waveform is then synthesized using a Neural Audio Codec decoder as the vocoder. However, alignment-free diffusion models producing mel spectrograms like E2 @eskimez_e2_2024 or F5 @chen_f5_2024 have achieved similar performance.

=== Architectures and Training Objectives

The development of modern TTS has been driven by the exploration of diverse neural network architectures and the training objectives used to optimize them. While legacy systems were modular by necessity, current state-of-the-art approaches are predominantly #abbr.pla[DNN]. These systems typically take as input a #smallcaps[Semantic] representation in the form of a character or phoneme sequence, and are trained to produce a #smallcaps[Generic] acoustic representation as output, most commonly a Mel spectrogram or a sequence of discrete audio codec tokens. This section details the primary architectural paradigms that define how these models generate speech, and the training objectives used to guide their learning process.

==== Architectural Paradigms <03_arch>

The architecture of a TTS model dictates how it transforms input text into an acoustic representation. The most fundamental distinction is whether the model generates its output autoregressively or non-autoregressively. The building blocks of these architectures have also evolved, with Transformers and Conformers becoming standard components.

*Autoregressive (AR) Models:* These models generate the output sequence one step at a time, conditioning each step on the previously generated outputs. If we let the input linguistic sequence be $T$ and the target acoustic sequence be $S = (s_1, ..., s_n)$, an AR model factorizes the probability sequentially:
$ p(S|T) = product_(k=1)^K p(s_k|s_1,dots,s_(k-1),T) $
This sequential dependency allows AR models to capture complex, long-range correlations in the speech signal. For example, Tacotron 2 @shen_natural_2018, uses an attention mechanism to solve the alignment problem dynamically. At each decoder step $i$, the attention mechanism computes a set of alignment weights $alpha_i$, which determine how much "attention" to pay to each input embedding from the text encoder when generating the current audio frame. These weights are calculated based on the previous decoder state $d_(i-1)$ and the encoder outputs $h$:
$
alpha_i = "Attention"(d_(i-1), h)
$
The attention module calculates a context vector $c_i$ as the weighted sum of the encoder outputs, $c_i = sum_j alpha_(i,j) h_j$, which is then used to predict the output frame $s_i$. This process allows the model to dynamically learn the alignment between the text and the much longer audio sequence. While this approach produces high-quality speech, its sequential nature makes inference slow and can lead to error propagation. More recently, this paradigm has been applied to discrete token generation in models like VALL-E @wang_valle_2023, which use a Transformer architecture to autoregressively predict neural codec tokens.

*Non-Autoregressive (NAR) Models:* To overcome the slow inference of AR models, NAR models were developed to generate the entire output sequence in parallel. These models assume conditional independence between the output frames given the full input conditioning:
$ p(S|T) = product_(k=1)^K p(s_k|T) $
This parallel generation is extremely fast but presents a new challenge: aligning the variable-length input text sequence with the much longer output acoustic sequence. NAR models can be further subdivided based on how they solve this alignment problem.
Among non-autoregressive models, #emph[Explicit Duration-Based Models] address the alignment challenge directly by predicting the duration for each input unit (typically a phone). An early example, FastSpeech @ren_fastspeech_2019, introduced the Transformer architecture to NAR TTS. The Transformer's self-attention mechanism is highly effective at modeling long-range dependencies within the text sequence, which is crucial for producing coherent prosody. FastSpeech uses a "Length Regulator" to expand the input sequence according to predicted durations. The more robust and widely adopted approach, seen in FastSpeech 2 @ren_fastspeech_2021, integrates this into a Variance Adaptor module, which predicts low-level #smallcaps[Prosody] correlates like duration, pitch, and energy.
#emph[Implicit Alignment Models] represent a more advanced class of NAR architectures that learn the alignment end-to-end. Glow-TTS @kim_glowtts_2020 achieved this using normalizing flows coupled with a Monotonic Alignment Search mechanism. A highly successful example is VITS @kim_vits_2021, which uses a Variational Autoencoder (VAE) framework to jointly learn a stochastic duration predictor and an alignment between the latent text and speech representations, guided by a connectionist temporal classification (CTC)-style loss. This allows the model to be trained end-to-end while discovering the alignment on its own. A recent work simplifies this further by only predicting the total utterance length, making prediction of phoneme durations fully implicit @eskimez_e2_2024.

// add info on encoder-decoder transformers?

#figure(
image("../figures/3/unet_transformer.png", width: 100%),
caption: [A side-by-side comparison of U-Net and Transformer/ Conformer neural network architectures.],
placement: top,
) <fig_arch_overview>

Underpinning many of these advanced architectures is the Conformer block, which has become a state-of-the-art building block adapted from ASR into TTS @gulati_conformer_2020. The Conformer architecture effectively combines the strengths of Transformers for capturing global context and convolutions for modeling local feature interactions. By sandwiching a convolution module between two feed-forward layers within a Transformer block, it can model both the long-range semantic and prosodic dependencies and the local acoustic patterns of phonemes with high efficiency.
The U-Net architecture, is also commonly used when a bottleneck in the representation is desirable. Originally developed for biomedical image segmentation @ronneberger_u-net_2015, the U-Net is exceptionally well-suited for denoising tasks where the output must have the same resolution as the input while being informed by multi-scale contextual information. The architecture consists of a downsampling encoder path, which progressively reduces the spatial resolution of the input (e.g., the Mel spectrogram) to capture high-level contextual features, and a symmetric upsampling decoder path, which reconstructs the high-resolution output. The key innovation of the U-Net is the use of skip connections that link the feature maps from the encoder directly to the corresponding layers in the decoder. These connections allow the model to fuse low-level, high-resolution details from the encoder path with the high-level, semantic context from the decoder path. For the diffusion task, this means the model can simultaneously understand the overall structure of the speech it is trying to generate while precisely predicting the fine-grained noise that needs to be removed at each step, a capability that has made it a cornerstone of diffusion-based TTS models like Grad-TTS @popov_gradtts_2021.

==== Training Objectives

The training objective, or loss function, guides the optimization of the model's parameters. Modern TTS systems employ several different objectives, often in combination, to balance fidelity, diversity, and training stability. For simplicity, we omit conditioning $Z$ in this chapter.

*Mean Squared Error (MSE) / L1 Loss:* For models that predict a continuous representation like a Mel spectrogram ($R^"MEL" (S)$), the simplest and most common objective is a reconstruction loss. This is typically the Mean Squared Error (L2 loss) or the Mean Absolute Error (L1 loss). The objective is to minimize the distance between the model's prediction $f(T;theta)$ and the ground-truth target:
$ cal(L)_(text("MSE"))(theta)=EE[||S-f(T;theta)||_2^2] $
While stable and easy to optimize, this loss is known to produce "oversmoothed" outputs. By penalizing large errors heavily, it encourages the model to predict the average of all plausible outputs, resulting in a loss of fine-grained detail and texture in the final synthesized speech @ren_revisiting_2022. It was the primary objective for models like Tacotron 2 and the FastSpeech series.

*Negative Log-Likelihood (NLL) / Cross-Entropy Loss:* This is the standard objective for models that generate discrete sequences, such as the token-based Speech Language Models. After converting the continuous waveform $S$ into a discrete token sequence $s = (s_1, ..., s_K)$, an autoregressive model is trained to predict each token given the previous ones. The training objective is to minimize the NLL of the ground-truth sequence, which is equivalent to maximizing the log-probability of the data:
$ cal(L)_(text("NLL"))(theta) = EE_(S) [ - sum_(k=1)^K log p_theta (s_k | bold(s)_(<k)) ] $. This is the fundamental loss used to train powerful token-based models like VALL-E.

*Adversarial Loss:* To improve the perceptual quality of the output, many systems incorporate adversarial training. This involves a discriminator network that is trained to distinguish real from synthesized audio, while the main generator model is trained to produce outputs that can "fool" the discriminator. If the discriminator is powerful enough, this pushes the generator to create outputs that are perceptually closer from real data, capturing fine details that reconstruction losses can miss. GAN training is a core component of vocoders like HiFi-GAN and is also integrated directly into end-to-end TTS models like VITS.

*Diffusion Loss:* The training of diffusion models relies on a specialized objective. The model, $epsilon_theta$, is optimized to predict the noise $epsilon$ that was added to a clean sample $S_0$ to create a noised version $S_k$ at timestep#footnote[not to be confusied with the previous timesteps in a sequence used for AR model training] $k$. The simplified loss is typically an MSE loss at each timestep $k$:
$ cal(L)_(text("DDPM"))(theta) = EE_(S_0, bold(epsilon), k) [|bold(epsilon) - f (bold(S)_(k), k;theta)\|_2^2] $
By learning to predict the noise, the model effectively learns the score function of the data distribution. This allows it to generate high-fidelity samples that do not suffer from the oversmoothing issue of direct MSE prediction on the data itself. This is the core objective for diffusion-based TTS models like NaturalSpeech 2. Some state-of-the-art systems, such as StyleTTS 2 @li_styletts_2023, use a combination of these losses—for instance, a reconstruction loss (like L1), an adversarial loss, and perceptual feature matching losses—to achieve the best results.

=== Cross-Domain and Multilingual Synthesis

While early TTS research typically focused on a single domain -- one speaker reading prepared text in one language -- generalization and robustness across diverse domains has become more and more important as systems improve. This involves building models that can handle multiple languages, accents, and speaking styles, moving TTS from only producing English, read speech to more versatile and adaptable systems.

==== Multilingual and Cross-Lingual TTS
The challenge of building a single model that can speak multiple languages is substantial. Different languages have distinct phoneme inventories, syntactic structures, and prosodic rules @clark_language_1996. A naive approach of simply pooling data can lead to negative interference, where learning one language degrades performance on another. More sophisticated approaches use a universal, language-agnostic phonetic representation (like the International Phonetic Alphabet, IPA) and provide the model with a learned language ID embedding as part of the conditioning set $Z$ @casanova_yourtts_2022. This allows the model to share statistical strength across languages while still learning language-specific characteristics. The ultimate goal is not just multilingualism (speaking many languages) but also cross-lingual synthesis, where a model can, for instance, synthesize speech in Spanish using the voice characteristics of a monolingual English speaker @badlani_multilingual_2023. This requires models that can effectively disentangle the #smallcaps[Speaker] factor from the #smallcaps[Semantic] and #smallcaps[Prosody] factors tied to a specific language. Large, pre-trained multilingual SSL models like XLS-R @conneau_xlsr_2021 provide a powerful foundation for this, as they learn universal acoustic representations from vast amounts of unlabeled speech across many languages.

==== Cross-Domain Speaking Styles
A parallel challenge is generalizing across different speaking styles and acoustic conditions. The vast majority of benchmark TTS datasets, such as LibriTTS @zen_libritts_2019, consist of read speech from audiobooks. This data is acoustically clean, well-structured, and prosodically planned, making it an ideal but unrealistic training target. Real-world human speech is often conversational, a domain characterized by entirely different phenomena. Conversational speech datasets, such as the Emilia dataset, feature filled pauses ("um", "uh"), disfluencies, self-corrections, overlapping speech between interlocutors, and highly dynamic, spontaneous prosody. Furthermore, the acoustic conditions are far more variable, with background noise and reverberation. Training a TTS model on such data is significantly more challenging. A model trained on read speech will sound unnaturally formal and robotic in a conversational context. Conversely, a model trained on conversational speech must learn to generate these "imperfections" in a natural way to be perceived as authentic. This requires not only more robust architectures but also a re-evaluation of training objectives, which might otherwise penalize the very phenomena that define conversational speech. Successfully modeling these diverse domains is a key step towards creating truly interactive and human-like synthetic voices.
While this diversity across languages and domains is crucial, in this work, we also investigate how diverse synthetic speech can be within the relatively constrained domain of read, English speech, and how this diversity can be increased, as is detailed in the next chapters.