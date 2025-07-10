#import "../abbr.typ" 
#import "../quote.typ": *
#import "../math.typ": *
#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#import fletcher.shapes: house, hexagon
#let blob(pos, label, tint: white, width: 26mm, ..args) = node(
	pos, align(center, label),
	width: width,
	fill: tint.lighten(60%),
	stroke: 1pt + tint.darken(20%),
	corner-radius: 5pt,
	..args,
)

== Introduction <01_intro>

#q(
  [#citep(<taylor_tts_2009>)], 
  [#emph[Text-to-Speech Synthesis]],
  [… it is normally fairly
  easy to tell that it is a computer talking rather than a human, and so substantial progress is still to
  be made.]
)

Substantial progress has been made in #abbr.a[TTS] -- it is now often impossible to tell if a computer or human is talking to listeners -- at the outset of our research, this was not necessarily the case, but since then, several works have claimed to reach parity with real speech @tan_naturalspeech_2024@chen_vall-e_2024@eskimez_e2_2024. The main lens we view synthetic speech as in this work is as a distribution -- and the way human parity is defined by previous work is devoid of that concept. This, in our opinion, incomplete definition might lead to speech that on an individual, utterance-by-utterance level is on par with real speech, but fails to capture the breadth and diversity present in reality. 

#link(<part_01>, [Part I]) introduces background information necessary for this work, in the form of speech representations (@02_factors), #abbr.a[TTS] (@03_tts) and #abbr.a[ASR] (@04_asr).

In #link(<part_01>, [Part II]) our main contributions are:
- Investigating how big the gap between real and synthetic speech is with respect to ASR performance -- we show it is more significant than would be suggested by subjective evaluation.
- Presenting ways to improve this diversity by conditioning (@06_attr) and scaling (@07_scaling), and how there seems to be a ceiling on this improvement not explained by either of these approaches.

In #link(<part_02>, [Part III]) our main contributions are:
- Introducing #abbr.a[TTSDS], an objective measure which utilises the distributional distances between real and synthetic speech across a range of speech representations.
- Validating this measure by showing correlation with longitudinal listening test data (from 2008 to 2024) as well as for systems reaching human parity across a range of domains.

=== Notation

In machine learning, it is convention to denote $X$ as model input and $Y$ as model output, with a model $f$ with parameters $theta$ predicting $Y$ using $X$, such that $Y=f_theta (X)$. However, with speech synthesis and recognition in the same context, this can be ambiguous, as $X$ and $Y$ can either represent speech signals or transcripts. To maintain clarity throughout this work, we will denote speech signals as $𝑆$ and transcripts as $𝑇$. 

When both synthetic and real data are addressed in the same context, we use $tilde$ to denote the synthetic counterpart, for example $Syn$ for a set of speech signals or $stheta$ for #abbr.a[ASR] model parameters derived from synthetic speech.

==== Representations as transformations

We consider the raw audio waveform as the fundamental representation of a speech signal $S$ -- these can have different fidelity, in the form of sampling rate and bit depth. All other representations are derived by applying a transformation function, denoted as $cal(R)$, to this waveform. These transformations are categorized by the nature and purpose of the resulting representation.


*Invertible Transformations:* These are lossless transformations where the original waveform can be perfectly reconstructed, i.e., $cal(R)^(-1)(cal(R)(S)) = S$. An example is the Short-Time Fourier Transform (#abbr.a[STFT]), which retains both magnitude and phase, allowing for perfect reconstruction via the inverse #abbr.a[STFT].

*Reconstructible Transformations:* These transformations are intentionally lossy but are designed to retain sufficient information for a high-fidelity, perceptually similar waveform to be synthesized. The inverse operation is an approximation, often performed by a dedicated model or algorithm $f_theta$ (e.g., a vocoder or decoder), such that $f_theta (cal(R)(S)) = Syn$. Note the resulting speech is now classed as synthetic, since it is not reconstructed perfectly and subject to change based on the model or algorithm used. Examples of such transformations are Mel spectrograms or autoencoder-based audio codecs @defossez_encodec_2023@kumar_dac_2023.

*Reductive Transformations:* These transformations distill specific attributes from the signal into a representation that is not intended for direct audio reconstruction. The output can range from a time-series of vectors to a single scalar value. This could be a series of embeddings extracted from a #abbr.a[SSL] model, a single speaker embedding, the detected pitch over time or even written description of the audio by a human.

==== Conditioning

Whenever we use more than just text transcripts ($T$) to condition speech generation with, we group all secondary conditioning sets into $cal(C)$. They are usually reductive transformations of the signal as outlined above. We denote embedding-based ones as $E_dots$ (e.g. $E_"SPK"$ for speaker embeddings), audio-based as $A_dots$ (e.g. $A_"REF"$ for reference speech in lieu of speaker embeddings) textual ones as $T_dots$ (e.g. textual "style prompts" in ParlerTTS by #citea(<lyth_parler_2024>) are $T_"PRM"$) and numerical ones as $F_dots$ (e.g. $F_"F0"$ for pitch).


=== Text-to-speech

#figure(
  diagram(
    spacing: 5pt,
    cell-size: (8mm, 10mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 70%,

    // legend
    node((-1,3.2), align(left)[$T$ ... transcripts], width: 40mm),
    node((-1,2.6), align(left)[$S$ ... speech signals], width: 40mm),
    node((-1,1.7), align(left)[$cal(C)$ ... conditioning (e.g. style, speaker, speaking rate, #sym.dots.h)], width: 40mm), 
  

    // tts

    //subsets of C
    node((1.1,4), $E_"SPK"$),
    node((1.45,4), $dots.h$),
    node((1.8,4), $T_"PRM"$),

    // arrows to subsets of C
    edge((1.45,3.1), (1.8,3.8), "--|>", bend: 40deg),
    edge((1.45,3.1), (1.45,3.8), "--|>"),
    edge((1.45,3.1), (1.1,3.8), "--|>", bend: -40deg),

    
    node((1.45,3), align(center)[$cal(C)$]),
    edge((1.45,3), (2,2), "--|>", bend: -15deg),
    node((2,3), $T$),
    edge((2,3), (2,2), "-|>"),
    blob((2,2), [Text-to-Speech], tint: orange, width: 40mm),
    edge((2,2), (2,1), "-|>"),
    node((2,1), $Syn$)
  ),
  placement: none,
  caption: "Text-to-Speech (TTS) inputs and output.",
) <fig_tts>

#abbr.l[TTS] systems, as shown in @fig_tts, map text input $T$ to corresponding speech waveforms $S$ using a model or chain of models $f_theta (T)=S$, where $theta$ represents the model parameters trained on large-scale paired text-speech datasets. Multi-speaker systems need some form of speaker representation in $cal(C)$, such as reference audios ($A_"REF"$) or speaker embeddings ($E_"SPK"$). Other conditioning can include prosodic correlates such as pitch ($F_"F0"$), and learned style vectors $E_"STY"$ @nose_style_2007.

==== Other speech generation approaches
#abbr.l[VC] systems transform a source speaker's speech $S$ into a target speaker's voice (represented as $E_"SPK"$ or $A_"SPK"$) using a model $g_phi (S,cal(C))=Syn$, where $phi$ represents the model parameters trained on paired or unpaired speech data from multiple speakers @mohammadi_vc_2017.

Speech can also be generated unconditionally, not specifying no lexical content $T$ or other conditioning $cal(C)$ @childers_vc_1989. Some models are also capable of both unconditionally and conditionally generating speech, or performing #abbr.a[TTS] and #abbr.a[VC] @le_voicebox_2023.

Yet another set of approaches, articulatory synthesis, attempts to approximate a physical model of the vocal tract to generate speech. However, this is most commonly used for analysis rather than synthesis, since the quality is not comparable to learned methods @panchapagesan_aai_2011.


==== Voice-cloning, learned #abbr.a[TTS]

We constrain this work to voice-cloning #abbr.a[TTS], necessitating some speaker representation in $cal(C)$. We mostly use and investigate learned methods, where a set of parameters $theta$ of a #abbr.a[DNN] is optimized using some loss which is intended to minimize the differences between the synthetic and real speech.

If we compare this with speech in the real world, there is no speech without communicative intent @searle_speechacts_1969@clark_language_1996 -- in #abbr.a[TTS], this represented by the text. At the same time, there can be no speech without a speaker. In #abbr.a[TTS], this usually encompassed by some speaker representation, which then makes it #emph[voice-cloning] #abbr.a[TTS].

 While there are many more factors that affect speech -- since speech often encompasses much more than a string of words uttered by a specific speaker @hinde_nvc_1972 -- the lexical content and speaker identity are unambiguously present in every case. Voice-cloning #abbr.a[TTS] at its most basic attempts to model speech based on this most barebones conditioning on text and speaker identity -- and we use this as our starting point for synthetic speech in this work.

=== The distribution gap

#figure(
  image(
    "../figures/1/syntpp_figure.png",
    alt: "A line graph comparing synthetic data distribution (green curve) and true data distribution (black curve). The synthetic distribution peaks earlier than the true distribution and does not fully cover its range. The x-axis is divided into four labelled regions: 'artifacts' on the far left where only the synthetic distribution has values, 'over-sampled' where the synthetic peak is higher than the true distribution, 'under-sampled' where the true distribution peak is higher than the synthetic distribution, and 'missing samples' on the far right where only the true distribution has values."
  ),
  caption: [Visualisation of hypothesised distribution of synthetic speech compared to real speech. Some speech is completely unrealistic (artifacts), while some is over- or undersampled, or cannot be generated at all. Figure by #citep(<hu_syntpp_2022>).],
  placement: none,
)

A frequently-mentioned difficulty of matching the distributions of the real and synthetic speech is the one-to-many problem @ren_fastspeech_2019 -- one combination of speaker identity and text can have many possible realisations. The fundamental impossibility of modelling all possible realisations without having a perfect model of each speaker leads to some fundamental differences between real and synthetic speech distributions: some speech might be produced which could never be uttered by a human -- think the classic robotic sound caused by certain vocoders @hu_syntpp_2022, but these could also be more subtle and imperceptible by listeners such as is explicitly the goal for speech watermarking @nematollahi_watermarking_2013. Some speech might be more common in the synthetic data distribution than the real data distribution. And yet more speech might be present in the real distribution but not, or rarely, in the synthetic one.

The goal of this work is to investigate how these distributions differ in #emph[meaningful] ways -- that is, in ways that matter to human listeners. For example, it might be interesting that a specific algorithmic metric, such as #abbr.a[MCD] between synthetic and real speech is lower for one #abbr.a[TTS] system than for another. However, in this work, we only focus on such differences if they do affect listeners' perceptions of the speech.

=== TTS-for-ASR evaluation

To begin our investigation in the distribution gap, the task of using synthetic speech for #abbr.a[ASR] model training is a good proxy for said distributions. While the distributions do not have to match exactly to lead to good #abbr.a[ASR] training, if they did match exactly, we would expect equivalent results. Another reason this is a meaningful proxy is that the ability to learn speech recognition from speech is a fundamental task which every human goes through as well, and in some ways, at least on a phonetic level, perception of speech functions similarly between humans and #abbr.a[DNN] #abbr.a[ASR] systems @schatz_neuralperception_2018.

The research questions we ask in this area are:
- How big is the distribution gap for #abbr.a[TTS]-for-#abbr.a[ASR] compared to subjective evaluation results?
- Is this distance explained by some fundamental missing factor on the #abbr.a[TTS] side, such as distributional conditioning (@06_attr) and scaling (@07_scaling)?

While we find that these changes improve #abbr.a[TTS]-for-#abbr.a[ASR] performance, conditioning only explains a small part of the gap, and our research on scaling indicates diminishing returns. 

=== Text-to-speech distribution score

#abbr.a[TTS]-for-#abbr.a[ASR] is computationally expensive and complex (two models have to be trained, after all) and does not offer explicit explanations as to which parts or components of the speech might be lacking. For these reasons, we move on to introduce an evaluation framework which is more efficient and can give us insight into the individual factors.

Most #abbr.a[TTS] evaluation paradigms at the time of writing are paired sample-to-sample comparisons, no matter if conducting #emph[subjective] (asking people to compare) or #emph[objective] (using algorithms or models) evaluation. But if we think of speech as a distribution, this makes little sense -- when comparing two utterances, even if they contain the same content and are produced by the same speaker, they could sit at two wildly different, but equally valid, locations of the distribution space. For subjective evaluation we expect real speakers to be able to compensate for this if we ask them the right questions -- after all, they should be able to tell they are both natural. But can we expect this to be the case for objective algorithms and models?

The research questions this leads us to are:
- Is there a way to measure the differences between synthetic and real speech that captures the distributional distance?
  - Can it predict human judgement, especially for modern systems which are close to (or have reached) human parity?
  - Can it be interpretable with respect to different aspects/factors of the speech?
  // - Can it show if synthetic speech has reached human parity, as subjective evaluation would suggest, or if it is still lacking, as #abbr.a[TTS]-for-#abbr.a[ASR] evaluation would make us believe?

To address these questions, we introduce the #abbr.l[TTSDS], which measures the distributional distances of various speech representations to arrive at individual distances for different #emph[factors] of speech as well as an overall distribution distance which correlates with human evaluation results across time and domains while also showing full human parity has only been reached in a limited set of domains and languages.


=== Published works

#link(<part_00>, [Part I]) contains background information necessary for the remainder of the thesis. A part of the following work is used, as it is one of the few #abbr.a[SSL] prosody representations available.

- #cite(<wallbridge_mpm_2025>, form: "full", style: "iso-690-numeric")

#link(<part_01>, [Part II]) contains #abbr.a[TTS]-for-#abbr.a[ASR] research, and is informed by the following publications:

- #cite(<minixhofer_evaluating_2023>, form: "full", style: "iso-690-numeric")

- #cite(<minixhofer_oversmoothing_2025>, form: "full", style: "iso-690-numeric")

#link(<part_01>, [Part III]) introduces #abbr.a[TTSDS] as a distributional measure for synthetic speech:

- #cite(<minixhofer_ttsds_2024>, form: "full", style: "iso-690-numeric")

- #cite(<minixhofer_ttsds2_2025>, form: "full", style: "iso-690-numeric")

=== Conclusions and open questions

In this work, we discuss and address the difficult problem of synthetic speech evaluation. First, we investigate how well the synthetic speech performs for the proxy task of #abbr.a[TTS]-for-#abbr.a[ASR], and find the gap is larger than expected from previously reported naturalness ratings by humans, and cannot, to the best of our knowledge, be fully explained by changes to the #abbr.a[TTS] systems. The fact listening test do not show this gap motivates us to approach this task from a distributional perspective: We introduce #abbr.a[TTSDS], a metric which approximates the actual distance between synthetic and real speech distributions. We accomplish this by utilising a number of representations related to different aspects or factors of the speech, effectively creating an ensemble of distribution distances. Our metric both outperforms all state-of-the-art metrics for #abbr.a[TTS] evaluation we compare it against and is robust across time, domains, systems and languages. However, some of our findings provide more questions than answers, especially when it comes to subjective evaluation. While traditional subjective evaluation still can detect differences between systems, they are becoming smaller and smaller. There is also the even bigger problem of only sample-wise comparisons being conducted for subjective evaluations, when we predict that soon diversity of synthetic speech, such as being able to generate many variations of the same utterance, will be valued more. While our objective evaluation methods addresses some of these issues, better subjective evaluation methods are necessary to inform us on the actual preferences and impressions of the listeners which will ultimately interact and be exposed to synthetic speech systems.