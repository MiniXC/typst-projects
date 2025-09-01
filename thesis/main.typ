// PACKAGES

#import "abbr.typ"
#import "@preview/wordometer:0.1.4": word-count, total-words, total-characters
#import "@preview/i-figured:0.2.4"

#show: word-count


#set quote(block: true)
#set cite(style: "annual-reviews-author-date")

#abbr.make(
  ("KDE", "Kernel Density Estimation", "Kernel Density Estimates"),
  ("PCA", "Principal component analysis"),
  ("MFCC", "Mel Frequency Cepstral Coefficient"),
  ("SSL", "Self-Supervised Learning"),
  ("NLP", "Natural Language Processing"),
  ("TTS", "Text-to-Speech"),
  ("CNN", "Convolutional Neural Network"),
  ("G2P", "Grapheme-to-Phoneme"),
  ("VAD", "Voice Activity Detection"),
  ("DNN", "Deep Neural Network"),
  ("SRMR", "Speech-to-reverberation modulation energy ratio"),
  ("PESQ", "Perceptual Evaluation of Speech Quality"),
  ("SNR", "Signal-to-noise ratio"),
  ("WADA", "Waveform Amplitude Distribution Analysis"),
  ("MOS", "Mean Opinion Score"),
  ("CMOS", "Comparison MOS"),
  ("SMOS", "Speaker Similarity MOS"),
  ("STOI", "Short‑Time Objective Intelligibility"),
  ("MCD", "Mel-Cepstral Distortion"),
  ("WER", "Word Error Rate"),
  ("WERR", "Word Error Rate Ratio"),
  ("CER", "Character Error Rate"),
  ("ASR", "Automatic Speech Recognition"),
  ("FID", "Fréchet Inception Distance"),
  ("FAD", "Fréchet Audio Distance"),
  ("EMD", "Earth Mover's Distance"),
  ("CDF", "Cumulative distribution function"),
  ("E2E", "End-to-End"),
  ("AR", "Autoregressive"),
  ("NAR", "Non-Autoregressive"),
  ("MSE", "Mean-Squared Error"),
  ("ELBO", "Evidence lower bound"),
  ("VAE", "Variational Autoencoder"),
  ("DDPM", "Denoising Diffusion Probabilistic Model"),
  ("HMM", "Hidden Markov Model"),
  ("CTC", "Connectionist Temporal Classification"),
  ("LF-MMI", "Lattice-Free Maximum Mutual Information"),
  ("GST", "Global Style Token"),
  ("RIR", "Room Impulse Response"),
  ("VC", "Voice Conversion"),
  ("LLM", "Large Language Model"),
  ("TTSDS", "Text-to-Speech Distribution Score"),
  ("FLAC", "Free Lossless Audio Codec"),
  ("MPM", "Masked Prosody Model"),
  ("CWT", "Continous Wavelet Transform"),
  ("STFT", "Short-time Fourier transform"),
  ("NLL", "Negative log-liklihood"),
  ("SLM", "Spoken Language Model"),
  ("GMM", "Gaussian Mixture Model"),
  ("JSD", "Jensen-Shannon Divergence"),
  ("MWERR", "Mean Word Error Rate Ratio"),
  ("TDNN", "Time-delay Neural Network"),
  ("EMA", "Exponentional Moving Average"),
  ("CLIP", "Contrastive Language-Image Pre-training")
)

// CONFIG
#let review = false

#let print = false

#set page(
    paper: "a4",
    margin: (
      top: 1.8cm,
      bottom: 1.8cm,
      inside: if review { 4.8cm } else {if print { 3.8cm } else { 3.4cm }},
      outside: if review { 4.8cm } else {if print { 2.8cm } else { 3.4cm }},
    ),
    binding: left,
  )
#set text(
  font: "New Computer Modern",
  size: 12pt,
  hyphenate: true,
)
#set par(
  justify: true,
  leading: 0.52em,
)
#let use-heading-lines = true
#let line = [
  #block(spacing: 0em, above: 1.5em, below: 1.5em)[
    #if use-heading-lines {
      [#line(length: 100%)]
    }
  ]
  
]
#let title-spacing() = 9em
#let author = "Christoph Minixhofer"
#set document(
  title: "Quantifying the Distributional Distance between Synthetic and Real Speech", 
  author: "Christoph Minixhofer",
  date: datetime.today()
)

// heading shenanigans
#let parts = counter("parts")
#parts.step()
#show heading.where(
  level: 1
): it => [
  #if "." in counter(heading).display() {
    pagebreak()
    set align(center+horizon)
    parts.step()
    block()
    block(above: 1.5em, below: 2em)[
      #line
      #set text(font: "Crimson Pro", size: 26pt, weight: "medium", hyphenate: false)
      #set par(justify: false)
      Part #parts.display("I")
      #h(.2em) -- #h(.2em)
      #(it.body)
      #line
    ]
  } else {
    pagebreak()
    block()
    set text(font: "Crimson Pro", size: 24pt, weight: "regular", hyphenate: false)
    set par(justify: false)
    set align(center)
    if counter(heading).display() == "0" {
      block(above: 1.5em, below: 2em)[
        #line
        #(it.body)
        #line
      ]
    } else {
      block(above: 1.5em, below: 2em)[
        #line
        Chapter #counter(heading).display() #h(.2em) -- #h(.2em) #(it.body)
        #line
      ]
    }
  }
]
#show heading.where(
  level: 3
): set text(font: "Crimson Pro", size: 18pt, weight: "semibold")
#show heading.where(
  level: 4
): set text(font: "Crimson Pro", size: 14pt, weight: "bold")

#let frontmatter-heading(content) = [
  #pagebreak()
  #set align(center)
  #block()
  #block(above: 1.5em, below: 1.5em)[
    #line
    #set text(font: "Crimson Pro", size: 24pt, weight: "regular", hyphenate: false, baseline: .45em)
    #set par(justify: false)
    #content.text
    #line
  ]
]



// TITLE PAGE
#align(center+horizon, 
  [
    #block(below: title-spacing(), width: 10em)[
      #image("logo/edinburgh_crest.png")
    ]
    #line
    #block(spacing: 0em, below: 1em)
    #text(24pt, font: "Crimson Pro")[
      *#context document.title*
    ]
    #block(spacing: 0em, above: 1em)
    #line
    #block(above: title-spacing(), below: title-spacing())[
      #text(18pt, style: "italic")[
        #context document.author.first()
      ]
    ]
    #if true {
      block(above: 0em, below: 2em)[
        #text(10pt, fill: blue)[
          Review version (#datetime.today().day()/#datetime.today().month()/#datetime.today().year())
  
          Word Count: $approx$#(context(calc.round(float(str(state("total-words").final()))/1000)*1000))
        ]
      ]
    }
    #text(14pt)[
        Doctor of Philosophy

        Centre for Speech Technology Research

        School of Informatics

        University of Edinburgh

        #context document.date.year()
    ]
  ]
)

#set page(numbering: "i", number-align: center)
#counter(page).update(1)

#set text(top-edge: if review {1em} else {"cap-height"}, bottom-edge: if review {-.8em} else {-.2em})

// Abstract
#frontmatter-heading([Abstract])

This thesis addresses the discrepancy between the high perceived naturalness of synthetic speech and its comparatively limited utility for training robust downstream applications, specifically Automatic Speech Recognition (ASR) systems. Despite recent Text-to-Speech (TTS) models achieving subjective naturalness ratings statistically indistinguishable from human speech, ASR models trained exclusively on synthetic data consistently exhibit significantly higher error rates when evaluated on real speech. We posit that this persistent #emph[synthetic-real gap] arises from the inability of current TTS models to fully approximate the nuanced, high-dimensional probability distribution of real speech, particularly concerning its inherent variability.

To quantify this disparity, we introduce the Word Error Rate Ratio (WERR), a metric that directly compares ASR performance when trained on synthetic versus real data. Our empirical investigations confirm a substantial WERR, indicating that ASR models trained on synthetic speech perform considerably worse than those trained on real speech, even when the synthetic utterances are subjectively perceived as highly natural. This observation suggests that synthetic speech, while perceptually clean, often lacks the intricate acoustic and prosodic variability crucial for ASR model robustness.

We explore methodologies to enhance synthetic speech diversity, including explicit conditioning on speaker, prosodic, and environmental attributes, as well as post-generation data augmentation. While these techniques demonstrably reduce the WERR, thereby narrowing the synthetic-real gap, our findings indicate a plateau in performance, suggesting an inherent ceiling for current synthesis paradigms in fully capturing real-world speech complexity. Furthermore, we conduct a comprehensive study on the scaling properties of synthetic data for ASR training, comparing Mean Squared Error (MSE)-based and Denoising Diffusion Probabilistic Model (DDPM)-based TTS architectures. Our results demonstrate that DDPMs exhibit superior scalability and more effectively leverage large training datasets, leading to sustained improvements in ASR performance compared to MSE models, which rapidly plateau due to oversmoothing. However, even with the enhanced scaling of DDPMs, projections indicate that an extraordinarily large volume of synthetic data would be required to achieve parity with ASR models trained on real speech.

To provide a robust and objective evaluation framework for synthetic speech that directly addresses these distributional nuances, we propose the Text-to-Speech Distribution Score (TTSDS). This metric quantifies the dissimilarity between real and synthetic speech distributions across perceptually motivated factors—including generic acoustic similarity, speaker realism, prosody, and intelligibility by leveraging the 2-Wasserstein distance. Through extensive validation against subjective listening test data across time (2008-2024) and diverse domains and languages, TTSDS demonstrates strong and consistent correlations with human judgments. This validation establishes TTSDS as a reliable objective measure capable of predicting human perception and providing interpretable insights into specific areas of improvement for TTS systems. This work shows that while synthetic speech has reached impressive levels of subjective naturalness, it cannot yet accurately replicate the full distributional complexity of human speech.

#frontmatter-heading([Lay Summary])

Technology that turns text into spoken words, known as Text-to-Speech (TTS), has advanced to a point where computer-generated voices can be hard to tell apart from human voices when heard one at a time. Automatic Speech Recognition (ASR) addresses the opposite task of TTS, turning speech into text, and ASR models need large amounts of data to be trained. The progress in TTS suggests that TTS-generated speech -- called synthetic speech -- could be used to train ASR models.

However, despite how natural the synthetic speech sounds, using it to train ASR models leads to a large drop in the systems' performance. This unexpected result indicates that there must be underlying differences between synthetic and real speech that affect how well machines can learn from them. This work investigates these differences.

We create a new method for comparing collections of synthetic and real speech, rather than just single voice clips. We analyze how various sound features -- such as the rhythm and melody of speech, the unique qualities of a speaker's voice, and the clarity of the words -- are spread out across these sets of recordings. This analysis reveals measurable differences between the synthetic and real speech collections.

We find that the TTS systems that sound most natural to people also show distributions that are more similar to those of real speech. This result remains true under various conditions, including speech with background noise and recordings of children's voices, proving that our method is robust.

In conclusion, while single computer-generated voices can sound human, their combined features as a group are still different from real speech, which limits their use for certain tasks. Our work offers a reliable way to measure this difference of the overall distributions. This provides a tool to help guide future improvements in TTS technology, with the goal of matching the full complexity of human speech.

// Acknowledgements
#frontmatter-heading([Acknowledgements])

I would like to thank my supervisors Ondrej Klejch and Peter Bell who were encouraging and patient, and I admire how they managed to guide me in my research. The fellow students and scholars I have met throughout the years, within and without of CSTR have contributed to many ideas and insights for my work. Gustav Eje Henter was an endless source of advice and enthusiasm. Nick Rossenbach's research on TTS-for-ASR and discussions on the topic were inspiring and helped me find my footing in the field. Maria Dolak, Zeyu Zhao and Sung-Lin Yeh were awesome fellow 5th-floor-researchers and friends. I also had a great time with Ariadna Sanchez, Shahar Elisha, Ed Storey and Peter Vietling at various speech conferences, as well as many great discussions with fellow members of CSTR and the wider speech research community, too many to list them all.

My partner Celina always lent an ear and having her by my side means the world to me. She was there for all the ups and downs on this journey, and I am looking forward to spending the rest of our lives together.
I am also lucky to have an incredibly supportive family. My parents always believed I could do this, even when I did not, and came for a much-needed visit to Edinburgh when times were tough. My brother Benjamin, beyond being my best friend, has been a great source of inspiration and advice, and I am excited to read his thesis when the time comes. My grandmother was also always just a call away despite the big physical distance.

My friends, beyond listening to me about the perils of academia, have helped me keep up my resolve in so many ways. My flatmates over the years, Angus, Rory and David, have been nothing but lovely and kind. The D&D sessions with Kat, David, Elo, Alex and Sven were always amazing, as were the runs with Sam, the hikes and climbs with Erik, gaming with the TreemTeam and the discussions about German hip-hop with Dom.

Finally, I would also like to thank Huawei for funding this work and for the feedback their team in Cambridge provided.

#pagebreak()

#set text(top-edge: "cap-height", bottom-edge: "baseline")

// Table of Contents
#let parts-outline = counter("parts-outline")
#let p-page = state("p", none)
#let c-page = state("c", none)
#show outline.entry: it => context {
  link(
    it.element.location(),
    {
      set text(font: "Crimson Pro", size: 16pt)
      if it.element.depth == 1 {
        parts-outline.step()
        set text(font: "Crimson Pro", size: 17.5pt, style: "italic")
        if not (it.body().text.contains("Appendix") or it.body().text.contains("Bibliography")) {
          it.indented([
            Part #context [#parts-outline.display("I")] #h(.3em) --
          ], [
            #it.body()
          ])
       } else {
         if it.body().text.contains("Bibliography") {
         }
         [
           #it.body() 
           #box(width: 1fr, repeat[.])
           #text(font: "Noto Sans", size: 12pt, style: "normal")[#it.page()]
           #linebreak()
           #linebreak()
         ]
       }
      } else {
        if it.element.depth == 2 {
          set text(font: "Crimson Pro", size: 15pt)
          c-page.update(it.page())
          context if c-page.get() != p-page.get() {
            it.indented(it.prefix(), [#it.body() #box(width: 1fr, repeat[.]) #text(font: "Noto Sans", size: 12pt)[#it.page()]])
          }
          context if c-page.get() == p-page.get() {
            it.indented(it.prefix(), [#it.body() #h(1fr)])
          }
          p-page.update(it.page())
        }
        if it.element.depth == 3 {
          set text(font: "Crimson Pro", size: 14pt)
          c-page.update(it.page())
          context if c-page.get() != p-page.get() {
            it.indented(it.prefix(), [#it.body() #box(width: 1fr, repeat[.]) #text(font: "Noto Sans", size: 12pt)[#it.page()]])
          }
          context if c-page.get() == p-page.get() {

            it.indented(it.prefix(), [#it.body() #box(width: 1fr, repeat[.]) #text(font: "Noto Sans", size: 12pt)[#it.page()]])
          }
          p-page.update(it.page())
        }
      }
    }
  )
}
#show outline.entry.where(
  level: 1
): set block(above: 1em)
#show outline: set heading(level: 1)
#outline()

#show heading.where(level: 2): set heading(numbering: "1", level: 1)
#show heading.where(level: 3): set heading(numbering: "1.1", level: 2)
#show heading.where(level: 4): set heading(numbering: "1.1", level: 3)

#set page(numbering: "1", number-align: center)
#context counter(page).update(1)

#set text(top-edge: if review {1em} else {"cap-height"}, bottom-edge: if review {-.8em} else {-.25em})

#show math.equation: i-figured.show-equation.with(numbering: "1.1")
#show heading: i-figured.reset-counters
#show figure: i-figured.show-figure

#include "chapters/01_introduction.typ"

= Speech Representations, Synthesis and Recognition <part_00>

#include "chapters/02_factors.typ"

#include "chapters/03_tts.typ"

#include "chapters/04_asr.typ"

= Synthetic speech for speech recognition <part_01>


In this part of our work, we explore how well-suited synthetic speech is for training speech recognition models.
If the distribution of real speech was perfectly modeled, we would assume similar performance when training with synthetic speech as when training with real speech. However, this is not the case, suggesting systematic differences between synthetic and real speech. We first quantify the extent of this gap in @05_ttsasr[Chapter], then reduce it through conditioning and augmentation in @06_attr[Chapter] and finally investigate how far it could be reduced using scaling in @07_scaling[Chapter].

#include "chapters/05_tts_asr.typ"

#include "chapters/06_attr_tts.typ"

#include "chapters/07_scaling.typ"

= Distributional distance of synthetic and real speech <part_02>

Since our work in TTS-for-ASR suggested a large discrepancy between expected and actual performance due to a mismatch in distributions, we explore in which ways TTS evaluation can be improved by considering these distributions rather than individual samples alone. We introduce the Text-to-Speech Distribution Distance (TTSDS) a measure utilising the distributional Wasserstein distance across several factors of speech, and show it correlates well with human judgement across domains and could feasibly be applied across languages.

#include "chapters/08_eval.typ"

#include "chapters/09_measuring.typ"

#include "chapters/10_conclusions.typ"

#show heading.where(
  level: 2
): it => [
  #pagebreak()
  #block()
  #set text(font: "Crimson Pro", size: 24pt, weight: "regular", hyphenate: false)
  #set par(justify: false)
  #if it.body.text.contains("Content") {
    [#(it.body)]
  } else {
    [
      #set align(center)
      #block(above: 1.5em, below: 2em)[
        #line
        Appendix #counter(heading).display() #h(.5em) -- #h(.5em) #(it.body)
        #line
      ]
    ]
  }
]

#context counter(heading).update(0)
#show heading.where(level: 2): set heading(numbering: "A", level: 1)
#show heading.where(
  level: 1
): it => [
  #pagebreak()
  #set align(center+horizon)
  #parts.step()
  #block()
  #block(above: 1.5em, below: 2em)[
    #line
    #set text(font: "Crimson Pro", size: 26pt, weight: "medium", hyphenate: false)
    #set par(justify: false)
    #(it.body)
    #line
  ]
]

#bibliography(
  (
    "references/benchmarks.bib",
    "references/metrics.bib",
    "references/misc.bib",
    "references/representations.bib",
    "references/tts.bib",
    "references/tts_for_asr.bib",
    "references/datasets.bib",
    "references/scaling.bib",
    "references/asr.bib",
    "references/own.bib"
  ),
  style: "annual-reviews-author-date",
  
)

#set text(top-edge: if review {1em} else {"cap-height"}, bottom-edge: if review {-.8em} else {"baseline"})

= Appendix

// == Additional Figures

== Abbreviations

#abbr.list(columns: 2)

== Open source contributions

As far as licenses and resources permitted, all code and datasets used in the making of this thesis have been published at the web locations below.

- *Phones* #sym.arrow #underline[#link("https://minixc.github.io/phones",[minixc.github.io/phones])]: A library for calculating distances between phones across languages.
- *TTSDS* #sym.arrow #underline[#link("https://ttsdsbenchmark.com",[ttsdsbenchmark.com])]: The TTSDS score libary and datasets introduced in @09_dist[Chapter].
- *MPM* #sym.arrow #underline[#link("https://github.com/MiniXC/masked_prosody_model",[github.com/MiniXC/masked_prosody_model])]: #abbr.a[SSL] prosody correlate model introduced in @02_factors[Chapter].
- *Speech Diffusion* #sym.arrow #underline[#link("https://github.com/MiniXC/speech-diffusion",[github.com/MiniXC/speech-diffusion])]: The diffusion architecture introduced in @07_scaling[Chapter].
- *LightningFastSpeech2* #sym.arrow #underline[#link("https://github.com/MiniXC/LightningFastSpeech2", [github.com/MiniXC/LightningFastSpeech2])]: A reimplemention of FastSpeech2 with additional prosodic correlates and conditioning, introduced in @06_attr[Chapter].
- #emph[Various datasets and pretrained models] #sym.arrow #underline[#link("https://huggingface.co/cdminix", [huggingface.co/cdminix])]: Includes forced-aligned versions of LibriTTS (@06_attr[Chapter]), Vocex (@02_factors[Chapter]), and detailed listening test results from TTSDS (@09_dist[Chapter]).


== GenAI

Generative AI (GenAI) was used in the making of this thesis in accordance with the University of Edinburgh's GenAI policy#footnote[#link(
  "https://information-services.ed.ac.uk/computing/comms-and-collab/elm/guidance-for-working-with-generative-ai",
  underline[https://information-services.ed.ac.uk/computing/comms-and-collab/elm/guidance-for-working-with-generative-ai],
)]. For most cases, the Edinburgh Language Model service was used.

How to use these tools in academia is controversial, and in the light of the contents of this thesis, we set out to use them in a way that does not reduce the diversity of the lexical distribution found in this work, as is so often the case with GenAI. We also publish prompt templates and commit history of this work at #link("https://github.com/minixc/typst_projects", underline[github.com/minixc/typst_projects])

The general process for using GenAI in this work was conducted as follows: 1) Outline the problem and decide if GenAI is an appropriate tool to help. #linebreak()2) Think of how to pose this problem to GenAI while minimising AI bias. #linebreak()3) Create a new chat with GenAI, pose the problem and look at the output. 4) Close the chat, think over the output, and then incorporate suggestions.

*Problems suitable for GenAI*: We generally deem the following problems suitable for GenAI use:
- Repetitive text formatting/processing -- for example, switching from full titles to standard conference abbreviations in the bibliography.
- Grammatical error checking -- for example, checking if a specific paragraph has any grammatical mistakes.
- Structural improvements -- for example, interrogating if a different structure could be more intuitive for a specific section.
- Summarising/searching long-form content -- for example, asking if certain viewpoints are featured in a lengthy paper or book.

*Minimising Bias*: We use the following rules/techniques to minimise the introduction of GenAI-induced bias.
- Never copy prose: No prose is every copied, nor is the GenAI output displayed to the author while writing.
- Critique, not revision: We never ask to create a "better version" of any given input, but instead ask for specific critique of potential shortcomings. We had to sometimes explicitly specify this to the GenAI tools since they seemed prone to replacing the input with their own version, even when we did not ask for it.
- Be specific: We never asked open-ended questions to GenAI, such as "My thesis topic is [...] how would you structure this thesis?" - we always posed a narrow, specific problem with text we had already created.

// *Transparency*: We publish all GenAI prompt templates used in the making of this work at #link("https://github.com/minixc/thesis_genai", underline[github.com/minixc/thesis_genai]) -- a full version history of the progress of this thesis is also available at #link("https://github.com/minixc/typst_projects", underline[github.com/minixc/typst_projects]).

== Chronology

Research is not a linear process, and goals and topics can shift over time. For this work, the original goal was to investigate synthetic speech for #abbr.a[ASR] training (TTS-for-ASR), with a focus on low-resource applications.

However, the large gap between synthetic and real speech when it came to this task, compared to how highly speech produced by modern systems is rated by humans, gave us pause. This at first lead to more detailed investigations of the underlying data distributions while still focusing on improving TTS-for-ASR, but later inspired investigating evaluating TTS itself in a way that was more true to these distributions rather than individual samples.

Chapters 5 and 6 fall firmly into our earlier TTS-for-ASR work but the distributional distances investigated to improve TTS conditioning were the inspiration for TTSDS1 and TTSDS2 introduced in Chapters 8 and 9. Chapter 7 was work started before TTSDS, but finalized after TTSDS1 was already completed. While it does not directly link the two approaches, it provides the valuable insight that the limitations of the synthetic speech distribution cannot be fully outscaled (for now).

== Party Summary

This is a Lay Summary in informal language, which I used to inform family and friends about my work.

If you are reading this, you have probably heard Text-to-Speech voices before. It could be the one telling you to stay on the line when calling your GP, the voice giving directions in your car, or the voiceover in TikToks. The models making those voices need a lot of data and computing power -- and recently they have gotten enough of both to become really good. We now think that soon they will be impossible to tell apart from real voices.
You have probably also heard of speech recognition, which is how your voice gets converted to text when speaking to Siri or dictating something to your phone, or how YouTube subtitles get generated. This also needs a lot of data.
Since the Text-to-Speech voices are getting so good, we thought we could use them for helping with speech recognition, since these voices could give us good data to learn from.
But unfortunately, while they sound very good, the voices were really bad for speech recognition. This made us think that maybe there is some differences between these voices and real speech that we can't detect by just listening to them. So instead of looking at single recordings, we took a bunch of Text-to-Speech recordings and a bunch of real recordings and compared how they were spread out in many different areas that researchers have looked into for a long time. These areas measure how we speak (for example, how loud or quiet), who is speaking and how easily we can understand the words. We found out that when we look at a bunch of recordings this way, we can see differences in how they are spread out in these different areas -- and real speech is actually still quite different to synthetic speech. And the Text-to-Speech systems that sound good when listening to them were spread out more similarly to the real speech than the ones that didn't sound so good -- and this still works if the speech has background noise, or if it's children speaking instead of adults. We also looked into multiple languages, and figured out how to apply this to 14 of them without big changes. This means our work is robust and can hopefully be extended to new settings quite easily. To summarise, our main point is that while Text-to-Speech is really good at this point, when we look at a bunch of recordings instead of single ones, we can see its not quite the same as real speech, and that explains why we can't yet use it in the same way. Our method can be used to measure how close new Text-to-Speech systems get to the real speech, to make sure they keep advancing.