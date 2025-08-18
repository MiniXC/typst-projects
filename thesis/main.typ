// PACKAGES

#import "abbr.typ"
#import "@preview/wordometer:0.1.4": word-count, total-words, total-characters
#import "@preview/equate:0.3.2": equate

#show: equate.with(breakable: false, sub-numbering: true)
#set math.equation(numbering: "(1.1)")

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
  title: "Quantifying the distributional distance between synthetic and real speech", 
  author: "Christoph Minixhofer",
  date: datetime.today()
)

// heading shenanigans
#let parts = counter("parts")
#parts.step()
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
    Part #parts.display("I")

    #(it.body)
    #line
  ]
]

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
        Chapter #counter(heading).display() #h(.5em) -- #h(.5em) #(it.body)
        #line
      ]
    ]
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
  #block(above: 1.5em, below: 2em)[
    #line
    #set text(font: "Crimson Pro", size: 24pt, weight: "regular", hyphenate: false)
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
  
          Word Count: \~#(context(calc.round(float(str(state("total-words").final()))/1000)*1000))
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

// conversational + full duplex

#abbr.a[TTS] data generation has advanced to the point where it often mimics real data convincingly, according to human judges. Does that mean that in these settings, TTS is solved? If this were the case, the resulting data could be used as a stand-in from any problem were real data is needed -- especially in the inverse of #abbr.a[TTS], #abbr.a[ASR]. In this work, we attempt to use purely synthetic, #abbr.a[TTS]-generated, data for #abbr.a[ASR] and find a large discrepancy between the expected performance based on human judgements and the actual suitability of the data. While we show this #emph[synthetic-real gap] can be reduced by conditioning on a broad set of utterance-level characteristics, it cannot be explained fully. Further we find that dataset scale can equally only explain part of the gap, leading to diminishing returns beyond a certain point.
In light of these findings, we continue to investigate the problem from the other directions -- can we evaluate #abbr.a[TTS] in a way that shows this gap, while also offering explainabilty as to which areas lack the most. To this end, we introduce a benchmark which measures the divergence between real and synthetic data on a distribution level across several factors, including prosody, speaker identity, ambient acoustics and intelligibility. By normalising distribution distances across these factors to follow the same scale, we can show their average correlates with human judgement, while simultaneously showing full saturation has not been reached, unlike said human judgements. As we extend our benchmark to 14 languages, we find this is especially the case for languages other than English and Chinese. Our work shows that work remains to be done in #abbr.a[TTS] research, and that evaluation techniques should be adapted and improved as systems reach human parity by traditional measures.

// Lay Summary
#frontmatter-heading([Lay Summary])

If you are reading this, you have probably heard Text-to-Speech voices before. It could be the one telling you to stay on the line when calling your GP, the voice giving directions in your car, or the voiceover in TikToks. The models making those voices need a lot of data and computing power -- and recently they have gotten enough of both to become really good. We now think that soon they will be impossible to tell apart from real voices.
You have probably also heard of speech recognition, which is how your voice gets converted to text when speaking to Siri or dictating something to your phone, or how YouTube subtitles get generated. This also needs a lot of data.
Since the Text-to-Speech voices are getting so good, we thought we could use them for helping with speech recognition, since these voices could give us good data to learn from.
But unfortunately, while they sound very good, the voices were really bad for speech recognition. This made us think that maybe there is some differences between these voices and real speech that we can't detect by just listening to them. So instead of looking at single recordings, we took a bunch of Text-to-Speech recordings and a bunch of real recordings and compared how they were spread out in many different areas that researchers have looked into for a long time. These areas measure how we speak (for example, how loud or quiet), who is speaking and how easily we can understand the words. We found out that when we look at a bunch of recordings this way, we can see differences in how they are spread out in these different areas -- and real speech is actually still quite different to synthetic speech. And the Text-to-Speech systems that sound good when listening to them were spread out more similarly to the real speech than the ones that didn't sound so good -- and this still works if the speech has background noise, or if it's children speaking instead of adults. We also looked into multiple languages, and figured out how to apply this to 14 of them without big changes. This means our work is robust and can hopefully be extended to new settings quite easily. To summarise, our main point is that while Text-to-Speech is really good at this point, when we look at a bunch of recordings instead of single ones, we can see its not quite the same as real speech, and that explains why we can't yet use it in the same way. Our method can be used to measure how close new Text-to-Speech systems get to the real speech, to make sure they keep advancing.

// Acknowledgements
#frontmatter-heading([Acknowledgements])

First and foremost, my supervisors Ondrej Klejch and Peter Bell were encouraging, patient and full of wisdom, and I admire how they managed to guide me in my research -- it was the perfect balance, keeping me on track while also allowing a great a deal of flexibility. In particular, I will never forget a meeting in which I anxiously pointed out the research was deviating quite strongly from the initial premise and Peter, after a brief pause, simply suggested to change the title -- and so we did. I would also like to thank Huawei for funding this work and for their team to provide valuable feedback and encouragement.

The fellow students and scholars I have met throughout the years, within and without of CSTR, were the best reasons to come to the office and conferences and have supplied many ideas and insights for my work. Gustav Eje Henter believed in my research from the start, and has and continues to be an endless source of advice and enthusiasm. Nick Rossenbach's research on TTS-for-ASR and discussions on the topic were inspiring and helped me find my footing in the early stages. While our research does not have much in common, Maria Dolak, Zeyu Zhao and Sung-Lin Yeh were great fellow 5th-floor-IF-researchers. Finally, the great discussions with fellow CSTR members, too many to list them all, have undoubtedly contributed a great deal to this work as well.

I am also lucky to have an incredible family. My partner Celina always lent an ear and having her by my side means the world to me. My parents always believed I could do this, even when I did not, and came for a much-needed visit to Edinburgh when times were tough. My brother Benjamin, beyond being my best friend, has been a great source of inspiration and advice, and I am excited to read his thesis when the time comes -- his star in academia will undoubtedly shine brighter than mine. My grandmother was also always just a call away despite the big geographical distance.

To my friends, beyond listening to me about the perils of academia, you have helped me keep up my resolve in so many ways. My flatmates over the years, Angus, Rory and David, have been nothing but lovely and kind. The D&D sessions with Kat, David, Elo, Alex, Anja and Sven were always a highlight, as were the runs with Sam, the hikes and climbs with Erik, gaming with the Treemteam and the discussions about German hip-hop with Dom. On that note, I would also like to thank SSIO for waiting to release his fifth album until after the submission of this thesis. 

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
           // pagebreak()
         }
         [
           #it.body() 
           #box(width: 1fr, repeat[.])
           #text(font: "Noto Sans", size: 12pt)[#it.page()]
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
            // it.indented(it.prefix(), [#it.body() #h(1fr)])
            it.indented(it.prefix(), [#it.body() #box(width: 1fr, repeat[.]) #text(font: "Noto Sans", size: 12pt)[#it.page()]])
          }
          p-page.update(it.page())
        }
        // if it.element.depth == 4 {
        //   set text(font: "Crimson Pro", size: 13pt)
        //   c-page.update(it.page())
        //   context if c-page.get() != p-page.get() {
        //     it.indented(it.prefix(), [#it.body() #h(2em) #it.page()])
        //   }
        //   context if c-page.get() == p-page.get() {
        //     it.indented(it.prefix(), [#it.body() #h(2em)])
        //   }
        //   p-page.update(it.page())
        // }
      }
    }
  )
}
#show outline.entry.where(
  level: 1
): set block(above: 1em)
#show outline: set heading(level: 2)
#outline()

#show heading.where(level: 2): set heading(numbering: "1", level: 1)
#show heading.where(level: 3): set heading(numbering: "1.1", level: 2)
#show heading.where(level: 4): set heading(numbering: "1.1", level: 3)

#set page(numbering: "1", number-align: center)
#context counter(page).update(1)

#set text(top-edge: if review {1em} else {"cap-height"}, bottom-edge: if review {-.8em} else {-.25em})

#include "chapters/01_introduction.typ"

= Speech; Synthesis and Recognition <part_00>



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

How to use these tools in academia is controversial, and in the light of the contents of this thesis, we set out to use them in a way that does not reduce the diversity of the lexical distribution found in this work, as is so often the case with GenAI.

The general process for using GenAI in this work was conducted as follows:

1. Outline the problem and decide if GenAI is an appropriate tool to help.

2. Think of how to pose this problem to GenAI while minimising AI bias.

3. Create a new chat with GenAI, pose the problem and look at the output.

4. Close the chat, think over the output, and then incorporate suggestions.

*Problems suitable for GenAI*: We generally deem the following problems suitable for GenAI use:
- Repetitive text formatting/processing -- for example, switching from full titles to standard conference abbreviations in the bibliography.
- Grammatical error checking -- for example, checking if a specific paragraph has any grammatical mistakes.
- Structural improvements -- for example, interrogating if a different structure could be more intuitive for a specific section.
- Summarising/searching long-form content -- for example, asking if certain viewpoints are featured in a lengthy paper or book.

*Minimising Bias*: We use the following rules/techniques to minimise the introduction of GenAI-induced bias.
- Never copy prose: No prose is every copied, nor is the GenAI output displayed to the author while writing.
- Critique, not revision: We never ask to create a "better version" of any given input, but instead ask for specific critique of potential shortcomings. We had to sometimes explicitly specify this to the GenAI tools since they seemed prone to replacing the input with their own version, even when we did not ask for it.
- Be specific: We never asked open-ended questions to GenAI, such as "My thesis topic is [...] how would you structure this thesis?" - we always posed a narrow, specific problem with text we had already created.

*Transparency*: We publish all GenAI prompt templates used in the making of this work at #link("https://github.com/minixc/thesis_genai", underline[github.com/minixc/thesis_genai]) -- a full version history of the progress of this thesis is also available at #link("https://github.com/minixc/typst_projects", underline[github.com/minixc/typst_projects]).

== Chronology

Research is not a linear process, and goals and topics can shift over time. For this work, the original goal was to investigate synthetic speech for #abbr.a[ASR] training. However, the large gap between synthetic and real speech when it came to this task, compared to how highly speech produced by modern systems is rated by humans, gave me pause. Instead of solely focusing on its usefulness for ASR, I decided to investigate the synthetic speech distribution as a whole, seeing the TTS-for-ASR task as an entry point for doing so.