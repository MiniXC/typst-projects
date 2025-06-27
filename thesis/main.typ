// PACKAGES

#import "abbr.typ"
#import "@preview/wordometer:0.1.4": word-count, total-words, total-characters

#show: word-count

#set quote(block: true)
#set cite(style: "springer-mathphys")


#abbr.make(
  ("KDE", "Kernel Density Estimation", "Kernel Density Estimates"),
  ("PCA", "Principal component analysis"),
  ("MFCC", "Mel Frequency Cepstral Coefficient"),
  ("SSL", "Self-Supervised Learning"),
  ("NLP", "Natural Language Processing"),
  ("TTS", "Text-to-Speech"),
  ("CNN", "Convolutional Neural Network"),
  ("g2p", "Grapheme-to-Phoneme"),
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
  ("WERR", "Relative Word Error Rate Reduction"),
  ("CER", "Character Error Rate"),
  ("ASR", "Automatic Speech Recognition"),
  ("FID", "Fréchet Inception Distance"),
  ("FAD", "Fréchet Audio Distance"),
  ("EMD", "Earth Mover's Distance"),
  ("CDF", "Cumulative distribution function")
)

// CONFIG
#let review = false

#let print = false

#set page(
    paper: "a4",
    margin: (
      top: 1.5cm,
      bottom: 1.5cm,
      inside: if review { 4.5cm } else {if print { 3.5cm } else { 3cm }},
      outside: if review { 4.5cm } else {if print { 2.5cm } else { 3cm }},
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
  #block(spacing: 0em, above: .8em, below: 1.2em)[
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
    #if review {
      block(above: 0em, below: 2em)[
        #text(10pt, fill: blue)[
          Review version with wide margins and spacing
  
          Word Count: #total-words
          
          Character Count #total-characters
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

#set text(top-edge: if review {1em} else {"cap-height"}, bottom-edge: if review {-.8em} else {"baseline"})

// Abstract
#frontmatter-heading([Abstract])
#lorem(200)


// Lay Summary
#frontmatter-heading([Lay Summary])
#lorem(200)

// Acknowledgements
#frontmatter-heading([Acknowledgements])
#lorem(200)

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
        set text(font: "Crimson Pro", size: 18pt, style: "italic")
        if not (it.body().text.contains("Appendix") or it.body().text.contains("Bibliography")) {
          it.indented([
            Part #context [#parts-outline.display("I")] #h(.3em) --
          ], [
            #it.body()
          ])
       } else {
         [
           #it.body()
           
         ]
       }
      } else {
        c-page.update(it.page())
        context if c-page.get() != p-page.get() {
          it.indented(it.prefix(), [#it.body() #h(2em) #it.page()])
        }
        context if c-page.get() == p-page.get() {
          it.indented(it.prefix(), [#it.body() #h(2em)])
        }
        p-page.update(it.page())
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

#set text(top-edge: if review {1em} else {"cap-height"}, bottom-edge: if review {-.8em} else {"baseline"})

#include "chapters/01_introduction.typ"

= Synthetic speech for speech recognition

In this first part of our work, we explore how well-suited synthetic speech is for training speech recognition models.
If the distribution of real speech was perfectly modeled, we would assume similar performance when training with synthetic speech as when training with real speech. However, this is not the case, suggesting systematic differences between synthetic and real speech, which we explore in the following chapters.

#include "chapters/02_modeling.typ"

#include "chapters/03_tts_asr.typ"

#include "chapters/04_attr_tts.typ"

#include "chapters/05_scaling.typ"

= Quantifying distances of synthetic and real speech

Since our work in TTS-for-ASR suggested a large discrepancy between expected and actual performance due to a mismatch in distributions, we explore in which ways TTS evaluation can be improved by considering these distributions rather than individual samples alone.

#include "chapters/06_factors.typ"

#include "chapters/07_eval.typ"

#include "chapters/08_measuring.typ"

#include "chapters/09_limitations.typ"


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
  )
)

= Appendix

== Additional Figures

== Abbreviations

#abbr.list(columns: 1)

== Open source contributions

As far as licenses and resources permitted, all code and datasets used in the making of this thesis have been published at the web locations below.

- *Phones* #sym.arrow #underline[#link("https://minixc.github.io/phones",[minixc.github.io/phones])]: A library for calculating distances between phones across languages.
- *TTSDS* #sym.arrow #underline[#link("https://ttsdsbenchmark.com",[ttsdsbenchmark.com])]: The TTSDS score libary and datasets introduced in @08_dist[Chapter].
- *MPM* #sym.arrow #underline[#link("https://github.com/MiniXC/masked_prosody_model",[github.com/MiniXC/masked_prosody_model])]: #abbr.a[SSL] prosody correlate model introduced in @06_factors[Chapter].
- *Speech Diffusion* #sym.arrow #underline[#link("https://github.com/MiniXC/speech-diffusion",[github.com/MiniXC/speech-diffusion])]: The diffusion architecture introduced in @05_scaling[Chapter].
- *LightningFastSpeech2* #sym.arrow #underline[#link("https://github.com/MiniXC/LightningFastSpeech2", [github.com/MiniXC/LightningFastSpeech2])]: A reimplemention of FastSpeech2 with additional prosodic correlates and conditioning, introduced in @04_attr[Chapter].
- #emph[Various datasets and pretrained models] #sym.arrow #underline[#link("https://huggingface.co/cdminix", [huggingface.co/cdminix])]: Includes forced-aligned versions of LibriTTS (@04_attr[Chapter]), Vocex (@06_factors[Chapter]), and detailed listening test results from TTSDS (@08_dist[Chapter]).


== On the use of GenAI

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

*Transparency*: We publish all GenAI chats used in the making of this work at #link("https://github.com/minixc/thesis_genai", underline[github.com/minixc/thesis_genai]).