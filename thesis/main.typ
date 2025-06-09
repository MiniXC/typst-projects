

// CONFIG
#let print = false
#set page(
  paper: "a4",
  margin: (
    top: 1.5cm,
    bottom: 1.5cm,
    inside: if print { 3.5cm } else { 3cm },
    outside: if print { 2.5cm } else { 3cm },
  ),
  binding: left,
)
#set text(
  font: "Source Sans Pro",
  size: 12pt,
  hyphenate: false,
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
    #set text(font: "Crimson Pro", size: 26pt, weight: "medium")
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
  #set text(font: "Crimson Pro", size: 24pt, weight: "regular")
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
    #set text(font: "Crimson Pro", size: 24pt, weight: "regular")
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

// Abstract
#frontmatter-heading([Abstract])
#lorem(200)


// Lay Summary
#frontmatter-heading([Lay Summary])
#lorem(200)

// Acknowledgements
#frontmatter-heading([Acknowledgements])
#lorem(200)

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
        it.indented([
          Part #context [#parts-outline.display("I")] #h(.3em) --
        ], [
          #it.body()
        ])
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


== Introduction

=== Synthetic speech
=== Distribution gap
=== Notation
=== Contributions

= Text-to-speech fundamentals

== Modeling and training approaches

=== Architectural choices

=== Training data considerations

=== Injecting stochasticity

== Evaluation

=== Listening and preference tests

=== Objective metrics

= Synthetic speech for speech recognition

== TTS-for-ASR task

=== Data augmentation with synthetic speech

=== Exclusively synthetic training 

=== Implications on speech distributions

== Attribute-driven speech synthesis

=== Prosodic correlates

=== Utterance conditioning

== Scaling properties for TTS-for-ASR

=== Dataset size impact on performance

=== Stochasticity and scaling

= Quantifying distances of synthetic and real speech

== Factors and representations of speech

=== Self-supervised learning representations

=== Prosody representations

=== Environmental effects

=== Speaker identity

== Distance measures for TTS

=== Algorithmic

=== Model-based

=== Distributional

== Measuring distributional distance

=== Wasserstein distance



// = Text-to-Speech for Automatic Speech Recognition

// == Text-to-Speech Fundamentals

// === Modeling and Training Approaches

// ==== Stochasticity in Speech Synthesis

// ==== Architectural Choices

// == Automatic-Speech-Recognition Fundamentals

// === Hybrid and Deep Architectures

// = Text-to-Speech Distribution Distance 