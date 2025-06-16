// PACKAGES

#import "abbr.typ"
#import "@preview/wordometer:0.1.4": word-count, total-words, total-characters

#show: word-count

#abbr.make(
  ("KDE", "Kernel Density Estimation", "Kernel Density Estimates"),
  ("PCA", "Principal component analysis"),
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

#include "chapters/06_eval.typ"

#include "chapters/07_factors.typ"

#include "chapters/08_distance.typ"

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

#bibliography("refs.bib")

= Appendix

== Additional Figures

== Abbreviations

#abbr.list(columns: 2)

== Open source contributions

== On the use of GenAI