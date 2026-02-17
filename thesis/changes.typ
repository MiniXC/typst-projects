#import "@preview/frame-it:2.0.0": *

#let changelog = true

#let (_minorchange, _majorchange) = frames(
  _minorchange: ("Minor Change", teal),
  _majorchange: ("Major Change", green),
)

#let minorchange(title, optional, text) = context {
  if changelog {
    _minorchange[#title][#optional][
      #text
    ]
  } else {
    text
  }
}

#let majorchange(title, optional, text) = context {
  if changelog {
    _majorchange[#title][#optional][
      #text
    ]
  } else {
    text
  }
}