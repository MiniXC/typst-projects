#let q(attribution, title, content) = {
  set text(top-edge:"cap-height", bottom-edge: "baseline"); 
  quote(
    attribution: [
      #attribution #linebreak() #title
    ],
    ["#content"]
  )
}

#let citep(key) = {
  cite(key, form: "prose")
}

#let citea(key) = {
  cite(key, form: "author")
}