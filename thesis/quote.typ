#let q(attribution, title, content) = {
  set text(top-edge:"cap-height", bottom-edge: "baseline"); 
  quote(
    attribution: [
      #attribution #linebreak() #title
    ],
    ["#content"]
  )
}
