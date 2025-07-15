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

#let comic((width, height), text, color) = {
  figure(
    diagram(
      spacing: 5pt,
      cell-size: (8mm, 10mm),
      edge-stroke: 1pt,
      edge-corner-radius: 5pt,
  
      blob((0,0), text, tint: color, width: width, height: height),
      node((0,1), [#emph([This is a draft figure which will be replaced.])])
    ),
    placement: top,
    caption: text,
  );
}