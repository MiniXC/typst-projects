#figure(
  diagram(
    spacing: 5pt,
    cell-size: (8mm, 10mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 70%,

    // legend
    node((-1,3.2), align(left)[$arrow.cw.half$#box(width: 1fr, repeat[.])#abbr.a[AR]], width: 50mm),
    node((-1,2.6), align(left)[$arrows.tt$#box(width: 1fr, repeat[.])#abbr.a[NAR]], width: 50mm),
    node((-1,1.5), align(left)[$("AR"#sym.arrow.l.r"NAR")$]),
    
    node((0,3.6), align(left)[$E$#box(width: 1fr, repeat[.])diffusion], width: 50mm),
    node((0,3.1), align(left)[$M$#box(width: 1fr, repeat[.])L1/L2], width: 50mm),
    node((0,2.6), align(left)[$H$#box(width: 1fr, repeat[.])cross-entropy], width: 50mm),
    node((0,2.1), align(left)[$O$#box(width: 1fr, repeat[.])combination/other], width: 50mm),
    node((0,1.5), align(left)[$["Objective"]$]),

    node((1,3.6), align(left)[$v$#box(width: 1fr, repeat[.])mel vocoder], width: 50mm),
    node((1,3.1), align(left)[$l$#box(width: 1fr, repeat[.])latent style], width: 50mm),
    node((1,2.6), align(left)[$i$#box(width: 1fr, repeat[.])#abbr.a[g2p]], width: 50mm),
    node((1,2.1), align(left)[$p$#box(width: 1fr, repeat[.])prosody], width: 50mm),
    node((1,1.5), align(left)[{$"Hierarchical"#sym.arrow.l.r"E2E"$}]),

    // timeline
    blob((-0.7,5), [
      #text(baseline: 1pt, top-edge: 0pt)[
      $(arrow.cw.half)[M]{v}$
      
      Tacotron (#citea(<wang_tacotron_2017>))

      Tacotron 2 (#citea(<shen_natural_2018>))

      $(arrow.cw.half)[H]{i,p,v}$
      
      Deep Voice (#citea(<arik_deepvoice_2017>))
      
      Deep Voice 2 (#citea(<gibiansky_deepvoice2_2017>))
    ]
    ], tint: orange, width: auto, name: <2017>),
    
    node(<2017.north>, align(top)[2017/2018], height: 50pt),
    edge(<2017.north>, <2019.north>, "|-|", shift: 5pt),

    blob((0.7,5), [
      #text(baseline: 0pt, top-edge: 0pt)[
      $(arrows.tt)[O]{v}$

      ParaNet (#citea(<peng_paranet_2020>))

      $(arrows.tt)[M]{i,p,v}$

      FastSpeech (#citea(<ren_fastspeech_2019>))

      
      FastSpeech 2 (#citea(<ren_fastspeech_2021>))

      $(arrows.tt)[O]{i,p,v}$

      Glow-TTS (#citea(<kim_glowtts_2020>))
    ]
    ], tint: orange, height: 40mm, name: <2019>),
    node(<2019.north>, align(top)[2019/2020], height: 50pt),

    edge(<2019.east>, "rr,d,lllllllllllllll,d", <2020.north>, "..", shift: 5pt),
    blob((-0.7,7), [
      #text(baseline: 1pt, top-edge: 0pt)[
      $(arrow.cw.half)[M]{v}$
      
      Tacotron (#citea(<wang_tacotron_2017>))

      $(arrow.cw.half)[H]{i,p,v}$
      
      DeepVoice (#citea(<arik_deepvoice_2017>))
    ]
    ], tint: orange, width: auto, name: <2020>),
    
    
  ),
  placement: none,
  caption: "Timeline of modern TTS systems and their attributes.",
) <fig_tts_timeline>