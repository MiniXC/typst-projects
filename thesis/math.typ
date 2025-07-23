// #let Syn = $attach(S, t: tilde, tr:"")$
// #let syn = $attach(s, t: tilde, tr:"")$
// #let stheta = $attach(theta, t: tilde, tr:"")$
// #let swer = $attach("WER", t: tilde, tr:"")$
// #let makesyn(inner) = {
//   $attach(#inner, t: tilde, tr:"")$
// }

#let Syn = $tilde(S)$
#let syn = $tilde(S)$
#let stheta = $tilde(theta)$
#let swer = $tilde("WER")$
#let makesyn(inner) = {
  $tilde(#inner)$
}