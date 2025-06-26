#import "../abbr.typ"
#import "../quote.typ": q 

== Measuring distributional distance

#q(
  [Yossi Rubner, Carlo Tomasi and Leonidas J. Guibas], 
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval"), 2000 @rubner_earth_2000],
  [… we want to define a consistent measure of distance, or dissimilarity, between two distributions of mass in a space that is itself endowed with a ground distance. … Practically, it is important that [such] distances between distributions correlate with human perception.]
)

#figure(
  image(
    "../figures/9/xvector.svg",
    alt: "Three 3D surface plots showing kernel density estimates of X-Vector speaker embeddings projected into 2D PCA space. The first plot, labeled 'Ground Truth,' shows two distinct high-density peaks. The second plot, labeled 'Synthetic,' has a similar distribution with slightly smoother peaks. The third plot, labeled 'Noise,' shows a single narrow peak, which is approximately 5 times higher than the peaks in the other figures."
  ),
  caption: [#abbr.pls[KDE] of X-Vector speaker representations projected into a 2-dimensional #abbr.s[PCA] space, shown for (left to right) ground truth, synthetic, and noise data. The density is normalized and scaled by $times 10^(-5)$.]
)

As we have established throughout this work, it is useful to think of speech as a distribution. In this chapter, we formalize this further, and introduce a method to empirically measure how far real and synthetic speech distributions are apart across systems, domains and languages.

=== Audio & speech distributions

We now define the set of all possible speech recordings with some specific constraints to illustrate the difficulty of matching the real speech distribution. Since we are concerned with synthesis of individual sentences we constrain utterances, for the sake of this example, to never exceed 60 seconds, with shorter utterances being padded with zeroes to reach this length. We allow each data point within an utterance to be one of $2^16$ values (referred to as a bit depth of 16) and set the sampling rate to a conservative 16kHz, which results in $16,000*60=960,000$ values per recording. The resulting number of possible recordings is hard to fathom: $2^(16 times 960,000)$ -- however, to human listeners, the vast majority of these recordings would sound like static noise.

// Of course, the number of possible sentences, even in English alone, is similarly large: If we assume vocabulary of 100,000 unique words, which cannot be formed by combining two or more words, if we assume a maximum number of $256$ words per sentence, we arrive at $2^(8*100,000)$ -- although again, just as with recordings, the majority of these combinations would not be perceived as well-formed sentences by most humans.

When creating a system capable of producing synthetic speech, we should aim to model the real speech distribution "hidden" within this impossibly large possible recording space -- however, if we knew said distribution, we would not need to model it in the first place. We therefore usually settle for estimating the distribution from data, and verify our models learned something approximating the real distribution by asking listeners to quantify their subjective perceptions.

=== Wasserstein distance

==== Fréchet inception distance

==== Distributional measures in speech

// METHODOLOGY 

=== Perceptually-motivated factorized evaluation

As we discussed in @07_distances, there are various ways to objectively evaluate if synthetic speech matches its real counterparts, and many are perceptually motivated, in-line with the representations presented in @06_perceptual.
