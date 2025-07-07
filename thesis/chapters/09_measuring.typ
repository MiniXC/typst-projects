#import "../abbr.typ"
#import "../quote.typ": * 
#import "@preview/drafting:0.2.2": inline-note

== Measuring distributional distance <09_dist>

#q(
  [#citep(<rubner_earth_2000>)], 
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval")],
  [… we want to define a consistent measure of distance, or dissimilarity, between two distributions of mass in a space that is itself endowed with a ground distance. … Practically, it is important that [such] distances between distributions correlate with human perception.]
)

As we have established throughout this work, it is useful to think of speech as a distribution. In this chapter, we formalize this further, and introduce a method to empirically measure how far real and synthetic speech distributions are apart across systems, domains and languages.

=== Audio & speech distributions

#figure(
  image(
    "../figures/9/xvector.svg",
    alt: "Three 3D surface plots showing kernel density estimates of X-Vector speaker embeddings projected into 2D PCA space. The first plot, labeled 'Ground Truth,' shows two distinct high-density peaks. The second plot, labeled 'Synthetic,' has a similar distribution with slightly smoother peaks. The third plot, labeled 'Noise,' shows a single narrow peak, which is approximately 5 times higher than the peaks in the other figures."
  ),
  caption: [#abbr.pls[KDE] of X-Vector speaker representations projected into a 2-dimensional #abbr.s[PCA] space, shown for (left to right) ground truth, synthetic, and noise data. The density is normalized and scaled by $times 10^(-5)$.],
  placement: none,
)

If we think of the set of all possible speech recordings with some specific constraints, the difficulty of matching the real speech distribution becomes clear. Even if we constrain utterances to never exceed 60 seconds, and allow each data point within an utterance to only be one of $2^16$ values (referred to as a bit depth of 16) and set the sampling rate to 16kHz, this results in $16,000*60=960,000$ values per recording. The resulting number of possible recordings is hard to fathom: $2^(16 times 960,000)$ -- however, to human listeners, the vast majority of these recordings would sound like meaningless noise.

// Of course, the number of possible sentences, even in English alone, is similarly large: If we assume vocabulary of 100,000 unique words, which cannot be formed by combining two or more words, if we assume a maximum number of $256$ words per sentence, we arrive at $2^(8*100,000)$ -- although again, just as with recordings, the majority of these combinations would not be perceived as well-formed sentences by most humans.

When creating a system capable of producing synthetic speech, we should aim to model the real speech distribution "hidden" within this impossibly large possible recording space -- however, if we knew said distribution, we would not need to model it in the first place. We therefore usually settle for estimating the distribution from data, and verify our models learned something approximating the real distribution by asking listeners to quantify their subjective perceptions as outlined in @08_eval[Chapter]. However, we can also quantify how closely the synthetic distribution resembles the real distribution as outlined in the remainder of this Chapter.

=== Earth movers distance

An inuitive way to measure the distance between two distributions is the #abbr.a[EMD], named and first introduced for the purpose of determining perceptual similarity for image retrieval by @rubner_earth_2000 and is derived from the Wasserstein distance @vaserstein_markov_1969 which in turn makes use of the Kantorovich–Rubinstein metric @kantorovich_planning_1939. Its motivation is explained as follows:

#q(
  [Yossi Rubner, Carlo Tomasi and Leonidas J. Guibas], 
  [#emph("The Earth Mover's Distance as a Metric for Image Retrieval"), 2000 @rubner_earth_2000],
  [Intuitively, given two distributions, one can be seen as a mass of earth properly spread in space, the other
as a collection of holes in that same space. Then, the #abbr.a[EMD] measures the least amount of work needed to fill the holes with earth. Here, a unit of work corresponds to transporting a unit of earth by a unit of ground
distance.]
)

This makes it a transport problem @hitchcock_transport_1941, which can be solved for the 2D case of image histograms as in the original work @rubner_earth_2000 -- however, speech representations (see @02_perceptual) are often high-dimensional, in which case this problem is prohibitively expensive.

=== Wasserstein metric

While the general #abbr.a[EMD] is computationally complex, a specific instance known as the *2-Wasserstein distance* offers tractable solutions in certain cases that are highly relevant for comparing distributions of speech representations.

Formally, the Wasserstein distance measures the distance between two empirical probability distributions, defined as the real distribution $P_R$ and the synthetic distribution $P_S$. It is defined as the minimum "cost" to transform one distribution into the other. The cost is calculated by multiplying the amount of "mass" moved by the distance it is moved. The set of all possible ways to move the mass is called the set of *transport plans*, $Pi(P_R, P_S)$. Each transport plan $gamma(x,y)$ is a joint distribution whose marginals are $P_R$ and $P_S$. Intuitively, $gamma(x,y)$ specifies how much mass to move from point $x$ to point $y$ to transform the distribution $P_R(x)$ into $P_S(y)$.

The $p$-Wasserstein distance is then the minimum cost over all possible transport plans:

$ W_p(P_R, P_S) = (inf_(gamma in Pi(P_R, P_S)) EE_((x,y)~gamma)[d(x,y)^p])^(1/p) $

where $d(x,y)$ is the distance between points $x$ and $y$. For our purposes, we will focus on the case where $p=2$ and $d(x,y)$ is the Euclidean distance $||x-y||_2$, rather than the #abbr.a[EMD] for which $p=1$. As with the general #abbr.a[EMD], computing this for arbitrary high-dimensional distributions remains difficult @kolouri_optimal_2017. However, two special cases provide efficient, closed-form solutions.

==== One-dimensional case

In the one-dimensional case, the 2-Wasserstein distance has a simple closed-form solution that avoids searching over all possible transport plans. It can be calculated directly from the inverse #abbr.a[CDF] of the two distributions. Given the #abbr.a[CDF] for the real and synthetic distributions, $C_R$ and $C_S$, the squared 2-Wasserstein distance is simply the squared L2-distance between their inverse functions @kolouri_optimal_2017:

$ W_2^2(P_R, P_S) = integral_0^1(C_R^(-1)(z)-C_S^(-1)(z))^2d z $

This property is the foundation of the *Sliced-Wasserstein distance*, which computes the average Wasserstein distance between distributions over many random one-dimensional projections. However, there is another, form for the high-dimensional case which does not rely on slices.

==== High-dimensional case with gaussian assumption

For sets of high-dimensional vectors, as is common for #abbr.a[DNN] features, computing the quantile functions is not feasible. However, as proposed by @heusel_fid_2017 in the context of image generation, we can make a simplifying assumption: that the embedding distributions can be approximated by multivariate Gaussians. This is a reasonable assumption for embeddings that have been projected into a well-behaved latent space @heusel_fid_2017. This approximation allows us to again compute the 2-Wasserstein distance in closed form using only the mean and covariance of the distributions.

Let the real and synthetic embedding distributions be modeled by multivariate Gaussians $N(mu_R, Sigma_R)$ and $N(mu_S, Sigma_S)$ respectively. The squared 2-Wasserstein distance, also known as the Fréchet distance @frechet_1925, between these two Gaussians is given by @dowson_frechet_1982:

$ W_2^2(P_R, P_S) = ||mu_R - mu_S||_2^2 + text("Tr")(Sigma_R + Sigma_S - 2(Sigma_R Sigma_S)^(1/2)) $

where:
- $mu_R$ and $mu_S$ are the mean vectors of the real and synthetic embeddings.
- $Sigma_R$ and $Sigma_S$ are the covariance matrices.
- $text("Tr")(dot)$ is the trace of a matrix.
- $(C_R C_S)^(1/2)$ is the matrix square root of the product of the covariance matrices.

The first term, $||mu_R - mu_S||_2^2$, measures the distance between the centers of the two distributions. The second term measures the difference in their spread and orientation (i.e., their covariances). In practice, we estimate the sample mean and covariance from a large number of real and synthetic embeddings, respectively, and then compute the distance using the formula above. This metric is the basis for the well-known *Fréchet Inception Distance* (#abbr.s[FID]) @heusel_fid_2017 in image generation, and the same principle can be applied to audio embeddings to create a *Fréchet Audio Distance* @kilgour_fad_2019.

// METHODOLOGY 

=== Perceptually-motivated factorized evaluation

As we discussed in @08_distances, there are various ways to objectively evaluate if synthetic speech matches its real counterparts, and many are perceptually motivated, in-line with the representations presented in @02_perceptual.

#inline-note()[
  The rest of this Section will be Methodology which is not covered in this version.
]