#import "../abbr.typ"
#import "../quote.typ": q

== Synthetic speech evaluation

#q(
  [Sebastian Möller, Tiago H. Falk],
  [#emph("Quality Prediction for Synthesized Speech"), 2009 @moller_quality_2009],
  [Each time a new TTS system is developed which potentially introduces new types of degradations, the validity and reliability of such a prediction algorithm has to be tested anew.]
)

In this chapter, we outline evaluation methodology for synthetic speech, both #emph[subjective] (determined by raters opinions; potential different outcomes every time) and #emph[objective] (determined by a fixed algorithm, formula or model; same outcome every time with respect to data and parameters).

=== Subjective listening and preference tests

Here we discuss the most common subjective listening test methodologies and best practices. Subjective tests are the gold standard for synthetic speech evaluation, however, there are drawbacks and trade-offs to any listening test since human behaviour can never be fully anticipated, especially across differing groups of listeners, spans of time, and sets of #abbr.a[TTS] systems.

==== Best practices & drawbacks

Before detailing the individual methods, we will outline best practice according to the literature, as well as commonly misunderstood aspects of listening tests, including their drawbacks.

A general *best practice* is clearly answer the following questions before conducting a test:
#enum(
[
  Who are the *listeners*? Usually, native speakers of the language of the  samples are preferred. Additionally, certain tests require a significant number of different speakers to be used @wester_listeners_2015.
],
[
  What is the *setting*? E.g. #sym.arrow In a lab or online? If possible, requiring use of headphones in a quiet environment is preferred.
],
[
  What are the *instructions*? Recently, #emph[naturalness] is the most commonly evaluated attribute of synthetic speech, and this should be phrased in an understandable way in the instructions.
],
[
  What is the *data*? E.g. #sym.arrow Is the reference or an anchor @schoeffler_mushra_2015 included in the set of samples to be evaluated?
],
[
  What are the concrete *stimuli*? E.g. #sym.arrow How many samples are presented at one time, can listeners re-listen to the same sample? How many points are on the evaluation scale, and are they labeled?
]
)

All these questions should be answered #emph[before] conducting a listening test, and reported in the work to frame the results.

There are also drawbacks to subjective evaluation. We will not go into detail on the advantages and disadvantages of every methodology here, but there are two main drawbacks to consider:

- *Lack of standardisation*: There is no standardised framework for most listening test methodologies, beyond the labels and values for particular scales. This means the questions above are answered differently (and often not reported) making it difficult or even impossible to compare results between studies. Subjective listening test results should only ever be compared to results obtained within the same study with the same participants and setup.
- *Scale/comparison trade-off*: Many methodologies operate by presenting an ordinal scale to listeners on which they rate recordings on, which are then averaged, however this is not necessarily statistically meaningful. But when instead using a comparison-based task (in which one sample has to be rated over another), many more comparisons are needed @wells_bws_2024.



==== #abbr.l[MOS]





=== Objective metrics <07_distances>

==== Algorithmic

==== Model-based

==== Distributional