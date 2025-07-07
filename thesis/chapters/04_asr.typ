#import "../abbr.typ" 
#import "../quote.typ": *

=== Automatic speech recognition

Automatic Speech Recognition (ASR) systems perform the inverse task of TTS, mapping an acoustic signal to its corresponding lexical transcription. Modern high-performance ASR systems are almost exclusively trained using discriminative objectives. In contrast to generative approaches that might model the probability of an acoustic sequence given a text, discriminative models are optimized to directly model the posterior probability $p(bold(T)|bold(S))$, maximizing the score of the correct transcription while simultaneously minimizing the scores of all incorrect competing hypotheses. This approach has proven more effective at achieving low Word Error Rates (WER). The two dominant paradigms for discriminative ASR training are hybrid HMM-DNN systems and end-to-end models.

==== Hybrid HMM-DNN systems

Hybrid systems combine the strengths of Deep Neural Networks (DNNs) for acoustic modeling with the temporal modeling capabilities of Hidden Markov Models (HMMs). In this architecture, the DNN, often a Time-Delay Neural Network (TDNN) @peddinti_time_2015, processes acoustic feature vectors $bold(o)_t$ at each time step $t$ and outputs a posterior probability distribution over the set of HMM states $q$:

$ p(q_t | bold(o)_t; theta) $

While this acoustic model can be trained with a simple cross-entropy loss against frame-level alignments, system performance is substantially improved through sequence-level discriminative training. The state-of-the-art objective for this is #abbr.a[LF-MMI] @povey_purely_2016. The MMI criterion aims to maximize the mutual information between the observation sequence $bold(O)$ and the reference word sequence $W_text("ref")$. This is achieved by maximizing the log-likelihood of the correct transcription (numerator) while minimizing the log-likelihood of all possible transcriptions (denominator):

$ cal(L)_(text("MMI"))(theta) = log (p_theta(bold(O) | cal(M)_(W_text("ref"))) p(W_text("ref"))) / (sum_W p_theta(bold(O) | cal(M)_W) p(W)) $

Here, $cal(M)_W$ is the HMM corresponding to a word sequence $W$, and $p(W)$ is a language model probability. The numerator represents the likelihood of the correct transcription, while the denominator sums over the likelihoods of all possible transcriptions, explicitly creating a margin that pushes down the scores of incorrect hypotheses. The "Lattice-Free" component streamlines this process by using a simpler phone-level decoding graph, allowing for more efficient end-to-end discriminative training.

==== End-to-end models

#abbr.a[E2E] models represent a paradigm shift, collapsing the entire ASR pipeline into a single, deep neural network. Architectures such as the Conformer @gulati_conformer_2020 have become standard, learning a direct mapping from an acoustic sequence $bold(X)$ to a label sequence $bold(Y)$ (e.g., characters or sub-words). The key enabling technology for this is the #abbr.a[CTC] loss function @graves_ctc_2012.

CTC circumvents the need for pre-aligned data by introducing a special "blank" token (–) to the output vocabulary. It defines a many-to-one mapping function, $cal(B)$, that first collapses any repeated non-blank labels and then removes all blank tokens from a frame-level output path $bold(pi)$ to yield the final, shorter label sequence $bold(Y)$. The total conditional probability of the target sequence is the sum of probabilities of all alignment paths that map to it:

$ p(bold(Y)|bold(X)) = sum_(bold(pi) in cal(B)^(-1)(bold(Y))) p(bold(pi)|bold(X)) $

The probability of any single path $bold(pi)$ is calculated as the product of the per-timestep softmax outputs from the neural network:

$ p(bold(pi)|bold(X)) = product_(t=1)^T p(pi_t | bold(X)) $

The CTC loss is the negative log-likelihood of this total probability, $cal(L)_("CTC") = -log p(bold(Y)|bold(X))$, which is calculated efficiently using dynamic programming. By maximizing the summed probability of all valid paths, the CTC objective implicitly minimizes the probability of all other label sequences, making it an inherently discriminative training criterion.

==== Pretrain-finetune approach

In recent #abbr.a[ASR] works, #abbr.a[SSL] is frequently employed, where a model is first pre-trained on a task like predicting masked portions of the speech. The resulting model is then fine-tuned with an additional "head" for the desired task and objective, most commonly #abbr.a[CTC]. See @02_ssl for more details.

// #inline-note[This and the next chapter have background only, is that ok?]

// #inline-note[TODO: Add some notes on WER, CER to set up WERR in the next chapter?]