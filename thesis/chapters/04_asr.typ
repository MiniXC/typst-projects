#import "../abbr.typ"
#import "../quote.typ": *

== Automatic speech recognition <04_asr>

Automatic Speech Recognition (#abbr.a[ASR]) systems perform the inverse task of #abbr.a[TTS], mapping an acoustic signal to its corresponding lexical transcription. Modern high-performance #abbr.a[ASR] systems are almost exclusively trained using discriminative objectives. In contrast to generative approaches that might model the probability of an acoustic sequence given a text, discriminative models are optimized to directly model the posterior probability $p(T|S)$, maximizing the score of the correct transcription while simultaneously minimizing the scores of all incorrect competing hypotheses. This approach has proven more effective at achieving low Word Error Rates (#abbr.a[WER]). The two dominant paradigms for discriminative #abbr.a[ASR] training are hybrid HMM-DNN systems and end-to-end models.

=== Hybrid HMM-DNN systems

Hybrid systems combine the strengths of Deep Neural Networks (DNNs) for acoustic modeling with the temporal modeling capabilities of Hidden Markov Models (HMMs). In this architecture, the DNN, often a Time-Delay Neural Network (TDNN) @peddinti_time_2015, processes acoustic feature vectors $s_t$ at each time step $t$ and outputs a posterior probability distribution over the set of HMM states $q$:

$ p(q_t | s_t; theta) $

While this acoustic model can be trained with a simple cross-entropy loss against frame-level alignments, system performance is substantially improved through sequence-level discriminative training. The state-of-the-art objective for this is #abbr.a[LF-MMI] @povey_purely_2016. The MMI criterion aims to maximize the mutual information between the observation sequence $S$ and the reference word sequence $T$. This is achieved by maximizing the log-likelihood of the correct transcription (numerator) while minimizing the log-likelihood of all possible transcriptions (denominator):

$ cal(L)_(text("MMI"))(theta) = log (p_theta(S | cal(M)_T) p(T)) / (sum_(T') p_theta(S | cal(M)_(T')) p(T')) $

Here, $cal(M)_(T')$ is the HMM corresponding to a word sequence $T'$, and $p(T')$ is a language model probability. The numerator represents the likelihood of the correct transcription, while the denominator sums over the likelihoods of all possible transcriptions, explicitly creating a margin that pushes down the scores of incorrect hypotheses. The "Lattice-Free" component streamlines this process by using a simpler phone-level decoding graph, allowing for more efficient end-to-end discriminative training.

=== End-to-end models

#abbr.a[E2E] models unify the entire #abbr.a[ASR] pipeline into a single, deep neural network. Transformer-based models are the standard architecture choice for this, with their Conformer @gulati_conformer_2020 variant frequently used, as it allows for some local context using convolutions. These models learn a direct mapping from an acoustic sequence $S$ to a label sequence $T$ (e.g., characters or sub-words). The key enabling technology for this is the #abbr.a[CTC] loss function @graves_ctc_2012.

CTC circumvents the need for pre-aligned data by introducing a special "blank" token (–) to the output vocabulary. It defines a many-to-one mapping function, $cal(B)$, that first collapses any repeated non-blank labels and then removes all blank tokens from a frame-level output path $bold(pi)$ to yield the final, shorter label sequence $T$. The total conditional probability of the target sequence is the sum of probabilities of all alignment paths that map to it:

$ p(T|S) = sum_(bold(pi) in cal(B)^(-1)(T)) p(bold(pi)|S) $

The probability of any single path $bold(pi)$ is calculated as the product of the per-timestep softmax outputs from the neural network:

$ p(bold(pi)|S) = product_(t=1)^T p(pi_t | S) $

The CTC loss is the negative log-likelihood of this total probability, $cal(L)_("CTC") = -log p(T|S)$, which is calculated efficiently using dynamic programming. By maximizing the summed probability of all valid paths, the CTC objective implicitly minimizes the probability of all other label sequences, making it an inherently discriminative training criterion.

=== Sequence-to-sequence models

An alternative #abbr.a[E2E] paradigm is the sequence-to-sequence (seq2seq) model, often employing an attention mechanism @chan_listen_2016. These models consist of an encoder network, which maps the input acoustic sequence $S$ into a high-level representation, and a decoder network, which autoregressively generates the output transcription $T$ one token at a time. The core mechanism linking the two is attention, which at each decoding step computes a context vector as a weighted average of the encoder's output states. This allows the decoder to dynamically focus on the most relevant acoustic frames when predicting the next output token.

The model is trained to maximize the conditional probability of the target sequence, which is factorized using the chain rule:

$ p(T|S) = product_(i=1)^L p(t_i | t_(<i), S) $

The training objective is therefore a standard cross-entropy loss over the predicted token distributions:

$ cal(L)_("att") = -sum_(i=1)^L log p(t_i | t_(<i), S) $

While powerful, pure attention-based models do not explicitly enforce monotonic alignment between speech and text, which can lead to instability during training and inference. This has led to the development of hybrid CTC/attention models @watanabe_joint_2017 that combine both objectives in a multi-task learning framework. The CTC loss acts as a regularizer, encouraging monotonic alignments, while the attention component models global dependencies, resulting in more robust and accurate systems.

=== Pretrain-finetune approach

In recent #abbr.a[ASR] works, #abbr.a[SSL] is frequently employed, where a model is first pre-trained on a task like predicting masked portions of the speech. The resulting model is then fine-tuned with an additional "head" for the desired task and objective, most commonly #abbr.a[CTC]. See @02_ssl for more details.