#import "../abbr.typ"
#import "../quote.typ": *
#import "../comic.typ"
#import "../moremath.typ": *

== Automatic speech recognition <04_asr>

#q(
  [#citep(<jurafsky_slp_2024>)],
  [#emph[Speech and Language Processing]],
  [The modern task of speech synthesis, also called text-to-speech or TTS, is exactly the reverse of ASR; to map text $dots$ to an acoustic waveform.]
)

Automatic Speech Recognition (#abbr.a[ASR]) is the task of automatically transcribing a speech utterance into its corresponding text sequence. Formally, given an acoustic signal $S$ and its corresponding text $T$, #abbr.a[ASR] aims to model the conditional probability distribution $P(T|S)$. This is typically framed as a prediction task, where a model $f^"ASR"_theta$ with parameters $theta$ is trained on $(T,S)$ pairs to learn an approximation of this distribution. This model can then be used to infer the most probable text sequence $T$ for a given input $S$:

$ T = argmax_(T in cal(T)) P_theta (T|S) $

Modern high-performance #abbr.a[ASR] systems are almost exclusively trained using discriminative objectives. In contrast to generative approaches that might model the joint probability $P(S, T)$, discriminative models are optimized to directly model the posterior probability $P(T|S)$, maximizing the score of the correct transcription while simultaneously minimizing the scores of all incorrect competing hypotheses. This approach has proven more effective at achieving low #abbr.a[WER]. The two dominant paradigms for discriminative #abbr.a[ASR] training are hybrid HMM-DNN systems and end-to-end models. For a comprehensive review of #abbr.a[E2E] #abbr.a[ASR] models, see #citep(<prabhavalkar_survey_2023>).

#figure(
image("../figures/4/tts_asr_comp.png", width: 100%),
caption: [Comparison of TTS and ASR problems, with TTS increasing diversity, and ASR both decreasing diversity through normalisation.],
placement: top,
) <fig_tts_asr_comp>


=== Speech variability

While TTS is a "one-to-many" problem where a single text can have many valid acoustic realisations, ASR is a "many-to-one" problem. A single intended text sequence can be realized as a vast number of acoustically distinct signals, all of which should map back to the same transcription. This variability is the central challenge of ASR and arises from the factors of speech discussed in @02_factors[Chapter]. Variability from the #smallcaps[Speaker] factor includes differences in vocal tract length, pitch, and accent. The #smallcaps[Prosody] factor introduces variability through changes in speaking rate, stress, and intonation. Finally, the #smallcaps[Ambient] factor adds variability through background noise, reverberation, and different recording channels.

To build a robust ASR system, this acoustic variability must be handled. Historically, this was primarily achieved through normalisation, a process designed to reduce "extrinsic" variability in the input signal before it reaches the acoustic model. To handle speaker variability, early systems used techniques like Vocal Tract Length Normalization (VTLN) @eide_vtln_1996, which warps the frequency axis of the spectrum to compensate for differences in vocal tract size, effectively normalizing low-level #smallcaps[Speaker] features like formant frequencies. Cepstral Mean and Variance Normalization (CMVN) @viikki_cmvn_1998 is another common technique that normalizes feature vectors on a per-speaker or per-utterance basis to have zero mean and unit variance.

In contrast to this normalization-based approach, the modern deep learning paradigm handles variability by making the model itself invariant through training on diverse data. This is often achieved with data augmentation, where the training set is artificially expanded by creating modified copies of the original audio. Common techniques include speed perturbation, which makes the model robust to changes in speaking rate (#smallcaps[Prosody]), and the addition of background noise and reverberation, which makes the model robust to different #smallcaps[Ambient] conditions, as well as masking part of the features such as with commonly used SpecAugment @park_specaugment_2019. By exposing the model to a wide range of acoustic variations during training, it should learn to focus more on the core linguistic content while ignoring the extrinsic factors, reducing potential overfitting.

=== Hybrid HMM-DNN systems

Hybrid systems combine Deep Neural Networks (DNNs) for their powerful pattern recognition capabilities with the temporal modeling structure of Hidden Markov Models (HMMs). This architecture has been the dominant paradigm for high-performance ASR for many years and is centrally supported by toolkits like Kaldi @povey_kaldi_2011. In this setup, the DNN, often a Time-Delay Neural Network (TDNN) @peddinti_time_2015, processes acoustic feature vectors $s_k$ at each timestep $k$ and outputs a posterior probability distribution over the set of HMM states $q$:

$ P(q_k|s_k; theta) $

While this acoustic model can be trained with a simple cross-entropy loss against frame-level alignments generated from a previous model, system performance is substantially improved through sequence-level discriminative training. The state-of-the-art objective for this is Lattice-Free Maximum Mutual Information (LF-MMI) @povey_purely_2016. The MMI criterion aims to maximize the mutual information between the observation sequence $S$ and the reference word sequence $T$. This is achieved by maximizing the log-likelihood of the correct transcription (the numerator) while minimizing the log-likelihood of all possible competing transcriptions (the denominator):

$ cal(L)_(text("MMI"))(theta) = log (P_theta (S | cal(M)_T) P(T)) / (sum_(T') P_theta (S | cal(M)_(T')) P(T')) $

Here, $cal(M)_(T')$ is the HMM decoding graph corresponding to a word sequence $T'$, and $P(T')$ is a language model probability. The "Lattice-Free" component streamlines this process by using a simpler phone-level decoding graph for the denominator, allowing for more efficient training. The conventional LF-MMI pipeline, however, still relies on a previously trained model to generate alignments for creating context-dependent decision trees. Addressing this, the work of @hadian_lfmmi_2018 introduces an "end-to-end" version of LF-MMI that operates in a flat-start manner. This approach eliminates the need for any previously trained models, forced alignments, or state-tying decision trees. It uses full biphones for context-dependency without tree-based tying and, crucially, uses the standard composite HMM (with self-loops) for the numerator graph. This gives the neural network more freedom to learn the alignments internally, making the training process truly "end-to-end" in the sense of not requiring a bootstrapped alignment model, while still retaining the powerful discriminative properties of the MMI objective.

=== End-to-end models

End-to-end (E2E) models unify the entire ASR pipeline into a single, deep neural network, learning a direct mapping from an acoustic sequence $S$ to a label sequence $T$ (e.g., characters or sub-words). Transformer-based models, particularly the Conformer variant @gulati_conformer_2020 which enhances Transformers with local context via convolutions, are a standard architecture choice. These models are trained with objectives that handle the alignment between the long acoustic sequence and the short label sequence automatically.

==== Connectionist Temporal Classification (CTC)

The Connectionist Temporal Classification (CTC) loss function is a highly popular approach for E2E ASR due to its simplicity and efficiency @graves_ctc_2012. CTC circumvents the need for pre-aligned data by introducing a special "blank" token (–) to the output vocabulary. It defines a many-to-one mapping function, $cal(B)$, that first collapses any repeated non-blank labels and then removes all blank tokens from a frame-level output path $bold(pi)$ to yield the final, shorter label sequence $T$. The total conditional probability of the target sequence is the sum of probabilities of all alignment paths that map to it:

$ P(T|S) = sum_(bold(pi) in cal(B)^(-1)(T)) P(bold(pi)|S) $

The probability of any single path is the product of the per-timestep softmax outputs from the network: $P(bold(pi)|S) = product_(k=1)^K P(pi_k | S)$. The CTC loss is the negative log-likelihood of this total probability, $cal(L)_("CTC") = -log P(T|S)$, which is calculated efficiently using dynamic programming. By maximizing the summed probability of all valid paths, the CTC objective implicitly pushes down the probability of all other label sequences, making it an inherently discriminative training criterion.

==== Sequence-to-sequence models

An alternative E2E paradigm uses sequence-to-sequence (seq2seq) models, often employing an attention mechanism @chan_listen_2016. These models consist of an encoder, which maps the input acoustic sequence $S$ into a high-level representation, and a decoder, which autoregressively generates the output transcription $T$ one token at a time. The core mechanism is attention, previously detailing in @03_arch, which at each decoding step computes a context vector as a weighted average of the encoder's outputs. This allows the decoder to dynamically "focus" on the most relevant acoustic frames when predicting the next output token. The model is trained to maximize the conditional probability of the target sequence, which is factorized using the chain rule:

$ P(T|S) = product_(k=1)^K P(t_k | t_(<k), S) $

The objective is a standard cross-entropy loss over the predicted token distributions: $cal(L)_("att") = -sum_(i=1)^L log P(t_i | t_(<i), S)$. While powerful, pure attention-based models do not explicitly enforce monotonic alignment, which can lead to instability. This has led to the development of hybrid CTC/attention models @watanabe_joint_2017 that combine both objectives in a multi-task learning framework. The CTC loss acts as a regularizer, encouraging monotonic alignments, while the attention component models global dependencies, resulting in more robust and accurate systems.

#figure(
image("../figures/4/hybrid_encdec_ctc.png", width: 100%),
caption: [Hybrid and end-to-end encoder-decoder and CTC models and their connection to a language model.],
placement: top,
) <fig_tts_asr_comp>

==== The Pre-train/Fine-tune Paradigm

The current state-of-the-art in Automatic Speech Recognition is dominated by the pre-train/fine-tune paradigm, which leverages Self-Supervised Learning (SSL) to learn powerful representations from vast quantities of unlabeled audio data. This approach fundamentally separates the problem into two stages: first, a large, general-purpose model is *pre-trained* on a data-rich but label-agnostic task; second, this pre-trained model is *fine-tuned* on a smaller, labeled dataset for the specific downstream task of ASR. This methodology has proven exceptionally effective, as it shifts the primary data requirement from expensive, manually transcribed speech to readily available, unlabeled audio.

The pre-training phase relies on a "pseudo-task" that the model can learn from the raw audio signal without any human-provided labels. The dominant approach for this is masked prediction, which forces the model to learn the inherent structure of speech—including its phonetic properties, co-articulation effects, and prosodic patterns—to succeed at the prediction task. The resulting model is a powerful feature encoder that produces the high-level #smallcaps[Generic] representations discussed in @02_factors[Chapter]. Two of the most influential models in this area are wav2vec 2.0 and HuBERT. The wav2vec 2.0 model @baevski_wav2vec_2020 learns by masking parts of a latent speech representation and training the model on a contrastive task to identify the correct quantized version of the masked content from a set of distractors. HuBERT (Hidden Unit BERT) @hsu_hubert_2021 also uses a masked prediction framework but employs an iterative process where the model is trained to predict cluster assignments of the acoustic data, with the model's own improved representations being used to generate better cluster targets for the next iteration. As discussed in @02_factors, the representations learned through these objectives are highly effective precisely because they capture a holistic view of the speech signal, with different layers corresponding to different levels of abstraction from acoustic to linguistic @pasad_layer-wise_2021.

Once the pre-training phase is complete, the resulting deep encoder model is adapted for the specific task of ASR in the fine-tuning phase. This process is remarkably efficient. A lightweight, task-specific "head" is added on top of the pre-trained encoder, which for ASR is typically a single linear layer with a softmax function whose output dimension corresponds to the task vocabulary. The entire model, comprising the pre-trained encoder and the new head, is then trained on a smaller, labeled dataset using a standard ASR objective, most commonly the CTC loss. The pre-trained weights of the encoder provide a powerful initialization, so the model only needs to learn how to map its already-rich representations to the specific output labels of the ASR task. This pre-train/fine-tune paradigm yields state-of-the-art results on numerous ASR benchmarks, often outperforming models trained from scratch by a significant margin, especially in low-resource settings where labeled data is scarce @prabhavalkar_survey_2023.

=== Language Modeling

A central component of high-performance ASR systems is the Language Model (LM). While the acoustic model is responsible for mapping sound to linguistic units, the language model is responsible for modeling the probability of a sequence of those units, typically words. Formally, an LM aims to estimate the prior probability $P(T)$ of a given text sequence $T$. Its primary purpose is to guide the ASR decoding process, constraining the vast search space of possible transcriptions to only those that are linguistically plausible. The LM acts as a powerful regularizer, helping the system resolve acoustic ambiguities—for instance, distinguishing between "recognize speech" and "wreck a nice beach," which can be acoustically similar but have vastly different linguistic probabilities.

The importance of the language model is fundamentally linked to the chronic mismatch between the availability of transcribed speech data and text-only data. Acquiring and accurately transcribing large speech corpora is an expensive and labor-intensive process, resulting in datasets that, while large, are dwarfed by the sheer volume of text available from sources like the web, books, and news articles. A separately trained language model provides a mechanism to inject the rich statistical knowledge learned from these massive text corpora into the ASR system, a strategy that is especially critical for achieving good performance in low-resource scenarios where transcribed speech data is particularly scarce @liu_lmasr_2024. The method of integration, however, differs significantly between hybrid and end-to-end architectures.

==== Language Modeling in Hybrid HMM-DNN Systems

In the classic hybrid ASR paradigm, the acoustic model and the language model are distinct, independently trained components that are combined during the decoding stage. The decoding process is governed by Bayes' rule, which formulates the search for the most probable text sequence $hat(T)$ as:

$ hat(T) = argmax_(T) P(S|T) P(T) $

Here, $P(S|T)$ is the likelihood of the acoustic signal given a text sequence, which is provided by the HMM-DNN acoustic model. The term $P(T)$ is the prior probability of the text sequence, provided by the language model. During decoding, these two probability sources are combined, typically within a Weighted Finite-State Transducer (WFST) framework, which is the core of the Kaldi toolkit @povey_kaldi_2011. The WFST composes the graphs representing the HMM, the lexicon (mapping phones to words), and the language model into a single, large search space. The decoder then finds the highest-probability path through this composite graph for a given acoustic input. For this paradigm, the standard language modeling technique for many years has been the use of N-gram models, which estimate the probability of a word given the previous N-1 words. These are typically trained using toolkits like SRILM @stolcke_srilm_2002 or KenLM @heafield_kenlm_2011 on vast text corpora.

==== Language Modeling in End-to-End Systems

End-to-end models present a more complex landscape for language model integration. The autoregressive decoder of a sequence-to-sequence model, by its very nature, learns an implicit language model from the transcripts of the speech data it is trained on. As it learns to predict the next token given the previous ones, it internalizes the linguistic patterns present in the training text. However, this implicit LM is limited by the size and domain of the transcribed speech corpus, which is often much smaller than available text-only corpora. To leverage this external text data, several fusion techniques have been developed to integrate an external, separately trained LM.

The most common technique is shallow fusion @chorowski_shallow_2017. During the beam search decoding process, the score of each candidate hypothesis at each step is calculated as a linear interpolation of the log-probability from the ASR model's decoder and the log-probability from the external LM. This method is simple to implement and very effective, acting as a post-hoc rescoring mechanism that guides the search towards more fluent sentences. This was a key technique used in systems like Deep Speech 2 to achieve high performance @amodei_deepspeech2_2016.

A more tightly integrated approach is deep fusion @toshniwal_fusion_2018. Instead of simply combining the final probability scores, this method combines the internal hidden state vectors of the ASR decoder and the external LM at each decoding step. This allows the external LM's rich contextual information to influence the ASR decoder's internal predictions more directly, potentially leading to better performance, although it is more architecturally complex. As with hybrid systems, the external LMs used for fusion have evolved from N-gram models to more powerful neural LMs, such as those based on LSTMs or Transformers, which can capture much longer-range dependencies and more nuanced linguistic context, further improving the final accuracy of the ASR system.

While this thesis is motivated by the potential of using TTS-generated data for ASR training, leveraging the ability of synthesis to create diverse acoustic realisations for a given text and thereby addressing the many-to-one nature of the ASR problem, the role of the language model complicates this picture. The disproportionate impact of language models trained on massive text-only corpora, shifts our focus away from the textual content of synthetic speech and to its acoustic properties. Since the linguistic plausibility of a transcription can be powerfully constrained by an external LM trained on billions of words, the primary contribution of synthetic data to an ASR system is not its linguistic novelty, but its acoustic realism and variability. Therefore, the subsequent chapters of this thesis focus on the acoustic realism and distributional characteristics of synthetic speech, and how these fundamental acoustic properties contribute to—or detract from—the training of robust and effective ASR models, while keeping the lexical content equal between synthetic and real speech.