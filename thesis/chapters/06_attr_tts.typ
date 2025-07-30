#import "../abbr.typ"
#import "../math.typ": *
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
#import "../comic.typ"

== Enhancing Synthetic Speech Diversity <06_attr>

Building on the gaps identified and quantified via WERR in Chapter 5, this chapter details methods to increase the diversity of synthetic speech and presents experimental results evaluating their impact. As established, synthetic speech often lacks the variability of real speech, leading to suboptimal ASR performance. Here, we explore three complementary paradigms—latent representations, explicit conditioning, and post-generation augmentation—aiming to bridge this divide. By systematically enhancing diversity, we test whether these approaches can reduce the distributional distance, as measured by WERR. The chapter progresses from unsupervised latent capture to targeted control and simulation, culminating in empirical validation. Throughout, we formally define key processes using notation consistent with the introduction (e.g., $Q_theta$ for TTS approximations).

=== Learning Latent Representations

Latent representations offer an unsupervised way to capture and inject stylistic variability into synthetic speech, addressing the one-to-many challenge without explicit attribute extraction. Before examining specific techniques, consider their shared motivation: real speech varies subtly in ways not captured by text alone (e.g., emphasis or emotion), and latent spaces learn these patterns from data, enabling sampling for diverse outputs. This paradigm has proven valuable in TTS-for-ASR, where broader coverage improves model robustness @sun_generating_2020. Mathematically, these methods model a latent variable $z$ such that the TTS distribution becomes $Q_theta (tilde(s) | t, z)$, with $z$ inferred from training data to approximate real variability.

==== Global Style Tokens

Global Style Tokens (GST) provide a discrete approach to style modeling, learning a fixed set of embeddings that represent different speaking styles @wang_style_2018. During training, an attention mechanism weighs these tokens based on reference audio, encoding the utterance's style as a weighted combination. Formally, given reference speech $s$, GST computes a style vector as a convex combination of $K$ learned tokens $G = {g_1, ..., g_K}$: $z = sum_k alpha_k g_k$, where $alpha_k$ are attention weights. At inference, tokens can be selected or interpolated to control output style, such as making speech more expressive or neutral. While effective for categorical style shifts, GST's discrete nature limits fine-grained variation compared to continuous methods.

==== Variational Autoencoder

Variational Autoencoders (VAE) extend this by learning a continuous latent distribution, allowing for smoother and more nuanced sampling @kingma_auto-encoding_2013. A VAE encodes reference speech $s$ into parameters of a Gaussian posterior $q_phi(z | s) = cal(N)(mu_phi(s), sigma_phi(s))$, approximating the true posterior via the evidence lower bound (ELBO): $cal(L)_"ELBO" = EE_(q_phi(z|s)) [log p_theta (s | z)] - "KL"(q_phi (z|s) || p(z))$, where $p(z)$ is a standard Gaussian prior. The TTS decoder then reconstructs $tilde(s) approx p_theta (s | z)$. By regularizing the latent space, inference-time sampling from $p(z)$ generates novel styles. This flexibility has made VAEs a go-to for TTS-for-ASR, yielding up to 16% WER improvements in low-real-data scenarios @sun_generating_2020. However, VAEs can suffer from posterior collapse if not carefully tuned, reducing effective diversity.

=== Explicit Conditioning on Attributes

For more interpretable and targeted diversity, explicit conditioning directly incorporates measurable speech attributes into the TTS process. Unlike latent methods, which infer styles abstractly, this approach specifies desired traits (e.g., fast speaking rate), ensuring generated speech aligns with real distributions. This is particularly useful for TTS-for-ASR, where controlling prosody or acoustics can mimic real-world variability @rossenbach_duration_2023. Formally, the TTS model $f^"TTS"_theta$ is trained to minimize a loss over conditioned outputs: $cal(L)(theta) = EE_(t,s,z) [l(f_theta (t,z), s)]$, where $z$ are extracted attributes and $l$ is a reconstruction loss (e.g., MSE).

==== Variance Adapter

Modern non-autoregressive models often implement this via a variance adapter, inserted between the text encoder and spectrogram decoder @ren_fastspeech_2019@ren_fastspeech_2021. Let $h$ be the encoder's hidden representation of text $t$. The adapter predicts and embeds attributes $z$ (e.g., pitch, energy), adding them to $h$: $h' = h + "embed"(z)$. During training, ground-truth $z$ from $s$ are used; at inference, predicted or sampled $z$ enable diversity. Without inputs, it defaults to mean values, causing collapse—supplying varied $z$ unlocks control, enabling diverse synthesis.

==== Controllable Attributes <06_prosodic_correlates>

Attributes are typically perceptual correlates (detailed in Chapter 2), enabling fine-tuned generation. Prosody includes pitch (F0 from PyWORLD @morise_world_2016), energy (RMS of Mel frames), and duration (speaking rate via forced alignment @mcauliffe_montreal_2017). Acoustic environment uses SRMR for reverberation @kinoshita_reverb_2013 and WADA SNR for noise @kim_wada_2008. By conditioning on these, TTS systems generate speech matching specific contours or environments, enhancing ASR training data realism. To generate realistic $z$ at inference, we use speaker-dependent Gaussian Mixture Models (GMMs): Fit a GMM per speaker on training attributes, then sample $z tilde "GMM"$ for synthesis, approximating real variability.

=== Post-Generation Data Augmentation

Post-generation augmentation complements internal methods by transforming clean synthetic output to simulate external variability. This external approach is straightforward yet powerful for TTS-for-ASR, as it directly addresses environmental mismatches without altering the core synthesis model @rossenbach_synthattention_2020. Formally, given synthetic $tilde(s) = f_theta (t, z)$, apply augmentations $a(tilde(s))$ (e.g., noise addition) to produce $tilde(s)' approx Q_theta (s | t, z, z_"env")$, where $z_"env"$ models real acoustics.

Techniques include adding background noise from diverse sources or convolving with Room Impulse Responses (RIRs) to mimic reverberation in varied spaces (e.g., RT60 0.15-0.8s, probability 0.8). Tools like audiomentations enable probabilistic application (e.g., Gaussian noise with SNR 5-40 dB). While excellent for acoustics, it cannot retroactively adjust prosody or speaker traits, making it synergistic with latent/explicit methods for comprehensive diversity.

=== Experimental Design

We test these paradigms via controlled experiments, incrementally enhancing a baseline TTS system and evaluating via WERR (as defined in Chapter 5). The setup ensures differences stem from diversity improvements, not confounding factors. TTS training formally minimizes $cal(L)(theta) = EE_(t,s,z) [||s - f_theta (t, z)||^2_2]$ (MSE loss), approximating $Q_theta (s | t, z)$. ASR training uses LF-MMI on (speech, transcripts) pairs to learn $P_Phi (t | s)$, with WER as evaluation metric.

All experiments use the train-clean-360 split of LibriTTS @zen_libritts_2019, selecting speakers with >=100 utterances (684 total). Half utterances per speaker form the TTS training set; the other half is reserved for inference (test set: unseen utterances for evaluation). Transcripts are selected to balance across speakers, generating 10 hours of synthetic audio with equal transcripts and speakers for fair comparison. Seen/unseen splits ensure no overlap, with unseen transcripts paired to probe generalization.

The baseline is a multi-speaker FastSpeech 2 @ren_fastspeech_2021, conditioned on d-vector embeddings ($E_text("SPK")$) @wan_generalized_2018. A HiFi-GAN vocoder converts Mel spectrograms to waveforms @kong_hifigan_2020. ASR uses a 6-layer hybrid HMM-TDNN with LF-MMI @povey_kaldi_2011 (4 epochs, ~3-4 hours on 4 GTX 1080 Ti), kept minimal to attribute results to data quality.

The Attributes system conditions on GMM-sampled utterance-level means for pitch ($F_("F0")$), energy, duration, SRMR, and SNR (2 components per GMM, variance floor 10^{-3}). An Oracle uses ground-truth values. Augmentation applies Gaussian noise (SNR 5-40 dB) and RIRs (RT60 0.15-0.8s, probability 0.8) post-synthesis via audiomentations.

#comic.comic((80mm, 40mm), "Comic of TTS system with attribute conditioning and augmentation pipeline", blue) <fig_tts_aug>

=== Results and Discussion

Results reveal targeted enhancements reduce the gap, though not fully, with detailed per-domain analysis highlighting strengths and limitations. The baseline yields a WERR of 3.66 (cross-referenced from Chapter 5), confirming limited variability. The Attributes system drops it to 3.55, showing varied prosody/acoustics help. Augmentation further reduces to 3.31 (10% relative gain), emphasizing environmental simulation's impact. The Oracle (3.24) bounds potential, highlighting GMM limitations.

#comic.comic((120mm, 120mm), "TTS-for-ASR results, evaluated via WERR. W2 metrics per domain show reductions, with augmentation yielding the largest overall gain.", blue) <tbl_werr_results>


While ASR language models can influence WER, our setup minimizes this by prioritizing acoustic modeling with minimal LM interference. Augmentation outperforms conditioning likely due to better real-world robustness; ablating attributes shows prosody contributes most. However, gaps persist, suggesting inherent TTS limits—diminishing returns motivate exploring scaling in Chapter 7.

In the speaker domain, Attributes reduces intra-speaker distance (FD-Intra) by 16.6% via GMM-sampled d-vectors, increasing variety. Inter-speaker distance (FD-Inter) improves 12.2%, possibly from better utterance matching during training. Augmentation slightly worsens these, but overall WR drops. The Oracle excels here, suggesting GMM limitations for speakers.

For prosody, baseline distributions diverge (e.g., bimodal F0, shifted energy, low-variance duration). Attributes mitigates this, reducing distances (e.g., F0 by ~89%), positively impacting WR. Oracle shows high energy distance yet low WR, indicating energy's lesser role in the overall gap.

Acoustic environment is crucial: Augmentation reduces WR by 6.7% (largest single gain), slashing SRMR (97%) and WADA SNR (60%). Environment prediction alone slightly increases WR, but with Attributes, SRMR drops 22.1%. WADA SNR shows minimal change, possibly due to inability to model Gaussian noise.

=== Limitations

These methods enhance diversity but face limits: latent approaches may collapse modes, conditioning relies on accurate sampling, and augmentation ignores core synthesis flaws. Gains plateau, indicating a ceiling. While they improve WERR, full parity requires more data—scaling offers another avenue, explored next.