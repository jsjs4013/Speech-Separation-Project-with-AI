# J_and_J_research
todo :
1. [ ] Relative positional encoding
2. [x] attention mask 적용
3. [x] padding적용 
4. [x] padding은 loss계산안하기 적용

LOSS - problems
1. si-sdr loss 적용
2. gan loss 적용 (adversarials)

Raw data embeddings
 - No STFT!!
1. Raw data embedding<br>
 -> Mask learning...<br>
 - Raw data window size 조절<br>
Multi-tasking learning<br>
 -> Single Speech Mask Modeling<br>
 - mask 방법은 데이터가 안에 들어오면 거기서 처리해준다.<br>
 -> Speech Order Prediction<br>
 -> Mixed Speech Mask Modeling<br>

2. fine-tuning
Permutation invariant Speech Separation ^^<br>
 -> T5 Speech Token Format
