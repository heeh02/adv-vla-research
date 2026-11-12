# Terms used in the manuscript

| Term | Meaning | Defined |
|---|---|---|
| Paired rollout | Clean and perturbed execution sharing the starting state, instruction and policy-noise labels; later observations follow each execution | Sec. 3 |
| Visual response | Difference between the recorded clean and perturbed tensor at a specified site | Sec. 3 |
| Response magnitude | Tensor-response norm divided by the corresponding clean norm | Sec. 3 |
| Response dictionary | Channel directions fitted to training responses; each test token/position has its own coefficients | Secs. 3–4 |
| Rank | Number of dictionary directions; distinct from channel width, number of selected raw channels or the complete tensor's degrees of freedom | Sec. 3; Supplement S2 |
| Residual retention | Reconstruction score after accounting for the saved training position mean | Sec. 4; Supplement S3 |
| Command readout | An affine predictor trained on response features to predict command differences | Sec. 4; Supplement S3.3 |
| First-flow | The native policy's first generation-velocity difference, used as a downstream predictive baseline | Supplement S3.3 / S11 |
| Attention shift | Total variation of the attention probability distribution over the stated key set | Sec. 3 |
| Mass-weighted shift | Within-region probability redistribution multiplied by its attention mass | Sec. 3 / Table 2 |
| Attention output | Value-weighted, projected output before the adaptive attention gate; the reported P/W features come from diagnostic replay | Sec. 3; Supplement S1 |
| Command | The first-five returned action coordinates; continuous gripper coordinates precede physical sign formatting | Sec. 3; Supplement |
| Motion | Measured end-effector position or orientation difference at the specified completed control step | Sec. 6 |
| Sampling reference | Difference between two clean executions with different policy-noise labels | Sec. 3 |
| Same-noise repeat | Repeated clean execution in the same execution context with the same input and noise schedule | Sec. 3 |
| Harm | Clean success paired with perturbed failure | Sec. 3 |
| Recovery | Clean failure paired with perturbed success | Sec. 3 |
| Net task cost | Decrease in paired success rate; recovery minus harm gives the signed success-rate change | Secs. 3, 7 |
| Outcome flip | Either harm or recovery, preserving direction when reporting cost | Sec. 3 |
| Identification bound | Range over outcomes left unknown in the planned population; distinct from a sampling confidence interval | Supplement S1 / S6 |

Physical query time, flow-generation time, planned action position and completed control step are different clocks. Native hidden states, diagnostic attention outputs and issued commands are different measurement objects.
