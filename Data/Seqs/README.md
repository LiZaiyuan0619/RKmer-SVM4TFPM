# README

The four directories named **"Liu"**, **"Li"**, **"Nguyen"**, and **"Ours"** contain the test results on the mouse dataset. **TF.txt** and **TFPM.txt** are the sequences predicted by the two-step method, while **TFPM2.txt** corresponds to the results of the second step for identifying TFPMs. The first two methods rely on online servers and do not provide public code. Since the web server is currently unavailable, we used the publicly available implementation for **"Nguyen"**, and report its best prediction results.

The **"1000SeqsTest"** directory includes 1,000 sequences used to evaluate computational efficiency. These sequences were randomly selected, with an average length of 539.86 amino acids (ranging from 51 to 4,834), to benchmark inference performance.
