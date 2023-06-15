sequence_test_marti

Tests for change in a sequence of point data using a martingale test statistic. 

This code is based on the paper (https://arxiv.org/pdf/2306.01566.pdf). The bounds in Theorem 1 and Lemma 1 are computed in the file **compute_bounds.py**. Further, for given point data, one can compute the martingale using the file **compute_martingale.py**.  An example of this for simulated data can be found in **example.py**. If you somehow have access to the biomechanical knee angles (also possible on reasonable request to the author), you can use the **example_running_data.py** in order to do the procedures of the aforementioned paper. 



