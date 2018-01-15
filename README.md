# rnn-benchmark
Benchmarking Recurrent Architectures for Robust Continuous Control

The code in this repository is based on an (outdated) fork of [rllab](https://github.com/openai/rllab). Please refer to that repository for full documentation.

The main contribution of this repository is additional (Theano) RNN policy implementations on top of the ones provided by rllab:
- [Discrete-Time RNN (Elman network)](https://github.com/sytham/rnn-benchmark/contrib/rnn/tensor/policies/gaussian_dtrnn_policy.py)
- [Continuous-Time RNN](https://github.com/sytham/rnn-benchmark/contrib/rnn/tensor/policies/gaussian_ctrnn_policy.py)
- [Echo State Network](https://github.com/sytham/rnn-benchmark/contrib/rnn/tensor/policies/gaussian_esn_policy.py)
