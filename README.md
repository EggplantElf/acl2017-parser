A greedy transition-based dependency parser with CNN as character composition model, described in the ACL2017 paper:
    https://arxiv.org/abs/1705.10814

Requirements:

    - Python 2.7

    - Numpy
    
    - Theano
    
    - Lasagne

Command:

    # Train
    python main.py -conll06 -train [train_file] -dev [dev_file] -model_to [model_dir] -type char -char_model CNN -out [output_file]

    # Test 
    python main.py -conll06  -test [test_file] -model_from [model_dir] -type char -out [output_file]

