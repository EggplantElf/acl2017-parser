Requirements:
    - Python 2.7
    - Numpy
    - Theano
    - Lasagne

Command:
    # Train and test
    python main.py -conll06 -train [train_file] -dev [dev_file] -test [test_file] -model_to [model_dir] -type char -char_model CNN -out [output_file]

    # Test only 
    python main.py -conll06  -test [test_file] -model_from [model_dir] -type char -out [output_file]

