# NLP_assignment_2
Develop a basic probabilistic parser for French that is based on the CYK algorithm and the PCFG model and that is robust to unknown words.



## Installing requirements.
To run the code, follow those steps:

Install requirements (in the repository):

```
pip install -r requirements.txt
```

## Launching the parser
### To run the parser on the test_corpus 

```
bash run.sh --train-size 0.8 --test-size 0.1 --bigram-coef 0.2 --output-path 'evaluation_data.parser_output'
```

if this command does not work, please try 

```
python main.py --train-size 0.8 --test-size 0.1 --bigram-coef 0.2 --output-path 'evaluation_data.parser_output'
```

(The main.py works when the sequoia_corpus and polyglot are in the same folder)

### To run the parser on a new setence (with default parameters like previously)

```
bash run.sh --text-path 'new_sentence.txt' --text-output-path 'new_sentence.parser_output'
```
If you run this command, please be aware that it only accepts a file with one sentence.
