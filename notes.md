

## GPT2 model finetuning

### training library selection

There are various approaches to finetune GPT2 model. one way is to finetuning it during against the pretrained model from the [GPT2 repo](git@github.com:jackxxu/gpt2_music.git). However, it will require that some work in setup. We could also use the [Huggingface GPT2 transformers](https://huggingface.co/transformers/model_doc/gpt2.html) to do similar things. However, it similarly will probably require certain level of setup before starting to run. 

What we have found is this library called [`gpt-2-simple`](https://github.com/minimaxir/gpt-2-simple). The library is quite popular with 2.4k stars. We decided to try this library because it probably a nice wrapper around the GPT2 pretrained model, and provides 2 convenience methods `finetune` for finetuning, and `generate` to auto generate text. In addition, it also provides a convenient command line so that we don't have to write code.


### data preparation

in order to feed music information into GPT2 model, we first need to convert the data into a text format that GPT2 model is able to proceed. for that, we used [`music21` library from MIT](http://web.mit.edu/music21/). `music21` parses the mid file into notes and chords (concurrent notes) and other types, which we can use to feel into the model. 

There are a few types of the music elments that we are interested in: 
1. notes: a single note that is played at a particular moment for a duration. We represent it by using its note + its octave, such as "C#3", "D-2", or "E3". 
2. chords: a collection of the notes that are played concurrently. We represent it by simply concatenating the aforementioned note representations with space.
3. other notes are displayed with their `music21` name. 

We then concatenate all these elements with command and output them into a single-column CSV file. Then we are ready for feeding the songs into the model for finetuning.


### finetuning and generation



## Visualization



## Discussion


## Future Improvements

