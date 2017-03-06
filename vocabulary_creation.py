import tensorflow as tf
import functools
from utils.IO_data import create_row_iter, create_csv_batch_iter

tf.flags.DEFINE_string("input_dir", "./data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 2000, "Evaluate after this many train steps")
tf.flags.DEFINE_integer("min_word_frequency", 5, "Minimum frequency of words in the vocabulary")
tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")
tf.flags.DEFINE_integer('batch_size', 20000, 'size of the inputs batch')

FLAGS = tf.flags.FLAGS


def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


def create_vocab(batch_input_iter, min_frequency, max_sentence_len):
    """
    Creates and returns a VocabularyProcessor object with the vocabulary
    for the input iterator.
    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_sentence_len,
        min_frequency=min_frequency,
        tokenizer_fn=tokenizer_fn)

    vocab_processor.fit(create_row_iter(batch_input_iter, func=lambda x:" ".join(x[:2])) )
    return vocab_processor

def write_vocabulary(vocab_processor, file_name, path='./data'):
    """
    Writes the vocabulary to a file, one word per line.
    """
    vocab_size = len(vocab_processor.vocabulary_)
    with open(path + '/' + file_name, "w") as writer:
        for id in range(vocab_size):
            word =  vocab_processor.vocabulary_._reverse_mapping[id]
            writer.write(word + "\n")
    print("Saved vocabulary to {}".format(path + '/' + file_name))

def create_tfrecords_file(input_file_name, output_file_name, example_fn, path='./data'):
    """
    Creates a TFRecords file for the given input data and example transofmration function
    """
    writer = tf.python_io.TFRecordWriter(path + '/' + output_file_name)
    print("Creating TFRecords file at {}...".format(path + '/' + output_file_name))
    for i, row in enumerate(create_row_iter(create_csv_batch_iter(input_file_name, batch_size=FLAGS.batch_size))):
        x = example_fn(row)
        writer.write(x.SerializeToString())
    writer.close()
    print("Wrote to {}".format(path + '/' + output_file_name))


def transform_sentence(sequence, vocab_processor):
    """
    Maps a single sentence into the integer vocabulary. Returns a python array.
    """
    return next(vocab_processor.transform([sequence])).tolist()


def create_text_sequence_feature(fl, sentence, sentence_len, vocab):
    """
    Writes a sentence to FeatureList protocol buffer
    """
    sentence_transformed = transform_sentence(sentence, vocab)
    for word_id in sentence_transformed:
        fl.feature.add().int64_list.value.extend([word_id])
    return fl


def create_example_train(row, vocab):
    """
    Creates a training example for the Ubuntu Dialog Corpus dataset.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    context, utterance, label = row
    context_transformed = transform_sentence(context, vocab)
    utterance_transformed = transform_sentence(utterance, vocab)
    context_len = len(next(vocab._tokenizer([context])))
    utterance_len = len(next(vocab._tokenizer([utterance])))
    label = int(float(label))

    # New SequentialExample, protocol to deal with sequential data (with different timestamp)
    example = tf.train.SequenceExample()
    # “context” for non-sequential features
    example.context.feature["context_len"].int64_list.value.append(context_len)
    example.context.feature["utterance_len"].int64_list.value.append(utterance_len)
    example.context.feature["label"].int64_list.value.append(label)

    # “feature_lists” for sequential features
    example.feature_lists.feature_list["context"].feature.add().int64_list.value.extend(context_transformed)
    example.feature_lists.feature_list["utterance"].feature.add().int64_list.value.extend(utterance_transformed)

    # example.features.feature["context"].int64_list.value.extend(context_transformed)
    # example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
    # example.features.feature["label"].int64_list.value.extend([label])
    return example


def create_example_test(row, vocab):
    """
    Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    context, utterance = row[:2]
    distractors = row[2:]
    context_len = len(next(vocab._tokenizer([context])))
    utterance_len = len(next(vocab._tokenizer([utterance])))
    context_transformed = transform_sentence(context, vocab)
    utterance_transformed = transform_sentence(utterance, vocab)

    # New Example
    example = tf.train.SequenceExample()
    example.feature_lists.feature_list["context"].feature.add().int64_list.value.extend(context_transformed)
    example.feature_lists.feature_list["utterance"].feature.add().int64_list.value.extend(utterance_transformed)

    example.context.feature["context_len"].int64_list.value.append(context_len)
    example.context.feature["utterance_len"].int64_list.value.append(utterance_len)

    # Distractor sequences
    for i, distractor in enumerate(distractors):
        dis_key = "distractor_{}".format(i)
        dis_len_key = "distractor_{}_len".format(i)
        # Distractor Length Feature
        dis_len = len(next(vocab._tokenizer([distractor])))
        # Distractor Text Feature
        dis_transformed = transform_sentence(distractor, vocab)

        example.context.feature[dis_len_key].int64_list.value.append(dis_len)
        example.feature_lists.feature_list[dis_key].feature.add().int64_list.value.extend(dis_transformed)
    return example


if __name__ == "__main__":
    print("Creating vocabulary...")
    input_batch_iter = create_csv_batch_iter('train.csv', batch_size=FLAGS.batch_size)
    vocab = create_vocab(input_batch_iter, min_frequency=FLAGS.min_word_frequency, max_sentence_len=FLAGS.max_sentence_len)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary.txt file
    write_vocabulary(vocab, "vocabulary_my.txt", path=FLAGS.input_dir)

    # Save vocab processor
    vocab.save(path=FLAGS.input_dir + '/' + 'vocab_processor_my.bin')


    # vocab = tf.contrib.learn.preprocessing.VocabularyProcessor(
    #     FLAGS.max_sentence_len,
    #     min_frequency=FLAGS.min_word_frequency,
    #     tokenizer_fn=tokenizer_fn)
    #
    # vocab = vocab.restore('../data/vocab_processor_my.bin')

    # Create validation.tfrecords
    create_tfrecords_file(
        input_file_name='valid.csv',
        output_file_name="validation_my.tfrecords",
        example_fn=functools.partial(create_example_test, vocab=vocab))

    # Create test.tfrecords
    create_tfrecords_file(
        input_file_name='test.csv',
        output_file_name="test_my.tfrecords",
        example_fn=functools.partial(create_example_test, vocab=vocab))

    # Create train.tfrecords
    create_tfrecords_file(
        input_file_name='train.csv',
        output_file_name="train_my.tfrecords",
        example_fn=functools.partial(create_example_train, vocab=vocab))