import tensorflow as tf
from preprocessing.vocabulary_processor import VocabularyProcessor
import functools
from utils.IO_data import create_row_iter, create_csv_batch_iter

tf.flags.DEFINE_string("input_dir", "../data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_integer("min_word_frequency", 5, "Minimum frequency of words in the vocabulary")
tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")
tf.flags.DEFINE_integer('batch_size', 20000, 'size of the inputs batch')
tf.flags.DEFINE_boolean('zero_padding', True, 'use zero parring in the sentence transformation')
FLAGS = tf.flags.FLAGS


def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


def create_vocab(batch_input_iter, min_frequency, max_sentence_len):
    """
    Creates and returns a VocabularyProcessor object with the vocabulary
    for the input iterator.
    """
    vocab_processor = VocabularyProcessor(
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
            word = vocab_processor.vocabulary_._reverse_mapping[id]
            writer.write(word + "\n")
    print("Saved vocabulary to {}".format(path + '/' + file_name))

def create_tfrecords_file(input_file_name, output_file_name, example_fn, path='./data'):
    """
    Creates a TFRecords file for the given input data and example transofmration function
    """
    writer = tf.python_io.TFRecordWriter(path + '/' + output_file_name)
    print("Creating TFRecords file at {}...".format(path + '/' + output_file_name))
    for i, row in enumerate(create_row_iter(create_csv_batch_iter(input_file_name, batch_size=FLAGS.batch_size))):
        example_fn(row, writer)
    writer.close()
    print("Wrote to {}".format(path + '/' + output_file_name))


def transform_sentence(sequence, vocab_processor, zero_padding=True):
    """
    Maps a single sentence into the integer vocabulary. Returns a python array.
    """
    return next(vocab_processor.transform([sequence], zero_padding=zero_padding)).tolist()

def __create_example(context, utterance, label):
    '''
    Metods used to create a single sequenceExample
    :param context: context to encode
    :param utterance: utterance to encode
    :param label: label of the example, can be positive or negative
    :return: sequence example
    '''

    # The object we return
    example = tf.train.SequenceExample()
    # A non-sequential feature of our example
    example.context.feature["context_len"].int64_list.value.append(len(context))
    example.context.feature["utterance_len"].int64_list.value.append(len(utterance))
    example.context.feature["label"].int64_list.value.append(label)

    fl_context = example.feature_lists.feature_list["context"]
    fl_utterance = example.feature_lists.feature_list["utterance"]
    for c_token in context:
        fl_context.feature.add().int64_list.value.append(c_token)
    for u_token in utterance:
        fl_utterance.feature.add().int64_list.value.append(u_token)
    return example


def create_example_train(row, writer, vocab):
    """
    Creates a training example for the Ubuntu Dialog Corpus dataset.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    context, utterance, label = row
    context_transformed = transform_sentence(context, vocab, zero_padding=FLAGS.zero_padding)        # not using the padding since it is dinamic when we create the batchs
    utterance_transformed = transform_sentence(utterance, vocab, zero_padding=FLAGS.zero_padding)    # not using the padding since it is dinamic when we create the batchs

    label = int(float(label))
    example = __create_example(context_transformed, utterance_transformed, label)
    writer.write(example.SerializeToString())                                           # write the new example


def create_example_test(row, writer, vocab):
    """
    Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    context, utterance = row[:2]
    context_transformed = transform_sentence(context, vocab, zero_padding=FLAGS.zero_padding)        # not using the padding since it is dinamic when we create the batchs
    utterance_transformed = transform_sentence(utterance, vocab, zero_padding=FLAGS.zero_padding)    # not using the padding since it is dinamic when we create the batchs

    distractors = row[2:]

    pos_example = __create_example(context_transformed, utterance_transformed, 1)
    writer.write(pos_example.SerializeToString())

    for dist in distractors:
        dist_transformed = transform_sentence(dist, vocab, zero_padding=True)
        neg_example = __create_example(context_transformed, dist_transformed, 0)
        writer.write(neg_example.SerializeToString())


if __name__ == "__main__":
    print("Creating vocabulary...")
    input_batch_iter = create_csv_batch_iter('train.csv', batch_size=FLAGS.batch_size)
    vocab = create_vocab(input_batch_iter, min_frequency=FLAGS.min_word_frequency, max_sentence_len=FLAGS.max_sentence_len)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary.txt file
    write_vocabulary(vocab, "vocabulary_my.txt", path=FLAGS.input_dir)

    # Save vocab processor
    vocab.save(file_name='vocab_processor_my.bin', path=FLAGS.input_dir)

    # Create validation.tfrecords
    create_tfrecords_file(
        input_file_name='valid.csv',
        output_file_name="validation_my.tfrecords",
        example_fn=functools.partial(create_example_test, vocab=vocab),
        path=FLAGS.input_dir)

    # Create test.tfrecords
    create_tfrecords_file(
        input_file_name='test.csv',
        output_file_name="test_my.tfrecords",
        example_fn=functools.partial(create_example_test, vocab=vocab),
        path=FLAGS.input_dir
    )

    # Create train.tfrecords
    create_tfrecords_file(
        input_file_name='train.csv',
        output_file_name="train_my.tfrecords",
        example_fn=functools.partial(create_example_train, vocab=vocab),
        path=FLAGS.input_dir
    )