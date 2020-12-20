from django.shortcuts import render, redirect
from django.views.generic.edit import FormView
from essaygrader.models import GradeEntry

from .forms import FileFieldForm, SubmitEssayForm
from django.core.files.storage import FileSystemStorage
from django.forms import formset_factory

import cv2
import pytesseract

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import re

from nltk.corpus import stopwords

from spacy.lang.en.stop_words import STOP_WORDS

import language_check
from string import punctuation

import string

import en_core_web_sm

# kappa metric for measuring agreement of automatic to human scores
from sklearn.metrics import confusion_matrix

import gensim

wordvec_model = gensim.models.Word2Vec.load('static/wordvec_model')

nlp = en_core_web_sm.load()
stopwords = stopwords.words('english')

pytesseract.pytesseract.tesseract_cmd = r'H:\Tesseract\tesseract.exe'
essay_sets = pd.read_pickle('static/training_features.pkl')
all_features = [
    'word_count',
    'corrections',
    'similarity',
    'token_count',
    'unique_token_count',
    'nostop_count',
    'sent_count',
    'ner_count',
    'comma',
    'question',
    'exclamation',
    'quotation',
    'organization',
    'caps',
    'person',
    'location',
    'money',
    'time',
    'date',
    'percent',
    'noun',
    'adj',
    'pron',
    'verb',
    'cconj',
    'adv',
    'det',
    'propn',
    'num',
    'part',
    'intj',
]
text_dim = 300


def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.
    In the course of calculating kappa, all items in ``y_true`` and ``y_pred`` will
    first be converted to floats and then rounded to integers.
    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.
    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.
    Parameters
    ----------
    y_true : array-like of float
        The true/actual/gold labels for the data.
    y_pred : array-like of float
        The predicted/observed labels for the data.
    weights : str or np.array, optional
        Specifies the weight matrix for the calculation.
        Options are ::
            -  None = unweighted-kappa
            -  'quadratic' = quadratic-weighted kappa
            -  'linear' = linear-weighted kappa
            -  two-dimensional numpy array = a custom matrix of
        weights. Each weight corresponds to the
        :math:`w_{ij}` values in the wikipedia description
        of how to calculate weighted Cohen's kappa.
        Defaults to None.
    allow_off_by_one : bool, optional
        If true, ratings that are off by one are counted as
        equal, and all other differences are reduced by
        one. For example, 1 and 2 will be considered to be
        equal, whereas 1 and 3 will have a difference of 1
        for when building the weights matrix.
        Defaults to False.
    Returns
    -------
    k : float
        The kappa score, or weighted kappa score.
    Raises
    ------
    AssertionError
        If ``y_true`` != ``y_pred``.
    ValueError
        If labels cannot be converted to int.
    ValueError
        If invalid weight scheme.
    """

    # Ensure that the lists are both the same length
    assert (len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError:
        raise ValueError("For kappa, the labels should be integers or strings "
                         "that can be converted to ints (E.g., '4.0' or '3').")

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, str):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k


def correct_language(df):
    """
    use language tool to correct for most spelling and grammatical errors. Also count the applied corrections.
    Using language_check python wrapper for languagetool:
    https://www.languagetool.org/dev
    """
    tool = language_check.LanguageTool('en-US')

    df['matches'] = df['essay'].apply(lambda txt: tool.check(txt))
    df['corrections'] = df.apply(lambda l: len(l['matches']), axis=1)
    df['corrected'] = df.apply(lambda l: language_check.correct(l['essay'], l['matches']), axis=1)

    return df


# Clean training_set essays before feeding them to the Word2Vec model.
punctuations = string.punctuation


# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_essays(essays, logging=False):
    texts = []
    counter = 1
    for essay in essays.corrected:
        if counter % 2000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(essays)))
        counter += 1
        essay = nlp(essay, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in essay if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# Define function to preprocess text for a word2vec model
def cleanup_essay_word2vec(essays, logging=False):
    sentences = []
    counter = 1
    for essay in essays:
        if counter % 2000 == 0 and logging:
            print("Processed %d out of %d documents" % (counter, len(essays)))
        # Disable tagger so that lemma_ of personal pronouns (I, me, etc) don't getted marked as "-PRON-"
        essay = nlp(essay, disable=['tagger'])
        # Grab lemmatized form of words and make lowercase
        essay = " ".join([tok.lemma_.lower() for tok in essay])
        # Split into sentences based on punctuation
        essay = re.split("[\.?!;] ", essay)
        # Remove commas, periods, and other punctuation (mostly commas)
        essay = [re.sub("[\.,;:!?]", "", sent) for sent in essay]
        # Split into words
        essay = [sent.split() for sent in essay]
        sentences += essay
        counter += 1
    return sentences


# Define function to create averaged word vectors given a cleaned text.
def create_average_vec(essay):
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    for word in essay.split():
        if word in wordvec_model.wv.vocab:
            average = np.add(average, wordvec_model.wv[word])
            num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average


reference_essays = {1: 161, 2: 3022, 3: 5263, 4: 5341, 5: 7209, 6: 8896, 7: 11796, 8: 12340}  # topic: essay_id

references = {}

stop_words = set(STOP_WORDS)

# generate nlp object for reference essays:
for topic, index in reference_essays.items():
    references[topic] = nlp(essay_sets.iloc[index]['essay'])


def avg_similarity(essay):
    sim = 0

    for ref in references:
        sim += nlp(essay).similarity(references[ref])

    return sim / 8


def extract_features(essay_text):
    d = {'essay': [essay_text]}
    essay_data = pd.DataFrame(data=d)

    essay_data['word_count'] = essay_data['essay'].str.strip().str.split().str.len()

    tool = language_check.LanguageTool('en-US')

    essay_data['matches'] = essay_data['essay'].apply(lambda txt: tool.check(txt))
    essay_data['corrections'] = essay_data.apply(lambda l: len(l['matches']), axis=1)
    essay_data['corrected'] = essay_data.apply(lambda l: language_check.correct(l['essay'], l['matches']), axis=1)

    sents = []
    tokens = []
    lemma = []
    pos = []
    ner = []

    stop_words = set(STOP_WORDS)
    stop_words.update(punctuation)

    nlp = en_core_web_sm.load()

    np.warnings.filterwarnings('ignore')

    for essay in nlp.pipe(essay_data['corrected']):
        tokens.append([e.text for e in essay])
        sents.append([sent.string.strip() for sent in essay.sents])
        pos.append([e.pos_ for e in essay])
        ner.append([e.text for e in essay.ents])
        lemma.append([n.lemma_ for n in essay])

    essay_data['tokens'] = tokens
    essay_data['lemma'] = lemma
    essay_data['pos'] = pos
    essay_data['sents'] = sents
    essay_data['ner'] = ner

    essay_data['similarity'] = essay_data.apply(lambda row: avg_similarity(row['essay']), axis=1)

    essay_data['token_count'] = essay_data.apply(lambda x: len(x['tokens']), axis=1)
    essay_data['unique_token_count'] = essay_data.apply(lambda x: len(set(x['tokens'])), axis=1)
    essay_data['nostop_count'] = essay_data \
        .apply(lambda x: len([token for token in x['tokens'] if token not in stop_words]), axis=1)
    essay_data['sent_count'] = essay_data.apply(lambda x: len(x['sents']), axis=1)
    essay_data['ner_count'] = essay_data.apply(lambda x: len(x['ner']), axis=1)
    essay_data['comma'] = essay_data.apply(lambda x: x['corrected'].count(','), axis=1)
    essay_data['question'] = essay_data.apply(lambda x: x['corrected'].count('?'), axis=1)
    essay_data['exclamation'] = essay_data.apply(lambda x: x['corrected'].count('!'), axis=1)
    essay_data['quotation'] = essay_data.apply(lambda x: x['corrected'].count('"') + x['corrected'].count("'"), axis=1)
    essay_data['organization'] = essay_data.apply(lambda x: x['corrected'].count(r'@ORGANIZATION'), axis=1)
    essay_data['caps'] = essay_data.apply(lambda x: x['corrected'].count(r'@CAPS'), axis=1)
    essay_data['person'] = essay_data.apply(lambda x: x['corrected'].count(r'@PERSON'), axis=1)
    essay_data['location'] = essay_data.apply(lambda x: x['corrected'].count(r'@LOCATION'), axis=1)
    essay_data['money'] = essay_data.apply(lambda x: x['corrected'].count(r'@MONEY'), axis=1)
    essay_data['time'] = essay_data.apply(lambda x: x['corrected'].count(r'@TIME'), axis=1)
    essay_data['date'] = essay_data.apply(lambda x: x['corrected'].count(r'@DATE'), axis=1)
    essay_data['percent'] = essay_data.apply(lambda x: x['corrected'].count(r'@PERCENT'), axis=1)
    essay_data['noun'] = essay_data.apply(lambda x: x['pos'].count('NOUN'), axis=1)
    essay_data['adj'] = essay_data.apply(lambda x: x['pos'].count('ADJ'), axis=1)
    essay_data['pron'] = essay_data.apply(lambda x: x['pos'].count('PRON'), axis=1)
    essay_data['verb'] = essay_data.apply(lambda x: x['pos'].count('VERB'), axis=1)
    essay_data['noun'] = essay_data.apply(lambda x: x['pos'].count('NOUN'), axis=1)
    essay_data['cconj'] = essay_data.apply(lambda x: x['pos'].count('CCONJ'), axis=1)
    essay_data['adv'] = essay_data.apply(lambda x: x['pos'].count('ADV'), axis=1)
    essay_data['det'] = essay_data.apply(lambda x: x['pos'].count('DET'), axis=1)
    essay_data['propn'] = essay_data.apply(lambda x: x['pos'].count('PROPN'), axis=1)
    essay_data['num'] = essay_data.apply(lambda x: x['pos'].count('NUM'), axis=1)
    essay_data['part'] = essay_data.apply(lambda x: x['pos'].count('PART'), axis=1)
    essay_data['intj'] = essay_data.apply(lambda x: x['pos'].count('INTJ'), axis=1)

    return essay_data


def essay_data_to_vec(essay_data):
    essay_cleaned = cleanup_essays(essay_data, logging=False)

    cleaned_vec = np.zeros((essay_cleaned.shape[0], text_dim), dtype="float32")
    for i in range(len(essay_cleaned)):
        cleaned_vec[i] = create_average_vec(essay_cleaned[i])

    return cleaned_vec


# Create your views here.
class FileFieldView(FormView):
    form_class = FileFieldForm
    template_name = 'upload.html'  # Replace with your template.
    success_url = 'essaypreview'  # Replace with your URL or reverse().

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file_field')
        file_url = []
        image_text = []
        if form.is_valid():
            fs = FileSystemStorage(location='/static/')
            for f in files:
                filename = fs.save(f.name, f)
                uploaded_file_url = fs.url(filename)
                file_url.append(uploaded_file_url)

                image = cv2.imread('/static/' + filename)

                thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

                result = cv2.GaussianBlur(thresh, (5, 5), 0)
                result = 255 - result

                data = pytesseract.image_to_string(result, lang='eng', config='--psm 6')
                image_text.append(data)

            request.session['essays'] = image_text
            request.session['urls'] = file_url
            return redirect('essaypreview')
        else:
            return self.form_invalid(form)


def essay_preview(request, *args, **kwargs):
    context = {}

    essays = request.session['essays']
    urls = request.session['urls']

    context['essays'] = essays
    context['urls'] = urls

    essay_formset = formset_factory(SubmitEssayForm, extra=len(essays))
    formset = essay_formset(request.POST or None)

    for form_i in range(len(essays)):
        formset[form_i]['essay_text'].initial = essays[form_i]

    if formset.is_valid():
        essay_data = []
        for form in formset:
            essay_data.append(form.cleaned_data)

        request.session['essay_data'] = essay_data
        return redirect('gradeessays')

    context['formset'] = formset
    context['form_urls'] = zip(formset.forms, urls)
    return render(request, "essaypreview.html", context)


def predict_score(essay_text):
    reconstructed_model = keras.models.load_model("static/sequential_30_model")

    essay = extract_features(essay_text)
    additional_essay_features = essay[all_features]

    additional_features = pd.read_pickle('static/training_features.pkl')
    additional_features = pd.concat([additional_features[all_features], additional_essay_features], ignore_index=True)

    stdscaler = StandardScaler()
    additional_features = stdscaler.fit_transform(additional_features)

    cleaned_vec = essay_data_to_vec(essay)

    all_essay_data = pd.concat([pd.DataFrame(additional_features), pd.DataFrame(cleaned_vec)], axis=1)
    score_pred = reconstructed_model.predict(all_essay_data[0:1])

    return score_pred[0][0]


def get_letter_grade(num_score):
    letter_score = 'F'

    if num_score >= 90:
        letter_score = 'A'
    elif num_score >= 80:
        letter_score = 'B'
    elif num_score >= 70:
        letter_score = 'C'
    elif num_score >= 60:
        letter_score = 'D'

    return letter_score


def grade_essays(request, *args, **kwargs):
    essays = request.session['essay_data']
    current_user = request.user

    grade_entry = []
    for data in essays:
        essay = data['essay_text']
        student = data['student_name']
        class_name = data['class_name']

        score = predict_score(str(essay))

        score = round(score * 10, 1)
        letter = get_letter_grade(score)

        grade_entry.append([student, score, letter, class_name, str(essay)])

    for entry in grade_entry:
        new_entry = GradeEntry.objects.create(
                                owner=current_user.username,
                                student_name=entry[0],
                                class_name=entry[3],
                                num_grade=entry[1],
                                letter_grade=entry[2],
                                essay=entry[4])

        new_entry.save()
        print(new_entry)

    return render(request, "gradeessays.html", {"grade_entry": grade_entry})
