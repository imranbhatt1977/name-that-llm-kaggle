# src/features.py
import math
import re
import numpy as np
import pandas as pd

FALLBACK_STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
}

STOP_WORDS = FALLBACK_STOP_WORDS

DISCOURSE_MARKERS = {
    'moreover', 'furthermore', 'ultimately', 'thus', 'therefore', 'consequently',
    'nonetheless', 'nevertheless', 'meanwhile', 'instead', 'however', 'yet',
    'finally', 'specifically', 'essentially', 'intriguingly', 'notably', 'crucially',
    'conversely', 'similarly', 'accordingly', 'overall', 'hence', 'indeed'
}

TRACKED_FUNCTION_WORDS = (
    'the', 'of', 'and', 'to', 'in', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'that', 'this', 'these', 'those', 'it', 'its', 'their', 'they', 'we', 'our', 'you', 'your',
    'for', 'with', 'on', 'by', 'at', 'from', 'as', 'into', 'through', 'across', 'within',
    'without', 'between', 'under', 'beneath', 'beyond', 'upon', 'around', 'against', 'amid',
    'while', 'where', 'when', 'though', 'although', 'because', 'if', 'unless', 'until',
    'not', 'no', 'nor', 'never', 'nothing', 'neither', 'but', 'or', 'yet', 'so', 'than',
    'rather', 'instead', 'merely', 'simply', 'only', 'even', 'still', 'already', 'also',
    'too', 'very', 'more', 'most', 'less', 'least', 'each', 'every', 'all', 'both', 'some',
    'any', 'many', 'few', 'several', 'own', 'same', 'such', 'other', 'another',
    'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
    'have', 'has', 'had', 'do', 'does', 'did', 'make', 'makes', 'made',
    'turn', 'turns', 'turned', 'turning', 'become', 'becomes', 'became', 'becoming',
    'remain', 'remains', 'remained', 'leave', 'leaves', 'leaving', 'left',
    'like', 'how', 'what', 'who', 'whom', 'whose', 'which', 'why', 'here', 'there', 'then', 'now'
)

LLM_STYLE_WORDS = (
    'quiet', 'quietly', 'silent', 'silently', 'silence', 'ancient', 'vast', 'invisible',
    'delicate', 'profound', 'profoundly', 'relentless', 'relentlessly', 'fragile', 'eternal',
    'slow', 'slowly', 'sudden', 'suddenly', 'gradually', 'violently', 'effortlessly',
    'perpetually', 'simultaneously', 'literally', 'effectively', 'fundamentally', 'ultimately',
    'precisely', 'exact', 'entire', 'single', 'tiny', 'immense', 'colossal', 'microscopic',
    'infinite', 'endless', 'deep', 'deeply', 'dark', 'darkness', 'light', 'shadow', 'shadows',
    'whisper', 'whispers', 'echo', 'echoes', 'rhythm', 'pulse', 'breath', 'breathing',
    'memory', 'memories', 'time', 'space', 'world', 'universe', 'cosmos', 'cosmic',
    'human', 'humanity', 'life', 'living', 'mind', 'brain', 'body', 'nature', 'system', 'systems',
    'pattern', 'patterns', 'structure', 'structures', 'network', 'networks', 'process',
    'balance', 'equilibrium', 'chaos', 'order', 'energy', 'force', 'forces', 'power'
)

KEEP_SKELETON_VOCAB = STOP_WORDS | DISCOURSE_MARKERS | set(TRACKED_FUNCTION_WORDS) | set(LLM_STYLE_WORDS)

SUFFIXES = ('ingly', 'edly', 'fully', 'lessly', 'ously', 'ively', 'ally', 'tion', 'sion',
            'ment', 'ness', 'ity', 'ance', 'ence', 'ship', 'hood', 'ward', 'wards',
            'able', 'ible', 'less', 'ious', 'eous', 'ous', 'ive', 'ish', 'ian', 'ic', 'al',
            'ing', 'ed', 'er', 'est', 'ly', 'es', 's')

PREFIXES = ('hyper', 'ultra', 'super', 'inter', 'trans', 'under', 'over', 'counter', 'multi',
            'semi', 'anti', 'non', 'pre', 'post', 'sub', 'mis', 'out', 'un', 're', 'in', 'im', 'dis', 'de')

SPATIAL_OPENERS = {'in', 'beneath', 'deep', 'within', 'across', 'under', 'from', 'through', 'at', 'on', 'beyond', 'behind', 'inside', 'outside', 'above', 'around', 'amid', 'along'}
TEMPORAL_OPENERS = {'when', 'as', 'before', 'after', 'once', 'every', 'each', 'whenever', 'while', 'during', 'until', 'since', 'by'}
IMPERATIVE_OPENERS = {'imagine', 'consider', 'picture', 'step', 'watch', 'listen', 'zoom', 'take', 'think', 'suppose', 'look', 'let'}

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
CLEAN_WORD_RE = re.compile(r"[a-zA-Z]+")
HYPHENATED_WORD_RE = re.compile(r"\b[A-Za-z]+-[A-Za-z]+(?:-[A-Za-z]+)*\b")
PARTICIPLE_CLAUSE_RE = re.compile(r",\s+[a-z]+ing\b")
CONTRACTION_RE = re.compile(r"\b[A-Za-z]+['’](?:t|s|re|ve|ll|d|m)\b")
SKELETON_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\w\s]|\s+")


def _word_to_skeleton(word: str) -> str:
    w_low = word.lower()
    if w_low in KEEP_SKELETON_VOCAB:
        return word if word[0].isupper() else w_low

    cap = "C" if word[0].isupper() else "w"
    length = len(w_low)
    len_bucket = "S" if length <= 4 else ("M" if length <= 7 else "L")

    for suf in SUFFIXES:
        if length > len(suf) + 2 and w_low.endswith(suf):
            return f"{cap}_{len_bucket}_{suf}"
    return f"{cap}_{len_bucket}"


def to_syntactic_skeleton(text: str) -> str:
    """Converts text into a topic-masked syntactic and morphological skeleton."""
    text_str = str(text) if text is not None else ""
    tokens = SKELETON_TOKEN_RE.findall(text_str)
    out = []
    for tok in tokens:
        if tok[0].isalpha():
            out.append(_word_to_skeleton(tok))
        elif tok[0].isdigit():
            out.append("<NUM>")
        else:
            out.append(tok)
    return "".join(out)


def extract_stylistic_features(text):
    """Extracts comprehensive stylistic, structural, lexical, and typographic features from a text passage."""
    text_str = str(text) if text is not None else ""
    char_count = max(len(text_str), 1)
    raw_words = text_str.split()
    word_count = max(len(raw_words), 1)
    scale = 100.0 / word_count

    alpha_words = CLEAN_WORD_RE.findall(text_str)
    alpha_lower = [w.lower() for w in alpha_words] if alpha_words else [w.lower() for w in raw_words]
    n_alpha = max(len(alpha_lower), 1)

    # 1. Word-level length metrics & quantiles
    word_lengths = np.array([len(w) for w in alpha_words] if alpha_words else [len(w) for w in raw_words], dtype=float)
    if len(word_lengths) == 0:
        word_lengths = np.array([0.0])

    avg_word_length = float(np.mean(word_lengths))
    std_word_length = float(np.std(word_lengths))
    q25_wl, q50_wl, q75_wl, q90_wl = np.percentile(word_lengths, [25, 50, 75, 90])
    long_word_ratio = float(np.mean(word_lengths >= 8))
    very_long_word_ratio = float(np.mean(word_lengths >= 10))
    short_word_ratio = float(np.mean(word_lengths <= 3))

    # Exact word length distribution (1 to 10+)
    wl_ratios = {}
    for k in range(1, 10):
        wl_ratios[f'wl_{k}_ratio'] = float(np.mean(word_lengths == k))
    wl_ratios['wl_10plus_ratio'] = very_long_word_ratio

    # 2. Sentence & rhythm cadence (Burstiness)
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text_str.strip()) if s.strip()]
    sentence_count = max(len(sentences), 1)
    sent_word_counts = np.array([len(s.split()) for s in sentences] if sentences else [word_count], dtype=float)
    avg_sentence_length = float(np.mean(sent_word_counts))
    std_sentence_length = float(np.std(sent_word_counts))
    min_sentence_length = float(np.min(sent_word_counts))
    max_sentence_length = float(np.max(sent_word_counts))
    sentence_length_cv = std_sentence_length / max(avg_sentence_length, 1.0)
    first_sentence_words = float(sent_word_counts[0])
    last_sentence_words = float(sent_word_counts[-1])
    first_to_avg_sentence_ratio = first_sentence_words / max(avg_sentence_length, 1.0)
    last_to_avg_sentence_ratio = last_sentence_words / max(avg_sentence_length, 1.0)

    if len(sent_word_counts) >= 2:
        x_idx = np.linspace(0.0, 1.0, len(sent_word_counts))
        sent_length_slope = float(np.polyfit(x_idx, sent_word_counts, 1)[0])
        sent_diff_mean = float(np.mean(np.abs(np.diff(sent_word_counts))))
    else:
        sent_length_slope = 0.0
        sent_diff_mean = 0.0

    paragraphs = [p for p in text_str.split('\n') if p.strip()]
    paragraph_count = len(paragraphs)

    # 3. Lexical & Quantitative Authorship Attribution Diversity Metrics
    word_freq = {}
    for w in alpha_lower:
        word_freq[w] = word_freq.get(w, 0) + 1
    unique_count = max(len(word_freq), 1)
    vocab_richness = unique_count / n_alpha
    root_ttr = unique_count / math.sqrt(n_alpha)
    log_ttr = math.log(unique_count) / max(math.log(max(n_alpha, 2)), 1e-6)

    hapax_count = sum(1 for c in word_freq.values() if c == 1)
    dis_count = sum(1 for c in word_freq.values() if c == 2)
    hapax_ratio = hapax_count / n_alpha
    dis_legomena_ratio = dis_count / n_alpha
    sichel_s = dis_count / unique_count
    honore_r = 100.0 * math.log(n_alpha) / max(1.0 - (hapax_count / unique_count), 0.05)

    # Yule's K characteristic (length-invariant vocabulary repetition metric)
    freq_spectrum = {}
    for c in word_freq.values():
        freq_spectrum[c] = freq_spectrum.get(c, 0) + 1
    s2 = sum((i * i) * vi for i, vi in freq_spectrum.items())
    yules_k = 10000.0 * (s2 - n_alpha) / max(n_alpha * n_alpha, 1)

    # 4. Stopword Ratio
    stopword_count = sum(1 for w in alpha_lower if w in STOP_WORDS)
    stopword_ratio = stopword_count / n_alpha

    # 5. Micro-Punctuation & Typography Signatures (per 100 words)
    comma_count = text_str.count(',')
    period_count = text_str.count('.')
    colon_count = text_str.count(':')
    semicolon_count = text_str.count(';')
    question_count = text_str.count('?')
    exclamation_count = text_str.count('!')

    em_dash_unicode = text_str.count('—')
    en_dash_unicode = text_str.count('–')
    double_hyphen = text_str.count('--')
    spaced_em_dash = text_str.count(' — ')
    spaced_en_dash = text_str.count(' – ')
    spaced_hyphen = text_str.count(' - ')
    spaced_dash = spaced_em_dash + spaced_en_dash + spaced_hyphen
    unspaced_em_dash = em_dash_unicode - spaced_em_dash
    em_dash_count = em_dash_unicode + double_hyphen

    hyphenated_words = len(HYPHENATED_WORD_RE.findall(text_str))
    paren_count = text_str.count('(') + text_str.count(')')

    curly_apostrophe = text_str.count('’') + text_str.count('‘')
    straight_apostrophe = text_str.count("'")
    curly_quote = text_str.count('“') + text_str.count('”')
    straight_quote = text_str.count('"')
    total_quotes = curly_quote + straight_quote
    total_apostrophes = curly_apostrophe + straight_apostrophe

    uses_curly_typography = 1.0 if (curly_apostrophe + curly_quote) > 0 else 0.0
    uses_straight_typography = 1.0 if (straight_apostrophe + straight_quote) > 0 else 0.0
    curly_to_total_punct_ratio = (curly_apostrophe + curly_quote) / max(total_quotes + total_apostrophes, 1)

    # Specific possessive vs contraction typography
    curly_possessive_s = text_str.count("’s") + text_str.count("s’ ")
    straight_possessive_s = text_str.count("'s") + text_str.count("s' ")

    # Clause connector patterns (comma + conjunction)
    text_low = text_str.lower()
    comma_and = text_low.count(', and ')
    comma_but = text_low.count(', but ')
    comma_yet = text_low.count(', yet ')
    comma_while = text_low.count(', while ')
    comma_where = text_low.count(', where ')
    comma_which = text_low.count(', which ')
    comma_making = text_low.count(', making ') + text_low.count(', turning ') + text_low.count(', leaving ') + text_low.count(', creating ')

    # 6. Character-level ratios
    uppercase_count = sum(1 for c in text_str if c.isupper())
    uppercase_ratio = uppercase_count / char_count
    non_initial_caps = max(uppercase_count - sentence_count, 0)
    non_initial_cap_density = non_initial_caps * scale
    digit_count = sum(1 for c in text_str if c.isdigit())
    digit_density = digit_count * scale

    # 7. Morphological Suffix & Prefix Densities
    adverb_ly_count = sum(1 for w in alpha_lower if len(w) > 4 and w.endswith('ly'))
    ing_word_count = sum(1 for w in alpha_lower if len(w) > 5 and w.endswith('ing'))
    ed_word_count = sum(1 for w in alpha_lower if len(w) > 4 and w.endswith('ed'))
    tion_word_count = sum(1 for w in alpha_lower if len(w) > 5 and (w.endswith('tion') or w.endswith('sion')))
    ness_ment_ity_count = sum(1 for w in alpha_lower if len(w) > 5 and (w.endswith('ness') or w.endswith('ment') or w.endswith('ity')))
    ous_ive_al_count = sum(1 for w in alpha_lower if len(w) > 5 and (w.endswith('ous') or w.endswith('ive') or w.endswith('al') or w.endswith('ic')))
    able_less_ful_count = sum(1 for w in alpha_lower if len(w) > 5 and (w.endswith('able') or w.endswith('ible') or w.endswith('less') or w.endswith('ful')))

    participle_clause_count = len(PARTICIPLE_CLAUSE_RE.findall(text_str))
    discourse_marker_count = sum(1 for w in alpha_lower if w in DISCOURSE_MARKERS)
    contraction_count = len(CONTRACTION_RE.findall(text_str))

    # Sentence-initial patterns
    sent_initial_transitions = 0
    sent_initial_the_a = 0
    sent_initial_spatial = 0
    sent_initial_temporal = 0
    sent_initial_ing = 0
    for s in sentences:
        first_tok = CLEAN_WORD_RE.findall(s[:30])
        if first_tok:
            fw0 = first_tok[0].lower()
            if fw0 in DISCOURSE_MARKERS:
                sent_initial_transitions += 1
            if fw0 in ('the', 'a', 'an'):
                sent_initial_the_a += 1
            if fw0 in SPATIAL_OPENERS:
                sent_initial_spatial += 1
            if fw0 in TEMPORAL_OPENERS:
                sent_initial_temporal += 1
            if len(fw0) > 5 and fw0.endswith('ing'):
                sent_initial_ing += 1

    first_word = alpha_lower[0] if alpha_lower else ""

    features = {
        'char_length': len(text_str),
        'word_count': len(raw_words),
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'avg_word_length': avg_word_length,
        'std_word_length': std_word_length,
        'q25_word_length': float(q25_wl),
        'q50_word_length': float(q50_wl),
        'q75_word_length': float(q75_wl),
        'q90_word_length': float(q90_wl),
        'long_word_ratio': long_word_ratio,
        'very_long_word_ratio': very_long_word_ratio,
        'short_word_ratio': short_word_ratio,
        'avg_sentence_length': avg_sentence_length,
        'std_sentence_length': std_sentence_length,
        'min_sentence_length': min_sentence_length,
        'max_sentence_length': max_sentence_length,
        'sentence_length_cv': sentence_length_cv,
        'first_sentence_words': first_sentence_words,
        'last_sentence_words': last_sentence_words,
        'first_to_avg_sentence_ratio': first_to_avg_sentence_ratio,
        'last_to_avg_sentence_ratio': last_to_avg_sentence_ratio,
        'sent_length_slope': sent_length_slope,
        'sent_diff_mean': sent_diff_mean,
        'vocab_richness': vocab_richness,
        'root_ttr': root_ttr,
        'log_ttr': log_ttr,
        'hapax_ratio': hapax_ratio,
        'dis_legomena_ratio': dis_legomena_ratio,
        'sichel_s': sichel_s,
        'honore_r': honore_r,
        'yules_k': yules_k,
        'stopword_ratio': stopword_ratio,
        'comma_density': comma_count * scale,
        'period_density': period_count * scale,
        'colon_density': colon_count * scale,
        'semicolon_density': semicolon_count * scale,
        'question_density': question_count * scale,
        'exclamation_density': exclamation_count * scale,
        'em_dash_density': em_dash_count * scale,
        'en_dash_density': en_dash_unicode * scale,
        'spaced_dash_density': spaced_dash * scale,
        'spaced_em_dash_density': spaced_em_dash * scale,
        'spaced_en_dash_density': spaced_en_dash * scale,
        'spaced_hyphen_density': spaced_hyphen * scale,
        'unspaced_em_dash_density': unspaced_em_dash * scale,
        'hyphenated_word_density': hyphenated_words * scale,
        'paren_density': paren_count * scale,
        'quote_density': total_quotes * scale,
        'curly_apostrophe_density': curly_apostrophe * scale,
        'straight_apostrophe_density': straight_apostrophe * scale,
        'curly_quote_density': curly_quote * scale,
        'straight_quote_density': straight_quote * scale,
        'uses_curly_typography': uses_curly_typography,
        'uses_straight_typography': uses_straight_typography,
        'curly_to_total_punct_ratio': curly_to_total_punct_ratio,
        'curly_possessive_density': curly_possessive_s * scale,
        'straight_possessive_density': straight_possessive_s * scale,
        'comma_and_density': comma_and * scale,
        'comma_but_density': comma_but * scale,
        'comma_yet_density': comma_yet * scale,
        'comma_while_density': comma_while * scale,
        'comma_where_density': comma_where * scale,
        'comma_which_density': comma_which * scale,
        'comma_participle_verb_density': comma_making * scale,
        'uppercase_ratio': uppercase_ratio,
        'non_initial_cap_density': non_initial_cap_density,
        'digit_density': digit_density,
        'adverb_ly_density': adverb_ly_count * scale,
        'ing_word_density': ing_word_count * scale,
        'ed_word_density': ed_word_count * scale,
        'tion_word_density': tion_word_count * scale,
        'ness_ment_ity_density': ness_ment_ity_count * scale,
        'ous_ive_al_density': ous_ive_al_count * scale,
        'able_less_ful_density': able_less_ful_count * scale,
        'participle_clause_density': participle_clause_count * scale,
        'discourse_marker_density': discourse_marker_count * scale,
        'contraction_density': contraction_count * scale,
        'sentence_initial_transition_ratio': sent_initial_transitions / sentence_count,
        'sentence_initial_the_a_ratio': sent_initial_the_a / sentence_count,
        'sentence_initial_spatial_ratio': sent_initial_spatial / sentence_count,
        'sentence_initial_temporal_ratio': sent_initial_temporal / sentence_count,
        'sentence_initial_ing_ratio': sent_initial_ing / sentence_count,
        'opens_with_spatial': 1.0 if first_word in SPATIAL_OPENERS else 0.0,
        'opens_with_temporal': 1.0 if first_word in TEMPORAL_OPENERS else 0.0,
        'opens_with_imperative': 1.0 if first_word in IMPERATIVE_OPENERS else 0.0,
        'opens_with_article': 1.0 if first_word in ('the', 'a', 'an') else 0.0,
    }

    features.update(wl_ratios)

    for fw in TRACKED_FUNCTION_WORDS:
        features[f'fw_{fw}_density'] = word_freq.get(fw, 0) * scale

    return features


def process_dataframe_features(df):
    """Applies stylistic feature extraction across an entire dataframe."""
    feature_list = [extract_stylistic_features(t) for t in df['text']]
    return pd.DataFrame(feature_list, index=df.index)
