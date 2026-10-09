"""Training-only content EDA, split quality audits, and report-ready artifacts."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from modules.config import LABEL_NAMES
from modules.data_loader import get_eda_statistics, plot_class_distribution, plot_text_length_distribution, plot_label_cooccurrence
from modules.preprocessing import default_stopwords, word_tokens, remove_extra_whitespace


def normalized_key(text):
    return remove_extra_whitespace(text).casefold() if isinstance(text, str) else None


def audit_splits(splits):
    rows, overlaps = [], []
    for name, df in splits.items():
        keys = df.text.map(normalized_key)
        labelsets = df.labels.map(lambda x: tuple(sorted(x)))
        conflicts = pd.DataFrame({'key':keys,'labels':labelsets}).groupby('key').labels.nunique()
        rows.append({'split':name, 'rows':len(df), 'missing_text':int(df.text.isna().sum()),
            'blank_text':int(df.text.str.strip().eq('').sum()),
            'missing_id':int(df.id.isna().sum() + df.id.eq('').sum()),
            'duplicate_ids':int(df.id.duplicated().sum()),
            'exact_duplicate_excess':int(df.text.duplicated().sum()),
            'normalized_duplicate_excess':int(keys.duplicated().sum()),
            'conflicting_text_groups':int((conflicts>1).sum())})
    for a,b in combinations(splits,2):
        ka = splits[a].text.map(normalized_key)
        kb = splits[b].text.map(normalized_key)
        shared = set(ka) & set(kb)
        overlaps.append({'split_a':a,'split_b':b,
            'shared_ids':len(set(splits[a].id)&set(splits[b].id)),
            'shared_exact_texts':len(set(splits[a].text)&set(splits[b].text)),
            'shared_normalized_texts':len(shared),
            'affected_rows_a':int(ka.isin(shared).sum()), 'affected_rows_b':int(kb.isin(shared).sum())})
    return pd.DataFrame(rows), pd.DataFrame(overlaps)


def run_eda(splits, output_dir='reports'):
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    out = Path(output_dir)
    tables = out/'tables'; figures = out/'figures'
    tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    train = splits['train']
    stats = {name:get_eda_statistics(df) for name,df in splits.items()}
    (out/'eda_statistics.json').write_text(json.dumps(stats,indent=2))
    quality, overlap = audit_splits(splits)
    quality.to_csv(tables/'quality_audit.csv',index=False)
    overlap.to_csv(tables/'split_overlap.csv',index=False)
    length_rows=[]
    for split,s in stats.items():
        for unit in ('word','char'):
            length_rows.append({'split':split,'unit':unit,**s[f'{unit}_count_stats']})
    pd.DataFrame(length_rows).to_csv(tables/'length_statistics.csv',index=False)
    supports = pd.DataFrame({name:df[LABEL_NAMES].sum() for name,df in splits.items()})
    prevalence = pd.DataFrame({name:df[LABEL_NAMES].mean() for name,df in splits.items()})
    supports.to_csv(tables/'label_support.csv',index_label='emotion')
    prevalence.to_csv(tables/'label_prevalence.csv',index_label='emotion')
    drift = prevalence.sub(prevalence['train'],axis=0)*100
    drift.to_csv(tables/'label_prevalence_drift_pp.csv',index_label='emotion')
    # Label support sums exceed N because each comment may have multiple labels.
    y = train[LABEL_NAMES].to_numpy(dtype=np.int64)
    cardinality = y.sum(axis=1)
    pd.Series(cardinality).value_counts().sort_index().to_csv(tables/'labels_per_comment.csv',index_label='label_count')
    co = y.T@y
    pd.DataFrame(co,index=LABEL_NAMES,columns=LABEL_NAMES).to_csv(tables/'cooccurrence.csv',index_label='emotion')
    pairs = []
    for i,j in combinations(range(len(LABEL_NAMES)),2):
        union = co[i,i]+co[j,j]-co[i,j]
        pairs.append({'emotion_a':LABEL_NAMES[i],'emotion_b':LABEL_NAMES[j],
                      'count':int(co[i,j]),'jaccard':float(co[i,j]/union) if union else 0})
    pd.DataFrame(pairs).sort_values('count',ascending=False).to_csv(tables/'emotion_pairs.csv',index=False)
    stops = default_stopwords()
    raw_tokens = train.text.map(word_tokens)
    content = raw_tokens.map(lambda tokens:[t for t in tokens if t not in stops and t not in {'name','religion'}])
    all_tokens = Counter(t for tokens in raw_tokens for t in tokens)
    content_tokens = Counter(t for tokens in content for t in tokens)
    stopcounts = Counter({t:n for t,n in all_tokens.items() if t in stops})
    for name, counter in [('raw_word_frequencies',all_tokens),('content_word_frequencies',content_tokens),('stopword_frequencies',stopcounts)]:
        pd.DataFrame(counter.most_common(),columns=['term','count']).to_csv(tables/f'{name}.csv',index=False)
    # Adjacent bigrams are constructed before filtering: no invented phrases across removed words.
    bigrams = Counter(' '.join((a,b)) for tokens in raw_tokens for a,b in zip(tokens,tokens[1:])
                      if a not in stops and b not in stops and a not in {'name','religion'} and b not in {'name','religion'})
    pd.DataFrame(bigrams.most_common(100),columns=['bigram','count']).to_csv(tables/'adjacent_bigrams.csv',index=False)
    category_rows, vocabulary_rows=[] , []
    for i,label in enumerate(LABEL_NAMES):
        selected = content[y[:,i]==1]
        counter = Counter(t for row in selected for t in row)
        category_rows.extend({'emotion':label,'term':t,'count':n} for t,n in counter.most_common(50))
        total = sum(counter.values())
        vocabulary_rows.append({'emotion':label,'comments':int(y[:,i].sum()),'content_tokens':total,
              'unique_tokens':len(counter),'type_token_ratio':len(counter)/total if total else 0,
              'mean_raw_words':float(train.loc[y[:,i]==1,'text'].str.split().str.len().mean())})
    pd.DataFrame(category_rows).to_csv(tables/'top_words_by_emotion.csv',index=False)
    pd.DataFrame(vocabulary_rows).to_csv(tables/'vocabulary_by_emotion.csv',index=False)
    # Training only; per-class mean TF-IDF and centroid similarity, not classifier results.
    vectorizer = TfidfVectorizer(tokenizer=word_tokens,token_pattern=None,lowercase=False,
        stop_words=sorted(stops | {'name','religion'}),min_df=3,max_features=15000)
    x = vectorizer.fit_transform(train.text)
    features = vectorizer.get_feature_names_out()
    centroids = np.vstack([np.asarray(x[y[:,i]==1].mean(axis=0)).ravel() for i in range(len(LABEL_NAMES))])
    terms=[]
    for i,label in enumerate(LABEL_NAMES):
        terms.extend({'emotion':label,'term':features[j],'mean_tfidf':float(centroids[i,j])}
                     for j in np.argsort(centroids[i])[-15:][::-1])
    pd.DataFrame(terms).to_csv(tables/'top_tfidf_by_emotion.csv',index=False)
    similarity=cosine_similarity(centroids)
    pd.DataFrame(similarity,index=LABEL_NAMES,columns=LABEL_NAMES).to_csv(tables/'centroid_similarity.csv',index_label='emotion')
    train_lengths=train.text.str.split().str.len()
    q1,q3=train_lengths.quantile([.25,.75]); threshold=q3+1.5*(q3-q1)
    long_rows=train.loc[train_lengths>threshold,['id','text','labels']].copy()
    long_rows['word_count']=train_lengths[train_lengths>threshold]
    long_rows.sort_values('word_count',ascending=False).to_csv(tables/'long_comment_review.csv',index=False)
    # IQR flag is for review; valid long comments are NOT deleted.
    flags = {'url':r'https?://|www\.', 'user_mention':r'(?:/?u/|@)[\w-]+',
             'masked_entity':r'\[(?:NAME|RELIGION)\]', 'negation':r"\b(?:no|not|never|cannot)\b|n['’]t\b",
             'exclamation':r'!', 'question':r'\?', 'non_ascii':r'[^\x00-\x7F]'}
    cues = pd.DataFrame([{'cue':k,'comments':int(train.text.str.contains(v,flags=re.I,regex=True).sum())} for k,v in flags.items()])
    cues['percent']=cues.comments/len(train)*100
    cues.to_csv(tables/'text_cues.csv',index=False)
    samples=[]
    for label in LABEL_NAMES:
        for _, row in train.loc[train[label]==1].head(2).iterrows():
            samples.append({'emotion':label,'id':row.id,'text':row.text,'labels':row.labels})
    pd.DataFrame(samples).to_csv(tables/'samples_by_emotion.csv',index=False)
    def save(fig,name):
        fig.savefig(figures/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    save(plot_class_distribution(train),'label_support')
    save(plot_text_length_distribution(train),'text_lengths')
    save(plot_label_cooccurrence(train),'cooccurrence')
    fig,ax=plt.subplots(figsize=(8,4));pd.Series(cardinality).value_counts().sort_index().plot.bar(ax=ax,color='#287c9a')
    ax.set(title='Training labels per comment',xlabel='Number of labels',ylabel='Comments');fig.tight_layout();save(fig,'label_cardinality')
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,counter,title in zip(axes,[content_tokens,bigrams],['Top content words','Top adjacent content bigrams']):
        items=counter.most_common(20)[::-1]
        ax.barh([t for t,n in items],[n for t,n in items],color='#287c9a');ax.set(title=title,xlabel='Training occurrences')
    fig.tight_layout();save(fig,'word_frequencies')
    fig,ax=plt.subplots(figsize=(12,10))
    import seaborn as sns
    sns.heatmap(similarity,xticklabels=LABEL_NAMES,yticklabels=LABEL_NAMES,cmap='viridis',vmin=0,vmax=1,ax=ax)
    ax.set_title('Cosine similarity of TRAINING mean TF-IDF centroids');fig.tight_layout();save(fig,'centroid_similarity')
    fig,ax=plt.subplots(figsize=(9,7));prevalence.mul(100).plot.barh(ax=ax)
    ax.set(title='Label prevalence by official split',xlabel='Percentage of comments');fig.tight_layout();save(fig,'split_prevalence')
    summary={'train_word_tokens':sum(all_tokens.values()),'train_lexical_vocabulary':len(all_tokens),
        'content_vocabulary':len(content_tokens),'stopword_token_fraction':sum(stopcounts.values())/sum(all_tokens.values()),
        'iqr_upper_word_threshold':float(threshold),'long_comment_flags':len(long_rows),
        'max_label_prevalence_drift_pp':float(drift[['validation','test']].abs().to_numpy().max())}
    (out/'content_summary.json').write_text(json.dumps(summary,indent=2))
    return {'statistics':stats,'quality':quality,'overlap':overlap,'content_summary':summary,'supports':supports}
