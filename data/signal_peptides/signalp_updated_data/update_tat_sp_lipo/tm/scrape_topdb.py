#parse topdb
import re
import pandas as pd
from tqdm import tqdm


def structdf_to_string(df):
    '''struct df to pseudo-uniprot string feature descriptor'''
    outstring = ''
    for idx, row in df.iterrows():
        stringified = f"{row['Localisation']}:{row['Begin']}..{row['End']}; "
        outstring = outstring + stringified

    return outstring

def string_to_structdf(string):
    '''pseudo-uniprot string feature descriptor to struct df'''
    features = string.split('; ')[:-1] #last one is empty
    outlist = []
    for feature in features:
        typ, borders = feature.split(':')
        start,end = borders.split('..')
        featuredict = {'Localisation': typ, 'Begin':int(start), 'End':int(end)}
        outlist.append(featuredict)

    return pd.DataFrame.from_dict(outlist)

requery=False
if requery ==True:
    import requests
    from bs4 import BeautifulSoup

    # 1. Get list of all IDs in TOPDB
    data  ={ 'm': 'search',
    'form': 1,
    'exptype': 'any',
    'expsubtype': 'any',
    'strtype': 'any',
    'numtm': 0,
    'reli_rel': 'any',
    'reli': 'any',
    'organism': 'any'
        }
    r = requests.post('http://topdb.enzim.hu/?m=search', data = data)


    # 2. Load page for each ID

    raw_pages = []
    for topid in tqdm(ids):
        r = requests.get(f'http://topdb.enzim.hu/?m=show&id={topid}')
        raw_pages.append(r)


    # 3. Parse pages

    uid = []
    tm = []
    sp = []
    exp = []
    struct_list = []
    for raw in tqdm(raw_pages):
        #r = requests.get(f'http://topdb.enzim.hu/?m=show&id={topid}')
        uniprot_name = re.findall(r'<td>UniProt</td><td>([a-zA-Z0-9_]+)</td>', raw.text)#r.text)

        soup = BeautifulSoup(raw.text, 'html.parser')
        content_tables = soup.find_all('table')[2].find_all('table')[1].find_all('table')[8].find_all('table')
        loc_table = content_tables[3]

        structure = pd.read_html(loc_table.prettify(), header=[0])[0]
        #structure = structure.loc[structure['Begin'].astype(int)<71]

        features = list(structure['Localisation'])

        has_sp = 'Signal' in features
        has_tm = any(['membrane' in x for x in features])
        
        exps = pd.read_html(content_tables[4].prettify(), header=[0])[0]['Experiment'].unique()
        exps =  list(exps)
        
        uid.append(uniprot_name)
        tm.append(has_tm)
        sp.append(has_sp)
        exp.append(exps)
        struct_list.append(structdf_to_string(structure))



    # 3. Write to df

    df = pd.DataFrame.from_dict({'acc': uid, 'sig': sp, 'topdb': ids, 'tm':tm, 'exp': exp, 'struct': struct_list})
    df['acc'] = df['acc'].apply(lambda x: x[0])
    df.to_csv('TOPDB_uniprot_acc_dump.tsv', sep='\t')


df = pd.read_csv('TOPDB_uniprot_acc_dump.tsv', sep='\t')


# 4. Filter for TM region in first 70 positions

membrane_names = ['Bacterial inner membrane', 'Bacterial outer membrane', 'Plasma membrane']
# convert each struct string back to df and filter for membrane localisations with start before 70 aas
for idx, row in df.iterrows():
    feature_df = string_to_structdf(row['struct'])
    feature_df = feature_df.loc[feature_df['Begin'] < 71]
    feature_df['Localisation']
    has_tm = any(feature_df['Localisation'].apply(lambda x: x in membrane_names))
    df.loc[idx,'tm'] = has_tm

df = df.loc[df['tm']]

# 5. Filter for Uniprot names that are already in the training set
df_signalp = pd.read_csv('uniprot-yourlist_M20201116A94466D2655679D1FD8953E075198DA81282EEB.tab', sep='\t')
df_signalp = df_signalp[['Entry', 'Entry name']]
df_signalp = df_signalp.rename({'Entry name': 'acc'}, axis=1)


df_merged = df.merge(df_signalp.drop_duplicates(), on=['acc'], how='left', indicator=True)
df_merged = df_merged[df_merged['_merge'] == 'left_only']
df_merged = df_merged.drop(['Unnamed: 0', 'exp', 'Entry', '_merge'],axis=1)
df_merged =  df_merged.rename({'sig': 'has_sp', 'tm':'has_tm'}, axis=1)
df_merged.to_csv('tm_not_in_signalp5.tsv', '\t')