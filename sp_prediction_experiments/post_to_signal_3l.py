import pandas as pd
import numpy as np
import requests
import re
import time

df = pd.read_csv('sp_prediction_experiments/uniprot_manual_plasmodium_sps.tsv', sep='\t')

result_urls = []
for idx, sequence in enumerate(df['Sequence']):
    url = 'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/Signal3L30.php'

    headers = {'content-type': 'application/x-www-form-urlencoded'}

    payload = {}
    payload['proteinID'] = df['Entry name'][idx]
    payload['seq'] = sequence
    payload['species'] = 'Euk'
    payload['email'] = 'benchmark@pickybuys.com'

    #get the result link from the response page
    response = requests.post(url, data=payload, headers=headers)
    m = re.search(r'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/[0-9]*/[0-9]*_result.html', response.text)
    result_url = m.group(0)
    print(result_url)
    result_urls.append(result_urls)

time.sleep(60*10)

series_list = [] #gather results as pd.Series
for url in result_urls:
    result=requests.get(url)

    #parse the results table
    table =re.search(r'<table.*/table>',result.text,  flags=re.DOTALL)
    table.group(0)
    df = pd.read_html(table.group(0))[0]
    series_list.append(df.iloc[0])


complete_df = pd.DataFrame(series_list)
complete_df.to_csv('signal_3l_plasmodium.csv')

#TODO redo
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003057/20201006003057_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003059/20201006003059_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003102/20201006003102_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003104/20201006003104_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003106/20201006003106_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003108/20201006003108_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003111/20201006003111_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003113/20201006003113_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003115/20201006003115_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003117/20201006003117_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003119/20201006003119_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003122/20201006003122_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003124/20201006003124_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003126/20201006003126_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003128/20201006003128_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003131/20201006003131_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003133/20201006003133_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003135/20201006003135_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003138/20201006003138_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003140/20201006003140_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003142/20201006003142_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003144/20201006003144_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003147/20201006003147_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003149/20201006003149_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003151/20201006003151_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003153/20201006003153_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003156/20201006003156_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003158/20201006003158_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003200/20201006003200_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003202/20201006003202_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003205/20201006003205_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003207/20201006003207_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003220/20201006003220_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003221/20201006003221_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003224/20201006003224_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003229/20201006003229_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003235/20201006003235_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003240/20201006003240_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003247/20201006003247_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003248/20201006003248_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003249/20201006003249_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003250/20201006003250_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003257/20201006003257_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003258/20201006003258_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003259/20201006003259_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003300/20201006003300_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003301/20201006003301_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003303/20201006003303_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003305/20201006003305_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003307/20201006003307_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003308/20201006003308_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003309/20201006003309_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003310/20201006003310_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003311/20201006003311_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003313/20201006003313_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003314/20201006003314_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003315/20201006003315_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003317/20201006003317_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003318/20201006003318_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003319/20201006003319_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003320/20201006003320_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003321/20201006003321_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003327/20201006003327_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003329/20201006003329_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003330/20201006003330_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003331/20201006003331_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003333/20201006003333_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003334/20201006003334_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003335/20201006003335_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003336/20201006003336_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003338/20201006003338_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003339/20201006003339_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003340/20201006003340_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003344/20201006003344_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003348/20201006003348_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003352/20201006003352_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003357/20201006003357_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003401/20201006003401_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003406/20201006003406_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003410/20201006003410_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003418/20201006003418_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003426/20201006003426_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003434/20201006003434_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003445/20201006003445_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003450/20201006003450_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003458/20201006003458_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003504/20201006003504_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003507/20201006003507_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003508/20201006003508_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003509/20201006003509_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003510/20201006003510_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003511/20201006003511_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003512/20201006003512_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003513/20201006003513_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003514/20201006003514_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003516/20201006003516_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003517/20201006003517_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003518/20201006003518_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003519/20201006003519_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003522/20201006003522_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003526/20201006003526_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003530/20201006003530_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003546/20201006003546_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003553/20201006003553_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003554/20201006003554_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003555/20201006003555_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003614/20201006003614_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003619/20201006003619_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003620/20201006003620_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003621/20201006003621_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003623/20201006003623_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003630/20201006003630_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003638/20201006003638_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003644/20201006003644_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003652/20201006003652_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003716/20201006003716_result.html'
'http://www.csbio.sjtu.edu.cn/bioinf/Signal-3L/temp/20201006003717/20201006003717_result.html'