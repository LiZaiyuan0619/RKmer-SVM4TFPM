import pandas as pd
import requests
import re
import random
import time
from requests.adapters import HTTPAdapter
from tqdm import tqdm


class Get_Seq(object):
    def __init__(self, uniprot_id):
        self.uniprot_id = uniprot_id

    def __parse_xml_page(self, content_xml):
        patt_seq = re.compile('<sequence length[\w\s="-]+>([A-Z]+)</sequence>', re.MULTILINE)
        match_seq = patt_seq.findall(content_xml)
        return match_seq

    def get_xml_page(self):
        s = requests.Session()
        s.mount('https://', HTTPAdapter(max_retries=2))
        headers = {"User-Agent": "Mozilla/5.0"}
        response = s.get('https://rest.uniprot.org/uniprotkb/' + self.uniprot_id + '.xml', headers=headers)
        content_xml = response.text
        sequ = self.__parse_xml_page(content_xml=content_xml)
        return sequ


def fetch_protein_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching Protein Data"):
        uniprot_id = row['converted_alias'].strip()  # 删除可能的前后空格
        try:
            protein_fetcher = Get_Seq(uniprot_id)
            sequ = protein_fetcher.get_xml_page()
            row_data = row.tolist() + [sequ[0] if sequ else 'Error']  # 将序列添加到原行数据中
            results.append(row_data)
        except Exception as e:
            print(f"Error processing {uniprot_id}: {e}")
            results.append(row.tolist() + ['Error'])

    # 保存结果到新的CSV文件，包括原有列和序列信息
    columns = df.columns.tolist() + ['Sequence']
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# 使用示例
input_csv = 'D:/ML/Lastest/iTFPM-RGDC-main/all_uniprot_ids.csv'
output_csv = './All_Human_TFs.csv'
fetch_protein_data(input_csv, output_csv)
