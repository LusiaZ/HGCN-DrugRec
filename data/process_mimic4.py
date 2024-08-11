from xml.dom.pulldom import ErrorHandler
import pandas as pd
import dill
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS

##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={"ndc": "category"}, encoding='ISO-8859-1')
    original_size = med_pd.shape[0]
    med_pd.drop(
        columns=[
            "pharmacy_id",
            "poe_id",
            "poe_seq",
            "order_provider_id",
            "stoptime",
            "drug_type",
            "formulary_drug_cd",
            "gsn",
            "prod_strength",
            "form_rx",
            "dose_val_rx",
            "dose_unit_rx",
            "form_val_disp",
            "form_unit_disp",
            "doses_per_24_hrs",
            "route",
        ],
        axis=1,
        inplace=True,
    )
    med_pd.drop(index=med_pd[med_pd["ndc"] == "0"].index, axis=0, inplace=True)
    med_pd.fillna(method="pad", inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd["starttime"] = pd.to_datetime(med_pd["starttime"], errors='coerce')
    med_pd.dropna(subset=["starttime"], inplace=True)
    med_pd.sort_values(by=["subject_id", "hadm_id", "starttime"], inplace=True)
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    print("Original medication entries:", original_size)
    print("Medication entries after processing:", med_pd.shape[0])
    return med_pd

# ATC3-to-drugname
def ATC3toDrug(med_pd):
    atc3toDrugDict = {}
    for atc3, drugname in med_pd[["ATC3", "drug"]].values:
        if atc3 in atc3toDrugDict:
            atc3toDrugDict[atc3].add(drugname)
        else:
            atc3toDrugDict[atc3] = set(drugname)
    return atc3toDrugDict

def atc3toSMILES(ATC3toDrugDict, druginfo):
    drug2smiles = {}
    atc3tosmiles = {}
    for drugname, smiles in druginfo[["name", "moldb_smiles"]].values:
        if type(smiles) == type("a"):
            drug2smiles[drugname] = smiles
    for atc3, drug in ATC3toDrugDict.items():
        temp = []
        for d in drug:
            try:
                temp.append(drug2smiles[d])
            except:
                pass
        if len(temp) > 0:
            atc3tosmiles[atc3] = temp[:3]
    return atc3tosmiles

# medication mapping
def codeMapping2atc4(med_pd):
    with open(ndc2RXCUI_file, "r") as f:
        ndc2RXCUI = eval(f.read())
    med_pd["RXCUI"] = med_pd["ndc"].map(ndc2RXCUI)
    print("NDC to RXCUI mapping completed.")
    print("Unique RXCUI codes:", med_pd["RXCUI"].unique().shape[0])
    print("RXCUI mapping missing count:", med_pd["RXCUI"].isna().sum())
    med_pd.dropna(inplace=True)
    RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
    RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
    RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)
    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
    print("RXCUI to ATC4 mapping completed.")
    print("Unique ATC4 codes before truncation:", med_pd["ATC4"].unique().shape[0])
    med_pd.drop(columns=["ndc", "RXCUI"], inplace=True)
    med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: x[:4])
    med_pd = med_pd.rename(columns={"ATC4": "ATC3"})
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    print("Unique ATC3 codes after truncation:", med_pd["ATC3"].unique().shape[0])
    return med_pd

##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=["subject_id", "hadm_id", "seq_num"], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)
    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={"icd_code": "category"})
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=["subject_id", "hadm_id", "seq_num"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)
    return pro_pd

###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):
    med_pd_key = med_pd[["subject_id", "hadm_id"]].drop_duplicates()
    diag_pd_key = diag_pd[["subject_id", "hadm_id"]].drop_duplicates()
    pro_pd_key = pro_pd[["subject_id", "hadm_id"]].drop_duplicates()

    combined_key = med_pd_key.merge(
        diag_pd_key, on=["subject_id", "hadm_id"], how="inner"
    )
    combined_key = combined_key.merge(
        pro_pd_key, on=["subject_id", "hadm_id"], how="inner"
    )
    diag_pd = diag_pd.merge(combined_key, on=["subject_id", "hadm_id"], how="inner")
    med_pd = med_pd.merge(combined_key, on=["subject_id", "hadm_id"], how="inner")
    pro_pd = pro_pd.merge(combined_key, on=["subject_id", "hadm_id"], how="inner")
    # flatten and merge
    diag_pd = (
        diag_pd.groupby(by=["subject_id", "hadm_id"])["icd_code"]
        .unique()
        .reset_index()
    )
    med_pd = med_pd.groupby(by=["subject_id", "hadm_id"])["ATC3"].unique().reset_index()
    pro_pd = (
        pro_pd.groupby(by=["subject_id", "hadm_id"])["icd_code"]
        .unique()
        .reset_index()
        .rename(columns={"icd_code": "PRO_CODE"})
    )
    med_pd["ATC3"] = med_pd["ATC3"].map(lambda x: list(x))
    pro_pd["PRO_CODE"] = pro_pd["PRO_CODE"].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=["subject_id", "hadm_id"], how="inner")
    data = data.merge(pro_pd, on=["subject_id", "hadm_id"], how="inner")
    data["ATC3_num"] = data["ATC3"].map(lambda x: len(x))
    return data

def statistics(data):
    print("#patients ", data["subject_id"].unique().shape)
    print("#clinical events ", len(data))
    diag = data["icd_code"].values
    med = data["ATC3"].values
    pro = data["PRO_CODE"].values
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    print("#diagnosis ", len(unique_diag))
    print("Unique diagnosis codes:", unique_diag)
    print("#med ", len(unique_med))
    print("Unique medication codes:", unique_med)
    print("#procedure", len(unique_pro))
    print("Unique procedure codes:", unique_pro)
    (
        avg_diag,
        avg_med,
        avg_pro,
        max_diag,
        max_med,
        max_pro,
        cnt,
        max_visit,
        avg_visit,
    ) = [0 for i in range(9)]
    for subject_id in data["subject_id"].unique():
        item_data = data[data["subject_id"] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row["icd_code"]))
            y.extend(list(row["ATC3"]))
            z.extend(list(row["PRO_CODE"]))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
    print("#avg of diagnoses ", avg_diag / cnt)
    print("#avg of medicines ", avg_med / cnt)
    print("#avg of procedures ", avg_pro / cnt)
    print("#avg of vists ", avg_visit / len(data["subject_id"].unique()))
    print("#max of diagnoses ", max_diag)
    print("#max of medicines ", max_med)
    print("#max of procedures ", max_pro)
    print("#max of visit ", max_visit)

##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    for index, row in df.iterrows():
        diag_voc.add_sentence(row["icd_code"])
        med_voc.add_sentence(row["ATC3"])
        pro_voc.add_sentence(row["PRO_CODE"])
    dill.dump(
        obj={"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
        file=open(vocabulary_file, "wb"),
    )
    return diag_voc, med_voc, pro_voc

# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = []  # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df["subject_id"].unique():
        item_df = df[df["subject_id"] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row["icd_code"]])
            admission.append([pro_voc.word2idx[i] for i in row["PRO_CODE"]])
            admission.append([med_voc.word2idx[i] for i in row["ATC3"]])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open(ehr_sequence_file, "wb"))
    return records

# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):
    TOPK = 40  # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)
    with open(cid2atc6_file, "r") as f:
        for line in f:
            line_ls = line[:-1].split(",")
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])

    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect
    ddi_most_pd = (
        ddi_df.groupby(by=["Polypharmacy Side Effect", "Side Effect Name"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(
        ddi_most_pd[["Side Effect Name"]], how="inner", on=["Side Effect Name"]
    )
    ddi_df = (
        fliter_ddi_df[["STITCH 1", "STITCH 2"]].drop_duplicates().reset_index(drop=True)
    )
    # weighted ehr adj
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open(ehr_adjacency_file, "wb"))
    # ddi adj
    ddi_adj = np.zeros((med_voc_size, med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row["STITCH 1"]
        cid2 = row["STITCH 2"]
        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:

                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open(ddi_adjacency_file, "wb"))
    return ddi_adj


def get_ddi_mask(atc42SMLES, med_voc):
    fraction = []
    for k, v in med_voc.idx2word.items():
        tempF = set()
        for SMILES in atc42SMLES[v]:
            try:
                m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
                for frac in m:
                    tempF.add(frac)
            except:
                pass
        fraction.append(tempF)
    fracSet = []
    for i in fraction:
        fracSet += i
    fracSet = list(set(fracSet))  # set of all segments
    ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))
    for i, fracList in enumerate(fraction):
        for frac in fracList:
            ddi_matrix[i, fracSet.index(frac)] = 1
    return ddi_matrix

if __name__ == "__main__":

    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    # please change into your own MIMIC folder
    med_file = "./input/prescriptions.csv"
    diag_file = "./input/diagnoses.csv"
    procedure_file = (
        "./input/procedures.csv"
    )

    # input auxiliary files
    med_structure_file = "./output/atc32SMILES.pkl"
    RXCUI2atc4_file = "./input/RXCUI2atc4.csv"
    cid2atc6_file = "./input/drug-atc.csv"
    ndc2RXCUI_file = "./input/ndc2RXCUI.txt"
    ddi_file = "./input/drug-DDI.csv"
    drugbankinfo = "./input/drugbank_drugs_info.csv"

    # output files
    ddi_adjacency_file = "./output/ddi_A_final.pkl"
    ehr_adjacency_file = "./output/ehr_adj_final.pkl"
    ehr_sequence_file = "./output/records_final.pkl"
    vocabulary_file = "./output/voc_final.pkl"
    ddi_mask_H_file = "./output/ddi_mask_H.pkl"
    atc3toSMILES_file = "output/atc3toSMILES.pkl"

    # for med
    med_pd = med_process(med_file)
    med_pd = med_pd.reset_index(drop=True)
    med_pd = codeMapping2atc4(med_pd)

    # med to SMILES mapping
    atc3toDrug = ATC3toDrug(med_pd)
    druginfo = pd.read_csv(drugbankinfo)
    atc3toSMILES = atc3toSMILES(atc3toDrug, druginfo)
    dill.dump(atc3toSMILES, open(atc3toSMILES_file, "wb"))
    med_pd = med_pd[med_pd.ATC3.isin(atc3toSMILES.keys())]
    print("complete medication processing")

    # for diagnosis
    diag_pd = diag_process(diag_file)
    print("complete diagnosis processing")

    # for procedure
    pro_pd = procedure_process(procedure_file)
    print("complete procedure processing")

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    statistics(data)
    print("complete combining")

    # create vocab
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    print("obtain voc")

    # create ehr sequence data
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    print("obtain ehr sequence data")

    # create ddi adj matrix
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    print("obtain ddi adj matrix")

    # get ddi_mask_H
    ddi_mask_H = get_ddi_mask(atc3toSMILES, med_voc)
    dill.dump(ddi_mask_H, open(ddi_mask_H_file, "wb"))
