import dill
import hypernetx as hnx
import numpy as np

# Construct EHR hypergraph (visit layer) from given records
def get_ehr_hypergraph(records, output_file):
    # Initialize hyperedge collections
    hyperedges = {
        "diagnosis": [],
        "procedure": [],
        "medication": []
    }

    # Initialize counters
    edge_counter = {
        "diagnosis": 1,
        "procedure": 1,
        "medication": 1
    }

    # Iterate through each patient record
    for patient in records:
        for adm in patient:
            # Retrieve diagnosis, procedure, and medication codes for each visit
            diag_set = adm[0]
            proc_set = adm[1]
            med_set = adm[2]

            # Create hyperedges and name them using counters
            hyperedges["diagnosis"].append((f"diag_{edge_counter['diagnosis']}", diag_set))
            hyperedges["procedure"].append((f"proc_{edge_counter['procedure']}", proc_set))
            hyperedges["medication"].append((f"med_{edge_counter['medication']}", med_set))

            # Update counters
            edge_counter["diagnosis"] += 1
            edge_counter["procedure"] += 1
            edge_counter["medication"] += 1

    # Create the hypergraph
    hypergraph = hnx.Hypergraph({
        edge[0]: edge[1] for edge_list in hyperedges.values() for edge in edge_list
    })

    # Convert the hypergraph to a serializable format
    hypergraph_dict = {
        "nodes": list(hypergraph.nodes),
        "edges": {key: list(value) for key, value in hypergraph.incidence_dict.items()}
    }

    # Serialize the dictionary
    with open(output_file, 'wb') as f:
        dill.dump(hypergraph_dict, f)

    return hypergraph

# Build a historical hypergraph adjacency matrix from records
def build_history_hypergraph_matrix(records, diag_voc_size, pro_voc_size, med_voc_size):
    total_voc_size = diag_voc_size + pro_voc_size + med_voc_size
    history_hypergraph_matrix = np.zeros((total_voc_size, total_voc_size))

    for patient in records:
        for adm in patient[:-1]:  # Ignore the last visit record, as it represents the current visit
            diag_nodes = adm[0]
            pro_nodes = adm[1]
            med_nodes = adm[2]

            # Hyperedges between diagnosis nodes
            for i in range(len(diag_nodes)):
                for j in range(i + 1, len(diag_nodes)):
                    history_hypergraph_matrix[diag_nodes[i], diag_nodes[j]] = 1
                    history_hypergraph_matrix[diag_nodes[j], diag_nodes[i]] = 1

            # Hyperedges between procedure nodes
            for i in range(len(pro_nodes)):
                for j in range(i + 1, len(pro_nodes)):
                    history_hypergraph_matrix[pro_nodes[i], pro_nodes[j]] = 1
                    history_hypergraph_matrix[pro_nodes[j], pro_nodes[i]] = 1

            # Hyperedges between medication nodes
            for i in range(len(med_nodes)):
                for j in range(i + 1, len(med_nodes)):
                    history_hypergraph_matrix[med_nodes[i], med_nodes[j]] = 1
                    history_hypergraph_matrix[med_nodes[j], med_nodes[i]] = 1

            # Hyperedges between diagnosis, procedure, and medication nodes
            for i in range(len(diag_nodes)):
                for j in range(len(pro_nodes)):
                    history_hypergraph_matrix[diag_nodes[i], pro_nodes[j]] = 1
                    history_hypergraph_matrix[pro_nodes[j], diag_nodes[i]] = 1

            for i in range(len(diag_nodes)):
                for j in range(len(med_nodes)):
                    history_hypergraph_matrix[diag_nodes[i], med_nodes[j]] = 1
                    history_hypergraph_matrix[med_nodes[j], diag_nodes[i]] = 1

            for i in range(len(pro_nodes)):
                for j in range(len(med_nodes)):
                    history_hypergraph_matrix[pro_nodes[i], med_nodes[j]] = 1
                    history_hypergraph_matrix[med_nodes[j], pro_nodes[i]] = 1

    return history_hypergraph_matrix

if __name__ == "__main__":
    # Define input and output file paths
    ehr_sequence_file = "./output/records_final.pkl"
    ehr_hypergraph_file = "./output/ehr_hypergraph_final.pkl"
    history_hypergraph_file = "./output/history_hypergraph_final.pkl"

    # Load EHR sequence data
    with open(ehr_sequence_file, 'rb') as f:
        records = dill.load(f)

    # Generate and save the EHR hypergraph
    get_ehr_hypergraph(records, ehr_hypergraph_file)
    print("EHR hypergraph has been generated and saved to", ehr_hypergraph_file)

    # Load vocabularies
    voc_path = "./output/voc_final.pkl"
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    diag_voc_size = len(diag_voc.idx2word)
    pro_voc_size = len(pro_voc.idx2word)
    med_voc_size = len(med_voc.idx2word)

    # Generate and save the historical hypergraph matrix
    history_hypergraph_matrix = build_history_hypergraph_matrix(records, diag_voc_size, pro_voc_size, med_voc_size)
    with open(history_hypergraph_file, 'wb') as f:
        dill.dump(history_hypergraph_matrix, f)

    print("The historical hypergraph matrix has been generated and saved as", history_hypergraph_file)
