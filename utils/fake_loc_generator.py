"""
Generadores de ubicaciones/Trayectorias falsas para DP de localización.

Estrategias:
- fake_loc_gen: falsos al azar fuera de las ubicaciones reales del usuario.
- plausible_loc_gen: trayectorias falsas plausibles guiadas por matriz de
  transición global y dominios de riesgo epidemiológico por día.
Mapea trayectorias falsas/verdaderas a IDs de hiper-arista usando index_seq.
"""

import os.path

import numpy as np
import random
from tqdm import tqdm

# A simple random fake location generator
def fake_loc_gen(traj_mat, seq_num):
    """Genera conjuntos de ubicaciones falsas por usuario y segmento."""
    trajs = np.split(traj_mat, seq_num, axis=1)
    real_locs = []
    fake_locs = []
    loc_set = set(np.unique(traj_mat))
    for i, traj in enumerate(trajs):
        real_locs.append({})
        fake_locs.append({})
        for uid, usr in enumerate(traj):
            r_loc = set(np.unique(usr))
            if -1 in r_loc:
                r_loc.remove(-1)
            f_loc = random.choices(list(loc_set - r_loc), k=len(r_loc))
            real_locs[i][uid] = r_loc
            fake_locs[i][uid] = f_loc
    return real_locs, fake_locs

def global_tf_mat(traj_mat):
    """Calcula matriz de transición global entre ubicaciones a partir de trayectorias."""
    trajs = traj_mat
    valid_locs = trajs[trajs >= 0]
    loc_num = valid_locs.max() + 1 if valid_locs.size else 0
    tf_mat = np.zeros((loc_num, loc_num))
    for t in range(trajs.shape[1] - 1):
        for usr in range(trajs.shape[0]):
            src, dst = trajs[usr, t], trajs[usr, t + 1]
            if src < 0 or dst < 0:
                continue  # ignora huecos (-1)
            tf_mat[src, dst] += 1
    tf_sum = tf_mat.sum(axis=1)
    tf_sum[tf_sum == 0] = 1  # evita división por cero
    tf_mat = tf_mat / tf_sum[:, np.newaxis]
    tf_mat = np.nan_to_num(tf_mat, nan=1 / loc_num)
    return tf_mat

# A function inverts a dict with list values.
# Source: https://stackoverflow.com/questions/35491223/inverting-a-dictionary-with-list-values
def invert_dict(d):
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = key
            else:
                raise ValueError
    return inverse

def eval_epi_domain(epi_risk, epi_levels=5):
    """Agrupa ubicaciones por niveles de riesgo epidemiológico por día."""
    epi_domain = {}
    for idx in range(epi_risk.shape[1]):
        epi_domain[idx] = {}
        risk = epi_risk[:, idx]
        arg_loc = np.argsort(risk)
        loc = np.array_split(arg_loc, epi_levels)
        epi_domain[idx]['dom2loc'] = {dom_id: list(dom_loc) for dom_id, dom_loc in enumerate(loc)}
        epi_domain[idx]['loc2dom'] = invert_dict(epi_domain[idx]['dom2loc'])
    return epi_domain

def plausible_loc_gen(traj_mat, seq_num, unique_len, fake_trajs_dir, epi_risk=None, index_seq=None):
    """
    Genera trayectorias falsas plausibles y las proyecta a hiper-aristas.
    - RW guiado por tf_mat y dominio de riesgo por día.
    - Persiste/lee de disco para reutilización.
    return: listas por segmento de dicts uid->lista de ids de hiper-arista (fake/real)
    """
    tf_mat = global_tf_mat(traj_mat)
    fake_traj_mat = np.full(traj_mat.shape, -1, dtype=int)
    epi_domain = eval_epi_domain(epi_risk)
    loc_set = set(np.unique(traj_mat))
    if not os.path.isfile(fake_trajs_dir):
        for time in tqdm(range(traj_mat.shape[1])):
            epi_domain = epi_domain  # TODO xxxx
            for uid in range(traj_mat.shape[0]):
                loc = traj_mat[uid, time]
                epi_time_idx = time // 48  # 48 indicates 48 half-hour each day.
                if loc == -1:
                    # si falta la ubicación, usa todas las de ese día
                    loc_epi_domain = np.concatenate(list(epi_domain[epi_time_idx]['dom2loc'].values()))
                else:
                    loc_epi_domain = epi_domain[epi_time_idx]['loc2dom'][loc]
                    loc_epi_domain = epi_domain[epi_time_idx]['dom2loc'][loc_epi_domain]
                if time == 0:
                    fake_loc = random.choice(loc_epi_domain)
                else:
                    last_loc = fake_traj_mat[uid, time-1]
                    tf_vec = tf_mat[last_loc, loc_epi_domain]
                    denom = tf_vec.sum()
                    if denom == 0:
                        fake_loc = random.choice(loc_epi_domain)
                    else:
                        tf_vec = tf_vec / denom
                        fake_loc = np.random.choice(loc_epi_domain, 1, p=tf_vec)
                fake_traj_mat[uid, time] = fake_loc
        np.save(fake_trajs_dir, fake_traj_mat)
    else:
        fake_traj_mat = np.load(fake_trajs_dir)

    unique_num = traj_mat.shape[1] // seq_num // unique_len

    fake_trajs = np.split(fake_traj_mat, seq_num, axis=1)
    real_trajs = np.split(traj_mat, seq_num, axis=1)
    fake_hyperedge_index = []  # seq_num * user_num * fake_edge_num
    real_hyperedge_index = []
    for seq_idx in tqdm(range(seq_num)):
        fake_edge_index = {}
        real_edge_index = {}
        for uid in range(traj_mat.shape[0]):
            fake_edge_index[uid] = []
            real_edge_index[uid] = []
            for t_idx in range(unique_num):
                fake_traj = fake_trajs[seq_idx][uid, t_idx*unique_len:(t_idx+1)*unique_len]
                fake_locs = [loc for loc in np.unique(fake_traj) if loc >= 0]
                real_traj = real_trajs[seq_idx][uid, t_idx * unique_len:(t_idx + 1) * unique_len]
                real_locs = [loc for loc in np.unique(real_traj) if loc >= 0]
                for fake_loc in fake_locs:
                    # NOTE: The fake locations may not exist in the spatia-temporal hyperedge.
                    if (fake_loc, t_idx) in index_seq[seq_idx]:
                        fake_edge_index[uid].append(index_seq[seq_idx][(fake_loc, t_idx)])
                for real_loc in real_locs:
                    real_edge_index[uid].append(index_seq[seq_idx][(real_loc, t_idx)])
        fake_hyperedge_index.append(fake_edge_index)
        real_hyperedge_index.append(real_edge_index)
    return fake_hyperedge_index, real_hyperedge_index
        # for uid, usr_traj in enumerate(traj):
        #     if idx == 0:
        #         fake_locs[uid][idx] =


def uni_iid(traj_mat):
    loc_set = np.unique(traj_mat)
    fake_traj_mat = np.random.choice(loc_set, size=(traj_mat.shape))
    return fake_traj_mat


def agg_iid(traj_mat):
    loc_set, cnt = np.unique(traj_mat, return_counts=True)
    cnt = cnt / cnt.sum()
    fake_traj_mat = np.random.choice(loc_set, size=(traj_mat.shape), p=cnt)
    return fake_traj_mat


def rw_agg(traj_mat):
    tf_mat = global_tf_mat(traj_mat)
    fake_traj_mat = np.full(traj_mat.shape, -1, dtype=int)
    loc_set = set(np.unique(traj_mat))
    for time in tqdm(range(traj_mat.shape[1])):
        for uid in range(traj_mat.shape[0]):
            loc_epi_domain = list(loc_set)
            if time == 0:
                fake_loc = random.choice(loc_epi_domain)
            else:
                last_loc = fake_traj_mat[uid, time - 1]
                tf_vec = tf_mat[last_loc, loc_epi_domain]
                tf_vec = tf_vec / tf_vec.sum()
                fake_loc = np.random.choice(loc_epi_domain, 1, p=tf_vec)
            fake_traj_mat[uid, time] = fake_loc
    return fake_traj_mat

# def rw_agg(traj_mat):
#     tf_mat = global_tf_mat(traj_mat)
#     fake_traj_mat = np.full(traj_mat.shape, -1, dtype=int)
#     epi_domain = {}
#     loc_set = set(np.unique(traj_mat))
#
#     for time in tqdm(range(traj_mat.shape[1])):
#         epi_domain = epi_domain  # TODO xxxx
#         for uid in range(traj_mat.shape[0]):
#             loc = traj_mat[uid, time]
#             # loc_epi_domain = epi_domain[time][loc]
#             loc_epi_domain = list(loc_set)
#             if time == 0:
#                 fake_loc = random.choice(loc_epi_domain)
#             else:
#                 last_loc = fake_traj_mat[uid, time - 1]
#                 tf_vec = tf_mat[last_loc, loc_epi_domain]
#                 tf_vec = tf_vec / tf_vec.sum()
#                 fake_loc = np.random.choice(loc_epi_domain, 1, p=tf_vec)
#             fake_traj_mat[uid, time] = fake_loc


if __name__ == "__main__":
    traj_num = 20

    data_path = '../datasets/beijing/large-filled-clustered/'
    traj_mat = np.load(data_path + "traj_mat(filled,sample).npy")
    traj_mat = traj_mat[:, :14*48]
    fake_generator = rw_agg

    if not os.path.exists(data_path + fake_generator.__name__):
        os.makedirs(data_path + fake_generator.__name__)
    # for i in tqdm(range(traj_num)):
    i = 19
    fake_mat = fake_generator(traj_mat=traj_mat)
    np.save(data_path + fake_generator.__name__ + '/fake_traj_{}.npy'.format(i), fake_mat)
