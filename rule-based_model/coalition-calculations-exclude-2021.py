import pandas as pd
import numpy as np
from collections import Counter
import itertools
from itertools import combinations
import math
from scipy.spatial.distance import jensenshannon
import re
import json

# -------------------------------
# Ideological spectrum
# -------------------------------

# 2D ideological map: (Left-Right, Prog-Cons) from Kieskompas
IDEOLOGY_2D_MAP = {
    "BIJ1": (-5.0, 5.0),
    "PvdD": (-5.0, 4.5),
    "GL/PvdA": (-2.8, 3.3),
    "DENK": (-3.4, 1.5),
    "SP": (-3.8, 1.0),
    "Volt": (-0.7, 4.6),
    "D66": (-0.3, 2.7),
    "CU": (-1.7, 1.0),
    "50PLUS": (-1.2, -0.2),
    "NSC": (-0.5, -0.4),
    "CDA": (1.2, -1.2),
    "SGP": (1.3, -2.1),
    "BBB": (0.5, -2.1),
    "VVD": (2.5, -1.5),
    "PVV": (0.5, -3.7),
    "FvD": (3.2, -5.0),
    "JA21": (3.8, -4.8),
    "BVNL": (5.0, -4.8)
}

# 4D ideological map: (Economic_Left_Right,Cultural_Progressive_Conservative,Globalist_Nationalist,Libertarian_Authoritarian)
IDEOLOGY_4D_MAP = {
    "50PLUS": (-0.43, 0.61, 0.36, -0.61),
    "BBB": (-0.23, 0.65, 0.62, -0.25),
    "BIJ1": (-0.1, 0.41, 0.73, -0.2),
    "CDA": (-0.38, 0.66, 0.68, -0.25),
    "CU": (-0.36, 0.7, 0.66, -0.27),
    "D66": (-0.4, 0.63, 0.65, -0.29),
    "DENK": (-0.17, 0.37, 0.72, -0.35),
    "FvD": (-0.2, 0.39, 0.64, -0.37),
    "GL/PvdA": (-0.39, 0.65, 0.56, -0.3),
    "JA21": (-0.21, 0.52, 0.64, -0.22),
    "NSC": (-0.47, 0.68, 0.64, -0.33),
    "PVV": (-0.27, 0.49, 0.57, -0.25),
    "PvdD": (-0.34, 0.83, 0.5, -0.29),
    "SGP": (-0.39, 0.54, 0.73, -0.23),
    "SP": (-0.44, 0.56, 0.52, -0.37),
    "VVD": (-0.33, 0.72, 0.63, -0.22),
    "Volt": (-0.36, 0.66, 0.62, -0.38)
}


def load_data():
    kabinetten = pd.read_csv('data/cabinets/kabinetten_schoongemaakt-no2023.csv')
    zetels_100 = pd.read_csv('data/zetelverdeling/zetel-data/tk_zetels100_1918-1956.csv')
    zetels_150 = pd.read_csv('data/zetelverdeling/zetel-data/tk_zetels150_1956-2023-no2023.csv')
    zetels = pd.concat([zetels_100, zetels_150], ignore_index=True)
    ek_50_old = pd.read_csv('data/zetelverdeling/zetel-data/ek_zetels50_1888-1956_filled.csv')
    ek_75_new = pd.read_csv('data/zetelverdeling/zetel-data/ek_zetels75_1956-2023_filled.csv')
    ek_zetels = pd.concat([ek_50_old, ek_75_new], ignore_index=True)

    kabinetten['Partijen'] = kabinetten['Partijen'].dropna().str.split(', ')

    with open("topic_vectors.json", "r") as f:
        json_ready_vectors = json.load(f)

    # Convert lists back to NumPy arrays
    topic_vectors = {k: np.array(v) for k, v in json_ready_vectors.items()}

    return kabinetten, zetels, ek_zetels, topic_vectors


# -------------------------------
# Build historical coalition frequency model
# -------------------------------
def build_coalition_frequency(kabinetten):
    coalition_counter = Counter()
    for partijen in kabinetten['Partijen'].dropna():
        for r in range(2, len(partijen) + 1):
            for combo in combinations(sorted(partijen), r):
                coalition_counter[combo] += 1
    return coalition_counter


# -------------------------------
# Ideological compatibility score (lower is better)
# -------------------------------
def ideological_distance(parties):
    points_2d = [IDEOLOGY_2D_MAP.get(p, (0.0, 0.0)) for p in parties]
    points_4d = [IDEOLOGY_4D_MAP.get(p, (0.0, 0.0, 0.0, 0.0)) for p in parties]

    if len(parties) <= 1:
        return 0.0

    # Compute average pairwise Euclidean distance in 2D
    dist_2d = [
        math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        for a, b in combinations(points_2d, 2)
    ]
    avg_2d = sum(dist_2d) / len(dist_2d)

    # Compute average pairwise Euclidean distance in 4D
    dist_4d = [
        math.sqrt(sum((a[i] - b[i])**2 for i in range(4)))
        for a, b in combinations(points_4d, 2)
    ]
    avg_4d = sum(dist_4d) / len(dist_4d)

    return (avg_2d * 0.5 + avg_4d * 0.5)




# -------------------------------
# Reduce impact of smaller parties
# -------------------------------
def scaled_seat_weight(seat_count):
    """Scale the impact of a party's seat count using a logarithmic function."""
    # We use log scale to reduce the influence of smaller parties.
    return math.log(seat_count + 1)  # +1 to avoid log(0)


# -------------------------------
# Define unrealistic combinations (only add the ones that are definitely unrealistic)
# -------------------------------
def is_unrealistic_combo(parties):
    # Add more logic if needed
    extremes = [
        ('FvD', 'Volt'),
        ('PVV', 'BIJ1'),
        ('SGP', 'BIJ1'),
        ('FvD', 'D66'),
        ('PVV', 'GL/PvdA'),
        ('PVV', 'DENK'),
        ('PVV', 'Volt'),
        ('SGP', 'Volt'),
        ('GL/PvdA', 'BBB'),
        ('PVV', 'D66'),
        ('PVV', 'CDA'),
        ('GL/PvdA', 'SGP'),
    ]
    party_set = set(parties)
    for a, b in extremes:
        if a in party_set and b in party_set:
            return True
    return False


# -------------------------------
# Define new parties
# -------------------------------
PARTY_LINEAGE = {
    "GL/PvdA": ["GL", "PvdA"],  # Merged parties
    "NSC": ["CDA"],             # NSC is a breakaway from CDA
    "JA21": ["FvD"],            # JA21 split from FvD
    # Add other mappings if necessary
}

def expand_party(party):
    """Return historical equivalents for a party (e.g., GL/PvdA -> [GL, PvdA])"""
    return PARTY_LINEAGE.get(party, [party])

def get_expanded_coalition(combo):
    """Expand a coalition to include historical equivalents"""
    expanded = set()
    for party in combo:
        expanded.update(expand_party(party))
    return expanded

def calculate_historical_score(combo, coalition_counter, seat_distribution):
    """Compute adjusted historical overlap score using lineage info and seat scaling"""
    expanded_combo = get_expanded_coalition(combo)

    score = 0
    total_weight = 0  # To keep track of the total weight for normalization
    
    for historical_coalition in coalition_counter:
        overlap = expanded_combo & set(historical_coalition)
        
        if len(overlap) >= 2:  # If there's enough overlap
            # Calculate overlap score: how much overlap, divided by the total length of the coalition
            overlap_score = len(overlap) / len(historical_coalition)
            
            # Check if it's a lineage-based match (partial weight) or direct match (full weight)
            if any(party in PARTY_LINEAGE for party in combo):
                # If it's from a different lineage, give partial weight
                overlap_score *= 0.5  # Apply 50% weight for lineage-based matches
            
            # Sum the weighted overlap score, scaled by seat count
            for party in combo:
                party_weight = scaled_seat_weight(seat_distribution[party])
                total_weight += party_weight
                score += coalition_counter[historical_coalition] * overlap_score * party_weight
    
    # Normalize the score by the total weight of the parties in the combo
    if total_weight > 0:
        score /= total_weight
    return score


# --------------------------------------------
# Get the Eerste Kamer seat distribution for a specific year
# --------------------------------------------
def get_ek_seat_distribution(ek_zetels, Jaar):
    year_data = ek_zetels[ek_zetels['Jaar'] == Jaar]
    if year_data.empty:
        return {}
    return dict(zip(year_data.columns[1:], year_data.iloc[0, 1:]))  # Skip 'Jaar'


# --------------------------------------------
# Calculate alignment score for a coalition in the Eerste Kamer
# --------------------------------------------
def calculate_ek_alignment_score(coalition, ek_seats, majority_threshold):
    expanded = get_expanded_coalition(coalition)
    coalition_ek_total = sum(ek_seats.get(p, 0) for p in expanded)
    total_ek = sum(ek_seats.values()) or 1  # avoid division by zero

    normalized_score = coalition_ek_total / total_ek

    if coalition_ek_total >= majority_threshold:
        return 1.0, coalition_ek_total  # Return score and EK seats
    return normalized_score, coalition_ek_total


# --------------------------------------------
# Compute mean Jensen-Shannon divergence for a set of parties
# --------------------------------------------

def mean_jsd_for_coalition(coalition, topic_vectors):
    if len(coalition) < 2:
        return 0.0  # trivial case

    jsd_values = []
    for p1, p2 in combinations(coalition, 2):
        v1 = topic_vectors.get(p1)
        v2 = topic_vectors.get(p2)
        if v1 is not None and v2 is not None:
            jsd = jensenshannon(v1, v2, base=2)
            jsd_values.append(jsd)
    return np.mean(jsd_values) if jsd_values else 0.0


# -------------------------------
# Define main prediction function
# -------------------------------
def predict_coalitions(seat_distribution, coalition_counter, ek_zetels, Jaar, threshold=76, top_k=5, topic_vectors=None):
    parties = list(seat_distribution.keys())

    # ✅ Get Eerste Kamer seat distribution for the given year
    ek_year_data = ek_zetels[ek_zetels['Jaar'] == Jaar].copy() 
    ek_seat_dist = dict(zip(ek_year_data['Partij'], ek_year_data['Zetels']))

    valid_coalitions = []

    for r in range(2, len(parties) + 1):
        for combo in combinations(parties, r):

            # -------------------------------
            # Check if the coalition includes the largest party (comment if opposition coalition)
            largest_party = max(seat_distribution.items(), key=lambda x: x[1])[0]
            if largest_party not in combo:
                continue  # Skip coalitions that don't include the largest party 
            # -------------------------------

            seats = sum(seat_distribution[p] for p in combo)
            if seats >= threshold:

                if is_unrealistic_combo(combo):
                    continue

                ek_year_data['party'] = ek_year_data['Partij']
                ek_seat_dist = dict(zip(ek_year_data['Partij'], ek_year_data['Zetels']))

                ek_score, ek_total_seats = calculate_ek_alignment_score(combo, ek_seat_dist, majority_threshold=38)

                # Calculate historical score with lineage adjustments and seat scaling
                historical_score = calculate_historical_score(combo, coalition_counter, seat_distribution)
                
                # Calculate ideology score
                ideology_score = ideological_distance(combo)

                # Apply penalties for party count and seat surplus
                party_penalty = max(0, len(combo) - 4) * 2
                surplus_penalty = max(0, seats - 90) * 0.5

                jsd_penalty = mean_jsd_for_coalition(combo, topic_vectors)

                # Final score computation
                score = (
                    (historical_score * 2)
                    - (ideology_score * 2)
                    + (ek_score * 0.25)  # new EK weight
                    - 10 * jsd_penalty
                    - (party_penalty * 2)
                    - surplus_penalty
                )

                # Given a fixed score range
                min_score = -2
                max_score = 2

                # Calculate percentage
                final_score = (score - min_score) / (max_score - min_score) * 100
                final_score = max(0, min(100, final_score))


                valid_coalitions.append({
                    "coalition": combo,
                    "seats": seats,
                    "historical_score": round(historical_score, 2),
                    "ideology_score": round(ideology_score, 2),
                    "ek_score": round(ek_score, 2),
                    "ek_total_seats": ek_total_seats,
                    "jsd_penalty": round(jsd_penalty, 2),
                    "party_penalty": round(party_penalty, 2),
                    "surplus_penalty": round(surplus_penalty, 2),
                    "final_score": round(final_score, 1)
                })



    valid_coalitions.sort(key=lambda x: (-x["final_score"], x["seats"]))  # Favor lower seat counts
    return valid_coalitions[:top_k]


