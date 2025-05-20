import pandas as pd
from collections import Counter
import itertools
from itertools import combinations
import math
import re

# -------------------------------
# Ideological spectrum (manually defined, -5=far-left to +5=far-right)
# -------------------------------
IDEOLOGY_MAP = {
    "SP": -5,
    "PvdA": -3,
    "GL": -3,
    "PvdD": -2,
    "D66": -1,
    "Volt": -1,
    "CDA": 1,
    "CU": 1,
    "SGP": 2,
    "VVD": 3,
    "JA21": 4,
    "PVV": 5,
    "BBB": 2,
    "Forum voor Democratie": 5,
    "DENK": -4,
    "50PLUS": 0,
    "BIJ1": -5,
    "LP": 3,
    "NSC": 1
}


def load_data():
    kabinetten = pd.read_csv('data/cabinets/kabinetten_schoongemaakt-no2023.csv')                   #Exclude 2021 data
    zetels_100 = pd.read_csv('data/zetelverdeling/zetel-data/tk_zetels100_1918-1956.csv')
    zetels_150 = pd.read_csv('data/zetelverdeling/zetel-data/tk_zetels150_1956-2023-no2023.csv')    #Exclude 2021 data
    zetels = pd.concat([zetels_100, zetels_150], ignore_index=True)
    ek_50_old = pd.read_csv('data/zetelverdeling/zetel-data/ek_zetels50_1888-1956_filled.csv')
    ek_75_new = pd.read_csv('data/zetelverdeling/zetel-data/ek_zetels75_1956-2023_filled.csv')
    ek_zetels = pd.concat([ek_50_old, ek_75_new], ignore_index=True)

    kabinetten['Partijen'] = kabinetten['Partijen'].dropna().str.split(', ')
    return kabinetten, zetels, ek_zetels


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
    ideologies = [IDEOLOGY_MAP.get(p, 0) for p in parties]
    if len(ideologies) <= 1:
        return 0
    pairwise_diffs = [abs(a - b) for a, b in itertools.combinations(ideologies, 2)]
    return sum(pairwise_diffs) / len(pairwise_diffs)


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
# Load topic codes from a structured txt file
# --------------------------------------------
# def load_party_topics(filename='party_classification_results.txt'):
#     party_topics = {}
#     current_party = None

#     with open(filename, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith("Party:"):
#                 current_party = line.replace("Party:", "").strip()
#                 party_topics[current_party] = set()
#             elif line.startswith("-") and current_party:
#                 match = re.match(r"-\s*(\d+)\s*-\s*.+?:", line)
#                 if match:
#                     topic_code = match.group(1)
#                     party_topics[current_party].add(topic_code)
#     return party_topics


# --------------------------------------------------
# Compute topic alignment score for a party combo
# --------------------------------------------------
# def topic_alignment_score(combo, party_topics):
#     topic_sets = [party_topics.get(party, set()) for party in combo if party in party_topics]

#     if len(topic_sets) < 2:
#         return 0.0

#     shared_topics = set.intersection(*topic_sets)
#     total_topics = set.union(*topic_sets)

#     if not total_topics:
#         return 0.0

#     return len(shared_topics) / len(total_topics)  # Value between 0 and 1



# -------------------------------
# Define main prediction function
# -------------------------------
def predict_coalitions(seat_distribution, coalition_counter, ek_zetels, Jaar, threshold=76, top_k=5):
    parties = list(seat_distribution.keys())
    # party_topics = load_party_topics()

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




                # topic_score = topic_alignment_score(combo, party_topics)

                # Calculate historical score with lineage adjustments and seat scaling
                historical_score = calculate_historical_score(combo, coalition_counter, seat_distribution)
                
                # Calculate ideology score
                ideology_score = ideological_distance(combo)

                # Apply penalties for party count and seat surplus
                party_penalty = max(0, len(combo) - 4) * 2
                surplus_penalty = max(0, seats - 90) * 0.5

                # Final score computation
                score = (
                    (historical_score * 2)
                    - (ideology_score * 2)
                    # + (topic_score * 2)
                    + (ek_score * 0.25)  # new EK weight
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
                    "party_penalty": round(party_penalty, 2),
                    "surplus_penalty": round(surplus_penalty, 2),
                    "final_score": round(final_score, 1)
                })



    valid_coalitions.sort(key=lambda x: (-x["final_score"], x["seats"]))  # Favor lower seat counts
    return valid_coalitions[:top_k]


