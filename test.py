import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

posts = pd.read_csv("data/post_data.csv")
users = pd.read_csv("data/user_data.csv")
views = pd.read_csv("data/view_data.csv")


# 1. define item space from posts
post_ids = posts['post_id'].unique()
post2id = {p:i for i,p in enumerate(post_ids)}

# 2. define user space from views
user_ids = views['user_id'].unique()
user2id = {u:i for i,u in enumerate(user_ids)}

# 3. filter views to valid posts/users
views = views[
    views['post_id'].isin(post2id.keys()) &
    views['user_id'].isin(user2id.keys())
]

# 4. map to indices
rows = views['user_id'].map(user2id)
cols = views['post_id'].map(post2id)

# 5. implicit feedback
data = np.ones(len(views))

# 6. build interaction matrix
R = coo_matrix(
    (data, (rows, cols)),
    shape=(len(user2id), len(post2id))
)

model = AlternatingLeastSquares(factors=50)
model.fit(R)

user_embeddings = model.user_factors.to_numpy()
item_embeddings = model.item_factors.to_numpy()

posts['text'] = posts['title'] + " " + posts['category']


vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
item_content_matrix = vectorizer.fit_transform(posts['text'])

def cosine_sim(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def get_user_content_vector(user_id):
    user_views = views[views['user_id'] == user_id]
    item_indices = user_views['post_id'].map(post2id).dropna().astype(int)

    if len(item_indices) == 0:
        return None

    vectors = item_content_matrix[item_indices]

    user_vec = np.asarray(vectors.mean(axis=0)).flatten()

    return user_vec

def cf_score(user_id, item_id):
    u = user2id[user_id]
    i = post2id[item_id]
    return np.dot(user_embeddings[u], item_embeddings[i])


def content_score(user_vec, item_vec):
    return cosine_similarity(user_vec, item_vec)[0][0]

def hybrid_score(user_id, item_id, alpha=0.5):
    cf = cf_score(user_id, item_id)

    user_vec = get_user_content_vector(user_id)
    if user_vec is None:
        return cf

    item_vec = item_content_matrix[post2id[item_id]].toarray().flatten()

    content = cosine_sim(user_vec, item_vec)

    return alpha * cf + (1 - alpha) * content


def mmr_recommend(user_id, candidate_items, k=10, lambda_=0.7):
    selected = []
    candidate_items = list(candidate_items)

    scores = {i: hybrid_score(user_id, i) for i in candidate_items}

    while len(selected) < k and len(candidate_items) > 0:
        mmr_scores = []

        for i in candidate_items:
            relevance = scores[i]

            if len(selected) == 0:
                diversity = 0
            else:
                sims = []
                for j in selected:
                    sim = cosine_sim(
                        item_content_matrix[post2id[i]].toarray().flatten(),
                        item_content_matrix[post2id[j]].toarray().flatten()
                    )
                    sims.append(sim)

                diversity = max(sims)

            mmr = lambda_ * relevance - (1 - lambda_) * diversity
            mmr_scores.append((i, mmr))

        best_item = max(mmr_scores, key=lambda x: x[1])[0]

        selected.append(best_item)
        candidate_items.remove(best_item)

    return selected

def get_candidates(user_id, N=100):
    if user_id not in user2id:
        return []

    u = user2id[user_id]

    scores = user_embeddings[u] @ item_embeddings.T

    top_idx = np.argsort(-scores)[:N] 

    id2post = {v:k for k,v in post2id.items()}
    return [id2post[i] for i in top_idx]


def recommend(user_id, top_k=10, candidate_N=100):
    candidates = get_candidates(user_id, N=candidate_N)

    final_items = mmr_recommend(
        user_id,
        candidate_items=candidates,
        k=top_k,
        lambda_=0.7
    )

    return final_items


# user_id = views['user_id'].iloc[0]  

# recs = recommend(user_id, top_k=10)

# print("Recommended posts:", recs)

import random

user_id = random.choice(views['user_id'].unique())

print("=" * 50)
print("Selected user_id:", user_id)

user_info = users[users['user_id'] == user_id].iloc[0]

print("\nUser profile:")
print("First name:", user_info.get('first_name'))
print("Last name:", user_info.get('last_name'))
print("Gender:", user_info.get('gender'))
print("City:", user_info.get('city'))
print("Academics:", user_info.get('academics'))

user_history = views[views['user_id'] == user_id]

recs = recommend(user_id, top_k=10)

print("\n" + "=" * 50)
print("Recommended posts:")
print(recs)

print("\n" + "=" * 50)
print("Recommended post details:\n")

for pid in recs:
    row = posts[posts['post_id'] == pid]

    if len(row) == 0:
        continue

    row = row.iloc[0]

    print(f"Post ID: {pid}")
    print("Title:", row.get('title'))
    print("Category:", row.get('category'))
    print("-" * 50)