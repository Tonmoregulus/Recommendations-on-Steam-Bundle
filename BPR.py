import numpy as np
import random
from collections import defaultdict
import pickle
import time

class BPR:
    def __init__(self, rank, n_users, n_items, lambda_u=0.1, lambda_i=0.1, lambda_j=0.01, learning_rate=0.05):
        self.rank = rank
        self.n_users = n_users
        self.n_items = n_items
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate

        self.user_factors = np.random.normal(0, 1, (n_users, rank)).astype(np.float32)
        self.item_factors = np.random.normal(0, 0.01, (n_items, rank)).astype(np.float32)

    def train(self, train_data, n_samples, batch_size=1000, epochs=10):
        print("Starting training...")
        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0
            total_samples = 0
            for _ in range(n_samples // batch_size):
                users, pos_items, neg_items = self._sample_batch(train_data, batch_size)
                batch_loss = self._update(users, pos_items, neg_items)
                total_loss += batch_loss
                total_samples += len(users) 
            # compute avg loss
            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

    def _sample_batch(self, train_data, batch_size):
        users, pos_items, neg_items = [], [], []
        for _ in range(batch_size):
            user, pos_item = random.choice(train_data)
            neg_item = random.randint(0, self.n_items - 1)
            while neg_item in user_item_map[user]:
                neg_item = random.randint(0, self.n_items - 1)
            users.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)
        return np.array(users), np.array(pos_items), np.array(neg_items)

    def _update(self, users, pos_items, neg_items):
        u_factors = self.user_factors[users]
        pos_factors = self.item_factors[pos_items]
        neg_factors = self.item_factors[neg_items]

        # compute x_uij and normalize
        x_uij = np.sum((u_factors / np.linalg.norm(u_factors, axis=1, keepdims=True)) *
                    ((pos_factors - neg_factors) / np.linalg.norm(pos_factors - neg_factors, axis=1, keepdims=True)), axis=1)
        sigmoid = 1 / (1 + np.exp(-x_uij))

        # grad compute
        grad_u = (pos_factors - neg_factors) * (1 - sigmoid)[:, np.newaxis] - self.lambda_u * u_factors
        grad_pos = u_factors * (1 - sigmoid)[:, np.newaxis] - self.lambda_i * pos_factors
        grad_neg = -u_factors * (1 - sigmoid)[:, np.newaxis] - self.lambda_j * neg_factors

        # grad license
        max_grad_norm = 1.0
        grad_u = np.clip(grad_u, -max_grad_norm, max_grad_norm)
        grad_pos = np.clip(grad_pos, -max_grad_norm, max_grad_norm)
        grad_neg = np.clip(grad_neg, -max_grad_norm, max_grad_norm)

        # update params
        self.user_factors[users] += self.learning_rate * grad_u
        self.item_factors[pos_items] += self.learning_rate * grad_pos
        self.item_factors[neg_items] += self.learning_rate * grad_neg

        # normalize embedding matrix
        self.user_factors[users] /= np.linalg.norm(self.user_factors[users], axis=1, keepdims=True)
        self.item_factors[pos_items] /= np.linalg.norm(self.item_factors[pos_items], axis=1, keepdims=True)
        self.item_factors[neg_items] /= np.linalg.norm(self.item_factors[neg_items], axis=1, keepdims=True)

        log_sigmoid_sum = -np.sum(np.log(sigmoid))
        user_regularization = 0.5 * self.lambda_u * np.sum(u_factors ** 2) 
        pos_item_regularization = 0.5 * self.lambda_i * np.sum(pos_factors ** 2) 
        neg_item_regularization = 0.5 * self.lambda_j * np.sum(neg_factors ** 2) 

        # test output loss
        #print(f"log_sigmoid_sum: {log_sigmoid_sum:.4f}")
        #print(f"user_regularization: {user_regularization:.4f}")
        #print(f"pos_item_regularization: {pos_item_regularization:.4f}")
        #print(f"neg_item_regularization: {neg_item_regularization:.4f}")

        # total loss
        loss = log_sigmoid_sum + user_regularization + pos_item_regularization + neg_item_regularization
        return loss



    def test(self, test_data, num_negatives=100):
        print("Testing model...")
        correct_pairs = 0
        total_pairs = 0
        
        for user, pos_item in test_data:
            pos_score = np.dot(self.user_factors[user], self.item_factors[pos_item])
            
            for _ in range(num_negatives):
                neg_item = random.randint(0, self.n_items - 1)
                while neg_item in user_item_map[user]:
                    neg_item = random.randint(0, self.n_items - 1)
                
                neg_score = np.dot(self.user_factors[user], self.item_factors[neg_item])
                
                if pos_score > neg_score:
                    correct_pairs += 1
                total_pairs += 1
        
        # compute AUC
        auc = correct_pairs / total_pairs
        print(f"Test AUC: {auc:.4f}")
        return auc



# load and pre-process data
print("Loading user_item_map...")
user_item_map=pickle.load(open('user_item_map','rb'))

print("Splitting data into train and test...")
train_data = []  
test_data = [] 
n_users = len(user_item_map)
n_items = 0

for user, items in user_item_map.items():
    n_items = max(n_items, max(items))
    for item in items:
        if random.random() < 0.8: 
            train_data.append((user, item))
        else:  
            test_data.append((user, item))

n_items += 1 

# initialize and train model
bpr_model = BPR(rank=10, n_users=n_users, n_items=n_items, learning_rate=0.01)
bpr_model.train(train_data, n_samples=len(train_data) * 10, batch_size=1000, epochs=10)

# test model
bpr_model.test(test_data)
