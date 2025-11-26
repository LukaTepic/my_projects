#%%
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from scipy.fft import rfft, irfft, fftshift, rfftfreq
from scipy.stats import norminvgauss

def gaussian(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def laplace(x, mean, scale):
    return (1 / (2 * scale)) * np.exp(-abs(x - mean) / scale)

def get_data(symbol, start_date, end_date, interval='1d'):
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data['Close']# Return only the 'Close' prices

def calculate_log_returns(data):
    log_returns = np.log(data / data.shift(1))

    return log_returns

def convert_params(alpha, beta, delta, mu):
    a = alpha * delta
    b = beta * delta
    loc = mu
    scale = delta
    return a, b, loc, scale

def cf_nig(t, alpha, beta, delta, mu):
    return np.exp(1j * mu * t + delta * (np.sqrt(alpha**2 - beta**2) - np.sqrt(alpha**2 - (beta + 1j * t)**2)))

def cf_mixture(t, weights, alphas, betas, deltas, mus):
    cf = np.zeros_like(t, dtype=complex)
    for w, alpha, beta, delta, mu in zip(weights, alphas, betas, deltas, mus):
        cf += w * cf_nig(t, alpha, beta, delta, mu)
    return cf

def gaussian(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

class InitHHMMFunctions:
    def __init__(self, n_sibs, data=None):

        self.n_sibs = n_sibs
        self.data = data

    def initialize_pi(self):
        pi = {}

        def recursive_initialize(depth=0, parent_idx=0):
            if depth == 0:
                pi[depth] = np.ones(self.n_sibs[0]) / self.n_sibs[0]
                for i in range(self.n_sibs[0]):
                    recursive_initialize(depth + 1, i)
            elif depth < len(self.n_sibs):
                if depth not in pi:
                    pi[depth] = {}
                num_states_current = self.n_sibs[depth]
                for j in range(num_states_current):
                    idx = parent_idx * num_states_current + j
                    pi[depth][parent_idx] = np.ones(self.n_sibs[depth]) / self.n_sibs[depth]
                    if depth + 1 < len(self.n_sibs):
                        recursive_initialize(depth + 1, idx)

        recursive_initialize()

        return pi

    def initialize_termination_probs(self):

        termination = {}

        def recursive_initialize(depth=0, parent_idx=0):
            if depth == 0:
                termination[depth] = np.ones(self.n_sibs[0]) * 0.1
                for i in range(self.n_sibs[0]):
                    recursive_initialize(depth + 1, i)
            elif depth < len(self.n_sibs) and depth != len(self.n_sibs) - 1:
                if depth not in termination:
                    termination[depth] = {}
                num_states_current = self.n_sibs[depth]
                for j in range(num_states_current):
                    idx = parent_idx * num_states_current + j
                    termination[depth][parent_idx] = np.ones(self.n_sibs[depth]) * 0.1
                    if depth + 1 < len(self.n_sibs):
                        recursive_initialize(depth + 1, idx)
            else:
                if depth not in termination:
                    termination[depth] = {}
                num_states_current = self.n_sibs[depth]
                for j in range(num_states_current):
                    idx = parent_idx * num_states_current + j
                    termination[depth][parent_idx] = np.ones(self.n_sibs[depth]) * 0.1
                    if depth + 1 < len(self.n_sibs):
                        recursive_initialize(depth + 1, idx)

        recursive_initialize()
        return termination

    def initialize_transition_matrix(self):
        A = {}

        def construct_matrix(N):
            matrix = np.zeros((N, N))
            np.fill_diagonal(matrix, 0.8)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        matrix[i, j] = 0.1 / (N - 1)
            return matrix

        def recursive_initialize(depth=0, parent_idx=0):
            if depth == 0:
                A[depth] = construct_matrix(self.n_sibs[depth])
                for i in range(self.n_sibs[0]):
                    recursive_initialize(depth + 1, i)
            elif depth < len(self.n_sibs):
                if depth not in A:
                    A[depth] = {}
                num_states_current = self.n_sibs[depth]
                for j in range(num_states_current):
                    idx = parent_idx * num_states_current + j
                    A[depth][parent_idx] = construct_matrix(n_sibs[depth])
                    if depth + 1 < len(self.n_sibs):
                        recursive_initialize(depth + 1, idx)

        recursive_initialize()

        return A

    def initialize_hierarchical_hmm(self):

        pi = self.initialize_pi(self.n_sibs)
        A = self.initialize_transition_matrix(self.n_sibs)
        termination = self.initialize_termination_probs(self.n_sibs)

        return pi, A, termination

    def num_parents_at_depth(self, depth):
        if depth == 0:
            return 0
        else:
            num_parents = 1
            for i in range(depth):
                num_parents *= self.n_sibs[i]
            return num_parents

    def num_states_bottom_level(self):
        num_states = 1
        for n in self.n_sibs:
            num_states *= n
        return num_states

    def initialize_emission_params(self):
        num_states = self.num_states_bottom_level()

        mu_lower = np.percentile(self.data, 30)
        mu_upper = np.percentile(self.data, 60)
        mus = np.random.uniform(mu_lower, mu_upper, num_states)

        sigma_lower = 0.8 * np.std(self.data)
        sigma_upper = 2 * np.std(self.data)
        sigmas = np.random.uniform(sigma_lower, sigma_upper, num_states)

        return mus, sigmas

    def num_states_at_each_depth(self):
        num_states_list = []
        product = 1
        for n in self.n_sibs:
            product *= n
            num_states_list.append(product)
        return num_states_list




class TrainHHMM:
    def __init__(self, n_sibs, O, N_iter, dist='gaussian'):

        self.n_sibs = n_sibs
        self.O = O
        self.dist = dist
        self.T = len(O)
        self.D = len(n_sibs)
        self.hmm_initializer = InitHHMMFunctions(n_sibs, O)
        self.pi = self.hmm_initializer.initialize_pi()
        self.A = self.hmm_initializer.initialize_transition_matrix()
        self.A_end = self.hmm_initializer.initialize_termination_probs()
        self.Means, self.Scales = self.hmm_initializer.initialize_emission_params()
        self.N_iter = N_iter

    def compute_pdf(self, x):
        if self.dist == 'gaussian':
            coefficients = 1 / (np.sqrt(2 * np.pi * np.array(self.Scales) ** 2))
            exponents = -((x - np.array(self.Means)) ** 2) / (2 * np.array(self.Scales) ** 2)
            return coefficients * np.exp(exponents)
        else:
            return np.ones(len(self.Means))  # Placeholder for other distributions

    def forward_activation_algorithm_np(self):
        max_state = self.hmm_initializer.num_states_bottom_level()
        num_states = self.hmm_initializer.num_states_at_each_depth()
        num_parents = [self.hmm_initializer.num_parents_at_depth(d) for d in range(self.D)]

        alpha_b = np.zeros((self.D, self.T, max_state))
        alpha_e = np.zeros((self.D, self.T, max_state))
        scales = np.ones(self.T)
        for t in range(self.T):
            if t == 0:
                alpha_b[0, 0, :num_states[0]] = self.pi[0]
                for d in range(1, self.D):
                    n_siblings = self.n_sibs[d]
                    alpha_i1 = 0
                    for i in range(num_parents[d]):
                        alpha_i1 += n_siblings
                        alpha_i2 = i * n_siblings
                        alpha_b[d, 0, alpha_i2:alpha_i1] = np.dot(alpha_b[d - 1, 0, i], self.pi[d][i])
            else:
                alpha_b[0, t, :num_states[0]] = np.dot(alpha_e[0, t - 1, :num_states[0]], self.A[0])
                for d in range(1, self.D):
                    n_siblings = self.n_sibs[d]
                    alpha_i1 = 0
                    for i in range(num_parents[d]):
                        alpha_i1 += n_siblings
                        alpha_i2 = i * n_siblings
                        alpha_b[d, t, alpha_i2:alpha_i1] = alpha_b[d - 1, t, i] * self.pi[d][i] + np.dot(
                            alpha_e[d, t - 1, alpha_i2:alpha_i1], self.A[d][i])

            alpha_e[self.D - 1, t, :] = alpha_b[self.D - 1, t, :] * self.compute_pdf(self.O[t])
            scale_factor = np.sum(alpha_e[self.D - 1, t, :])
            scales[t] = scale_factor

            alpha_e[self.D - 1, t, :] /= scale_factor
            for d in range(self.D - 2, -1, -1):
                n_siblings = self.n_sibs[d + 1]
                alpha_i1 = 0
                for i in range(num_parents[d + 1]):
                    alpha_i1 += n_siblings
                    alpha_i2 = i * n_siblings
                    alpha_e[d, t, i] = np.dot(alpha_e[d + 1, t, alpha_i2:alpha_i1], self.A_end[d + 1][i])

        return alpha_b, alpha_e, scales

    def backward_activation_algorithm_np(self, scales_forward):
        max_state = self.hmm_initializer.num_states_bottom_level()
        num_states = self.hmm_initializer.num_states_at_each_depth()
        num_parents = [self.hmm_initializer.num_parents_at_depth(d) for d in range(self.D)]

        beta_b = np.zeros((self.D, self.T, max_state))
        beta_e = np.zeros((self.D, self.T, max_state))
        # Initialization
        for t in range(self.T - 1, -1, -1):
            if t == self.T - 1:
                beta_e[0, t, :num_states[0]] = self.A_end[0]
                for d in range(1, self.D):
                    n_siblings = self.n_sibs[d]
                    beta_i1 = 0
                    for i in range(num_parents[d]):
                        beta_i1 += n_siblings
                        beta_i2 = i * n_siblings
                        beta_e[d, t, beta_i2:beta_i1] = np.dot(beta_e[d - 1, t, i], self.A_end[d][i])
            else:
                beta_e[0, t, :num_states[0]] = np.dot(self.A[0], beta_b[0, t + 1, :num_states[0]])
                for d in range(1, self.D):
                    n_siblings = self.n_sibs[d]
                    beta_i1 = 0
                    for i in range(num_parents[d]):
                        beta_i1 += n_siblings
                        beta_i2 = i * n_siblings
                        beta_e[d, t, beta_i2:beta_i1] = beta_e[d - 1, t, i] * self.A_end[d][i] + np.dot(self.A[d][i],
                                                                                                   beta_b[d, t + 1,
                                                                                                   beta_i2:beta_i1])

            beta_b[self.D - 1, t, :] = beta_e[self.D - 1, t, :] * self.compute_pdf(self.O[t])
            scale_factor = scales_forward[t]
            beta_b[self.D - 1, t, :] /= scale_factor

            for d in range(self.D - 2, -1, -1):
                n_siblings = self.n_sibs[d + 1]
                beta_i1 = 0
                for i in range(num_parents[d + 1]):
                    beta_i1 += n_siblings
                    beta_i2 = i * n_siblings
                    beta_b[d, t, i] = np.dot(beta_b[d + 1, t, beta_i2:beta_i1], self.pi[d + 1][i])
        return beta_b, beta_e
    def g_est_np(self):
        alpha_b, alpha_e, scales = self.forward_activation_algorithm_np()
        beta_b, beta_e = self.backward_activation_algorithm_np(scales)

        logl = np.sum(np.log(1 / scales))

        max_state = self.hmm_initializer.num_states_bottom_level()
        num_states = self.hmm_initializer.num_states_at_each_depth()

        for d in range(self.D):
            n_siblings = self.n_sibs[d]
            parents_at_d = self.hmm_initializer.num_parents_at_depth(d)
            beta_i1 = 0
            beta_i2 = 0

            if d == 0:

                self.pi[0] = alpha_b[d, 0, 0:num_states[0]] * beta_b[d, 0, 0:num_states[0]]
                self.pi[0] /= np.sum(self.pi[0])
                self.A_end[0] = alpha_e[d, -1, 0:num_states[0]] * beta_e[d, -1, 0:num_states[0]]

                for x in range(n_siblings):
                    for y in range(n_siblings):
                        A_dij = self.A[d][x][y]
                        product = alpha_e[d, :-1, x] * A_dij * beta_b[d, 1:, y]
                        self.A[d][x][y] = np.sum(product)

                    matrixsum = np.sum(self.A[d][x]) + self.A_end[d][x]
                    self.A_end[0][x] /= matrixsum
                    self.A[0][x] /= matrixsum
            else:
                for i in range(parents_at_d):
                    beta_i1 += n_siblings

                    alpha_reshaped = alpha_b[d - 1, 1:self.T, i].reshape(-1, 1)
                    intermediate_pi = alpha_reshaped * self.pi[d][i]

                    beta_reshaped = beta_e[d - 1, 0:self.T - 1, i].reshape(-1, 1)
                    intermediate_end = beta_reshaped * self.A_end[d][i]

                    self.pi[d][i] = alpha_b[d, 0, beta_i2:beta_i1] * beta_b[d, 0, beta_i2:beta_i1] + np.sum(
                        intermediate_pi * beta_b[d, 1:self.T, beta_i2:beta_i1], axis=0)
                    self.pi[d][i] /= np.sum(self.pi[d][i])

                    self.A_end[d][i] = np.sum(alpha_e[d, 0:self.T - 1, beta_i2:beta_i1] * intermediate_end, axis=0) + alpha_e[d,-1,beta_i2:beta_i1] * beta_e[d,-1,beta_i2:beta_i1]
                    for x in range(n_siblings):
                        for y in range(n_siblings):
                            A_dij = self.A[d][i][x][y]
                            product = alpha_e[d, :-1, x + beta_i2] * A_dij * beta_b[d, 1:, y + beta_i2]
                            self.A[d][i][x][y] = np.sum(product)
                        matrixsum = np.sum(self.A[d][i][x]) +self.A_end[d][i][x]
                        self.A_end[d][i][x] /= matrixsum
                        self.A[d][i][x] /= matrixsum
                    beta_i2 += n_siblings

        posterior = np.ones((max_state, self.T))
        total_prob = np.sum(alpha_e[self.D - 1, :, :] * beta_e[self.D - 1, :, :], axis=1)

        for i in range(max_state):
            posterior[i, :] = (alpha_e[self.D - 1, :, i] * beta_e[self.D - 1, :, i]) / total_prob[:]

        for i in range(max_state):
            # Compute weighted mean for state i
            weighted_sum = np.sum(posterior[i, :] * self.O)
            total_weight = np.sum(posterior[i, :])
            self.Means[i] = weighted_sum / total_weight

            # Compute weighted standard deviation for state i
            variance = np.sum(posterior[i, :] * (self.O - self.Means[i]) ** 2) / total_weight
            self.Scales[i] = np.sqrt(variance)

        return  posterior, logl

    def EM(self):
        log_likelihood = []
        for i in range(self.N_iter):
            posterior, logl = self.g_est_np()
            print(f"Iteration {i + 1} completed.")
            log_likelihood.append(logl)
        return posterior, log_likelihood



class ForecastHHMM:
    def __init__(self,n_sibs,O,max_T,N_iter, initial_state_idx = -1):
        self.S0 = O[-1]
        self.n_sibs = n_sibs
        self.O = O
        self.initial_state_idx = initial_state_idx
        self.max_T = max_T

        self.hmm_initializer = InitHHMMFunctions(n_sibs, O)
        hmm_train = TrainHHMM(n_sibs, O, N_iter)
        self.posterior, self.log_likelihood = hmm_train.EM()

        self.pi = hmm_train.pi
        self.A = hmm_train.A
        self.A_end = hmm_train.A_end
        self.Means = hmm_train.Means
        self.Scales = hmm_train.Scales

        self.bottom_states = self.hmm_initializer.num_states_bottom_level()
        self.probs = np.zeros((self.bottom_states, self.max_T + 1), dtype=float)
        if initial_state_idx == -1:
            self.probs[:, 0] = self.posterior[:, -1]
        else:
            self.probs[self.initial_state_idx, 0] = 1.0

    def find_siblings_and_parents(self, state1, state2):
        depth = len(self.n_sibs)
        sibling_and_parent_indices = {
            'sibling_indices_within_group': {i: None for i in range(depth)},
            'parent_indices': {i: None for i in range(1, depth)}
        }

        for i in range(depth):
            sibling_group_size = self.n_sibs[i]
            product_of_deeper_layers = 1
            for j in range(i + 1, depth):
                product_of_deeper_layers *= self.n_sibs[j]

            sibling1 = (state1 // product_of_deeper_layers) % sibling_group_size
            sibling2 = (state2 // product_of_deeper_layers) % sibling_group_size

            sibling_and_parent_indices['sibling_indices_within_group'][i] = (sibling1, sibling2)

        for i in range(1, depth):
            product_of_deeper_layers = 1
            for j in range(i, depth):
                product_of_deeper_layers *= self.n_sibs[j]

            parent1 = state1 // product_of_deeper_layers
            parent2 = state2 // product_of_deeper_layers

            sibling_and_parent_indices['parent_indices'][i] = (parent1, parent2)

        return sibling_and_parent_indices
    def find_transition_depth(self,sibling_and_parent_indices):
        parent_indices = sibling_and_parent_indices['parent_indices']
        transition_depth = 0
        for depth in range(1, len(parent_indices) + 1):
            parent1, parent2 = parent_indices[depth]
            if parent1 != parent2:
                transition_depth = depth
                break
        return transition_depth

    def compute_probs(self):
        for t in range(1, self.max_T + 1):
            for i in range(self.bottom_states):
                for j in range(self.bottom_states):
                    sibling_and_parent_indices = self.find_siblings_and_parents( i, j)
                    model_depth = len(n_sibs) - 1
                    md_D = len(n_sibs)
                    max_depth = self.find_transition_depth(sibling_and_parent_indices)
                    terminations = md_D - self.find_transition_depth(sibling_and_parent_indices)

                    if max_depth == 0:
                        target_sib_idx = sibling_and_parent_indices['sibling_indices_within_group'][model_depth][1]
                        source_sib_idx = sibling_and_parent_indices['sibling_indices_within_group'][model_depth][0]
                        source_parent_idx1 = sibling_and_parent_indices['parent_indices'][model_depth][0]

                        self.probs[j, t] += self.probs[i, t - 1] * self.A[model_depth][source_parent_idx1][source_sib_idx][
                            target_sib_idx]
                    elif terminations == 1:
                        target_sib_idx1 = sibling_and_parent_indices['sibling_indices_within_group'][model_depth][1]
                        source_sib_idx1 = sibling_and_parent_indices['sibling_indices_within_group'][model_depth][0]

                        target_sib_idx2 = sibling_and_parent_indices['sibling_indices_within_group'][model_depth - 1][1]
                        source_sib_idx2 = sibling_and_parent_indices['sibling_indices_within_group'][model_depth - 1][0]

                        source_parent_idx1 = sibling_and_parent_indices['parent_indices'][model_depth][0]
                        target_parent_idx1 = sibling_and_parent_indices['parent_indices'][model_depth][1]

                        if model_depth == 1:
                            self.probs[j, t] += self.probs[i, t - 1] * self.A_end[model_depth][source_parent_idx1][source_sib_idx1] * \
                                           self.A[model_depth - 1][source_sib_idx2][target_sib_idx2] * \
                                           self.pi[model_depth][target_parent_idx1][target_sib_idx1]
                        else:
                            source_parent_idx2 = sibling_and_parent_indices['parent_indices'][model_depth - 1][0]

                            self.probs[j, t] += self.probs[i, t - 1] * self.A_end[model_depth][source_parent_idx1][source_sib_idx1] * \
                                           self.A[model_depth - 1][source_parent_idx2][source_sib_idx2][target_sib_idx2] * \
                                           self.pi[model_depth][target_parent_idx1][target_sib_idx1]
                    elif terminations > 1:
                        int_prob = 1
                        for k in range(terminations):

                            target_sib_idx = \
                            sibling_and_parent_indices['sibling_indices_within_group'][model_depth - k][1]
                            source_sib_idx = \
                            sibling_and_parent_indices['sibling_indices_within_group'][model_depth - k][0]
                            source_parent_idx1 = sibling_and_parent_indices['parent_indices'][model_depth - k][0]
                            target_parent_idx1 = sibling_and_parent_indices['parent_indices'][model_depth - k][1]

                            if k == 0:
                                int_prob *= self.probs[i, t - 1] * self.A_end[model_depth - k][source_parent_idx1][
                                    source_sib_idx] * self.pi[model_depth - k][target_parent_idx1][target_sib_idx]
                            elif k > 0 and terminations > 2 and k < terminations - 1:
                                int_prob *= self.A_end[model_depth - k][source_parent_idx1][source_sib_idx] * \
                                            self.pi[model_depth - k][target_parent_idx1][target_sib_idx]

                            elif k == terminations - 1:
                                target_sib_idx1 = \
                                sibling_and_parent_indices['sibling_indices_within_group'][model_depth - k][1]
                                source_sib_idx1 = \
                                sibling_and_parent_indices['sibling_indices_within_group'][model_depth - k][0]
                                target_sib_idx2 = \
                                sibling_and_parent_indices['sibling_indices_within_group'][model_depth - k - 1][1]
                                source_sib_idx2 = \
                                sibling_and_parent_indices['sibling_indices_within_group'][model_depth - k - 1][0]

                                source_parent_idx1 = sibling_and_parent_indices['parent_indices'][model_depth - k][0]
                                target_parent_idx1 = sibling_and_parent_indices['parent_indices'][model_depth - k][1]
                                inter_prob = int_prob * self.A_end[model_depth - k][source_parent_idx1][source_sib_idx1] * \
                                             self.A_end[model_depth - k - 1][source_sib_idx2] * self.pi[model_depth - k - 1][
                                                 target_sib_idx2]
                                int_prob *= self.A_end[model_depth - k][source_parent_idx1][source_sib_idx1] * \
                                            self.A[model_depth - k - 1][source_sib_idx2][target_sib_idx2] * \
                                            self.pi[model_depth - k][target_parent_idx1][target_sib_idx1]

                        self.probs[j, t] += int_prob + inter_prob
            self.probs[:, t] = self.probs[:, t] / np.sum(self.probs[:, t])

        return self.probs

    def check_probabilities_sum_to_one(self, tolerance=1e-6):
        self.probs = self.compute_probs()
        if self.probs.shape[1] != self.max_T + 1:
            raise ValueError("The number of columns in the probs array does not match max_T + 1.")
        sums = np.sum(self.probs, axis=0)

        return print(np.isclose(sums, 1, atol=tolerance))

    def calculate_dynamic_range(self):
        self.probs = self.compute_probs()
        total_weighted_mean = 0
        total_weighted_variance = 0

        for t in range(self.max_T):
            weights = self.probs[:, t]
            means = self.Means[:]
            stds = self.Scales[:]
            weighted_means = weights * means
            weighted_variances = weights * stds ** 2

            total_weighted_mean += np.sum(weighted_means)
            total_weighted_variance += np.sum(weighted_variances)

        avg_weighted_mean = total_weighted_mean / self.max_T
        avg_weighted_variance = total_weighted_variance / self.max_T
        avg_weighted_std = np.sqrt(avg_weighted_variance)

        # Determine the range based on average mean and std
        minV = avg_weighted_mean - 30 * avg_weighted_std
        maxV = avg_weighted_mean + 30 * avg_weighted_std

        return minV, maxV

    def convolve_mixed_densities_over_time(self):
        self.probs = self.compute_probs()
        Mu = np.array(self.Means)
        Sig = np.array(self.Scales)

        minV, maxV = self.calculate_dynamic_range()
        x = np.linspace(minV, maxV, 10024)

        cumulative_distributions = []

        for t in range(self.max_T):
            time_step_distribution = np.zeros_like(x)
            weights = self.probs[:, t]
            means = Mu[:]
            stds = Sig[:]

            for weight, mean, std in zip(weights, means, stds):
                time_step_distribution += weight * gaussian(x, mean, std)

            if t == 0:
                current_distribution = time_step_distribution
            else:
                current_distribution = np.convolve(current_distribution, time_step_distribution, mode='same')

            current_distribution /= np.sum(current_distribution)
            cumulative_distributions.append(current_distribution.copy())

        return x, cumulative_distributions

    def log_return_to_price_density(self,S0):
        x_log_return, cumulative_distributions = self.convolve_mixed_densities_over_time()
        stock_price_density = []
        for distribution in cumulative_distributions:
            if len(distribution) != len(x_log_return):
                raise ValueError("The length of each distribution in cumulative_distributions must match x_log_return.")
            S = S0 * np.exp(x_log_return)

            # Adjust the PDF from log return to stock price by the Jacobian (dS/dX = S)
            stock_price_pdf = distribution / S

            normalization_factor = np.trapz(stock_price_pdf, S)
            normalized_stock_price_pdf = stock_price_pdf / normalization_factor

            stock_price_density.append(normalized_stock_price_pdf)

        return S, stock_price_density

    def create_probability_cone(self, S0, confidence=0.95):
        x_stock_price, stock_price_density = self.log_return_to_price_density(S0)

        lower_bounds = []
        upper_bounds = []
        mid_bounds = []

        for density in stock_price_density:
            cumulative_density = np.cumsum(density)

            total_mass = cumulative_density[-1]
            lower_threshold = ((1-confidence)/2) * total_mass
            upper_threshold = (1-((1-confidence)/2))  * total_mass
            mid_threshold = 0.5 * total_mass

            lower_idx = np.where(cumulative_density >= lower_threshold)[0][0]
            upper_idx = np.where(cumulative_density >= upper_threshold)[0][0]
            mid_idx = np.where(cumulative_density >= mid_threshold)[0][0]

            lower_bound = x_stock_price[lower_idx]
            upper_bound = x_stock_price[upper_idx]
            mid_bound = x_stock_price[mid_idx]

            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            mid_bounds.append(mid_bound)

        return lower_bounds, upper_bounds, mid_bounds



symbol = ("SPY")
start_date = "1980-12-01"
end_date = "2023-12-9"
data = pd.DataFrame()
interval = '1d'
data[symbol] = get_data(symbol, start_date, end_date,interval)
historical_data = data.copy()

data = calculate_log_returns(data)
data.dropna(inplace=True)
historical_data = historical_data.loc[data.index]
O = data[symbol].values.copy()


S0  = historical_data[symbol].iloc[-1] # Last closing price
n_sibs = [2, 2]                        # Model hierarchy structure 2 parent states followed by 2 children each
N = 20                               # Number of iterations for EM algorithm
initial_state_idx = -1                  # -1 - initial state probabilities are represented as a mixture of possible state probabilities as implied by the model , 0,1,2,3... (Can initialize state probabilities to 1 for any given state with indices (0,1,2... max_state)
maxT = 30                              # Forecast period

forecast = ForecastHHMM(n_sibs,O,maxT,N, initial_state_idx)

forecasted_state_probabilities = forecast.compute_probs()

x_returns, forecasted_return_densities = forecast.convolve_mixed_densities_over_time()

x_prices, forecasted_stock_densities = forecast.log_return_to_price_density(S0)

lower, upper, mid= forecast.create_probability_cone(S0,confidence=0.95)
model_posterior = forecast.posterior
model_likelihood = forecast.log_likelihood
check = forecast.check_probabilities_sum_to_one()


#%%

import matplotlib.pyplot as plt

def plot_probability_cone(lower_bounds, upper_bounds, mid_bounds):
    time_axis = range(len(lower_bounds))

    plt.figure(figsize=(10, 6))
    plt.fill_between(time_axis, lower_bounds, upper_bounds, color='skyblue', alpha=0.4, label='95% Confidence Interval')
    plt.plot(time_axis, mid_bounds, color='darkblue', linestyle='--', label='Median')

    plt.title('Probability Cone of Stock Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_3d_densities(density, x_range, T, type = "Price"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    time_steps = np.arange(1, T + 1)

    Y = x_range
    X, Z = np.meshgrid(time_steps, Y)

    W = np.vstack(density).T

    ax.plot_surface(X, Z, W, cmap='viridis')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(type)
    ax.set_zlabel('Density')

    plt.show()


def plot_posterior_plotly(posterior):
    fig = go.Figure()
    for state in range(posterior.shape[0]):
        fig.add_trace(go.Scatter(y=posterior[state, :], mode='lines', name=f'State {state}'))

    fig.update_layout(
        title="Posterior Distribution Over Time",
        xaxis_title="Time Steps",
        yaxis_title="Posterior Probability",
        legend_title="States"
    )
    pio.write_html(fig, file='filename.html', auto_open=True)

    fig.show()

def plot_probabilities(probs, max_T):
    bottom = len(probs[:,0])
    if probs.shape != (bottom, max_T+1):
        raise ValueError("The shape of the probs array does not match the expected dimensions.")
    plt.figure(figsize=(12, 6))
    time_points = np.arange(max_T + 1)
    for i in range(bottom):
        plt.plot(time_points, probs[i], label=f'Variable {i+1}')

    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Probabilities Over Time')
    plt.legend()
    plt.show()

plot_3d_densities(forecasted_return_densities,x_returns, maxT,type = "Returns")
plot_3d_densities(forecasted_stock_densities,x_prices, maxT, type = "Price")
plot_probabilities(forecasted_state_probabilities,maxT)
plot_posterior_plotly(model_posterior)
plot_probability_cone(lower, upper, mid)
# %%
