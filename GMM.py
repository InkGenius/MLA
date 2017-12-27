import numpy as np
import random as rd
from NMI import compute_cost


def random_parameters(data, K):
    cols = (data.shape)[1]

    mu = np.zeros((K, cols))
    for k in range(K):
        idx = np.int(rd.random() * len(data))
        for col in range(cols):
            mu[k][col] += data[idx][col]

    sigma = []
    for k in range(K):
        sigma.append(np.cov(data.T))

    pi = np.ones(K) * 1.0 / K
    return mu, sigma, pi


def _e_step(data, K, mu, sigma, pi):
    idvs = len(data)

    resp = np.zeros((idvs, K))
    for k in range(K):
        for i in range(idvs):
            resp[i][k] = pi[k] * gaussian(data[i], mu[k], sigma[k])

    return resp


def e_step(data, K, mu, sigma, pi):
    idvs = (data.shape)[0]
    cols = (data.shape)[1]

    resp = np.zeros((idvs, K))

    for i in range(idvs):
        for k in range(K):
            resp[i][k] = pi[k] * gaussian(data[i], mu[k], sigma[k]) / likelihood(data[i], K, mu, sigma, pi)

    return resp


def log_likelihood(data, K, mu, sigma, pi):
    log_likelihood = 0.0
    for n in range(len(data)):
        log_likelihood += np.log(likelihood(data[n], K, mu, sigma, pi))
    return log_likelihood


def likelihood(x, K, mu, sigma, pi):
    rs = 0.0
    for k in range(K):
        rs += pi[k] * gaussian(x, mu[k], sigma[k])
    return rs


def m_step(data, K, resp):
    idvs = (data.shape)[0]
    cols = (data.shape)[1]

    mu = np.zeros((K, cols))
    sigma = np.zeros((K, cols, cols))
    pi = np.zeros(K)

    marg_resp = np.zeros(K)
    for k in range(K):
        for i in range(idvs):
            marg_resp[k] += resp[i][k]
            mu[k] += (resp[i][k]) * data[i]
        mu[k] /= marg_resp[k]

        for i in range(idvs):
            # x_i = (np.zeros((1,cols))+data[k])
            x_mu = np.zeros((1, cols)) + data[i] - mu[k]
            sigma[k] += (resp[i][k] / marg_resp[k]) * x_mu * x_mu.T

        pi[k] = marg_resp[k] / idvs

    return mu, sigma, pi


def gaussian(x, mu, sigma):
    idvs = len(x)
    norm_factor = (2 * np.pi) ** idvs

    norm_factor *= np.linalg.det(sigma)
    norm_factor = 1.0 / np.sqrt(norm_factor)

    x_mu = np.matrix(x - mu)

    rs = norm_factor * np.exp(-0.5 * x_mu * np.linalg.inv(sigma) * x_mu.T)
    return rs


def _log_likelihood(data, K, mu, sigma, pi):
    score = 0.0
    for n in range(len(data)):
        for k in range(K):
            score += np.log(pi[k] * gaussian(data[n], mu[k], sigma[k]))
    return score


def EM(data, rst, K, threshold):
    converged = False
    mu, sigma, pi = random_parameters(data, K)

    current_log_likelihood = log_likelihood(data, K, mu, sigma, pi)
    max_iter = 100
    for it in range(max_iter):
        print(rst, "       |       ", it, "     |     ", current_log_likelihood[0][0])
        resp = e_step(data, K, mu, sigma, pi)
        mu, sigma, pi = m_step(data, K, resp)

        new_log_likelihood = log_likelihood(data, K, mu, sigma, pi)
        if (abs(new_log_likelihood - current_log_likelihood) < threshold):
            converged = True
            break

        current_log_likelihood = new_log_likelihood

    return current_log_likelihood, mu, sigma, resp, pi


#######################################################################
def assign_clusters(K, resp):
    idvs = len(resp)
    clusters = np.zeros(idvs, dtype=int)

    for i in range(idvs):
        # clusters[i][k] = 0
        clss = 0
        for k in range(K):
            if resp[i][k] > resp[i][clss]:
                clss = k
        clusters[i] = clss

    return clusters


def make_ce_matrix(clusters, ref_clusters, K):
    mat = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            ref_i = np.where(ref_clusters == i)[0]
            clust_j = np.where(clusters == j)[0]
            its = np.intersect1d(ref_i, clust_j)
            mat[i, j] = len(ref_i) + len(clust_j) - 2 * len(its)

    return mat


def read_data_study(file_name):
    with open(file_name) as f:
        data = np.loadtxt(f, delimiter=",", dtype="float", skiprows=0, usecols=(0, 1, 2, 3, 4))

    with open(file_name) as f:
        ref_classes = np.loadtxt(f, delimiter=",", dtype="str", skiprows=0, usecols=[5])
        unique_ref_classes = np.unique(ref_classes)
        ref_clusters = np.argmax(ref_classes[np.newaxis, :] == unique_ref_classes[:, np.newaxis], axis=0)

    return data, ref_clusters


def main():
    print("begining...")
    study_file_name = "study.data"
    # 程序启动4次
    nbr_restarts = 4
    # 设置收敛阈值为 0.001
    threshold = 0.001
    K = 6

    data, ref_clusters = read_data_study(study_file_name)

    mu_lst = []
    sigma_lst = []

    print("#restart | EM iteration | log likelihood")
    print("----------------------------------------")

    max_likelihood_score = float("-inf")
    for rst in range(nbr_restarts):
        log_likelihood, mu, sigma, resp, pi = EM(data, rst, K, threshold)
        clusters = assign_clusters(K, resp)
        print("最终结果")
        print("----------------------------------------")
        print("pi = ", pi)
        print("mu = ", mu)
        print("sigma = ", sigma)
        print("nmi = ", compute_cost(clusters, ref_clusters))
        if log_likelihood > max_likelihood_score:
            max_likelihood_score = log_likelihood
            max_mu, max_sigma, max_resp = mu, sigma, resp

    clusters = assign_clusters(K, max_resp)
    print("最终结果")
    print("----------------------------------------")
    print("pi = ", pi)
    print("mu = ", max_mu)
    print("sigma = ", max_sigma)
    print("nmi = ", compute_cost(clusters, ref_clusters))

if __name__ == '__main__':
    main()